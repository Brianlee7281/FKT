"""
Phase 4 Step 4.4: 포지션 청산 로직 (exit_manager.py)

보유 포지션을 매 틱마다 재평가하여 청산 여부 결정.

청산 트리거 3가지:
  1. Edge 소멸:  EV_adj < θ_exit (0.5¢)
  2. Edge 역전:  P_cons 반대편으로 이동 (진입 논리 역전)
  3. 만기 평가:  경기 종료 n분 전, hold vs exit EV 비교

Kalshi 수수료:
  - 거래 시: fee = 0.07 × P × (1-P)
  - 정산 시: 무료

사용법:
  mgr = ExitManager()
  decisions = mgr.evaluate_all(positions, snapshots, engine_phase, minute)
  for d in decisions:
      if d.action == ExitAction.CLOSE:
          # 청산 주문 제출
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .edge_detector import (
    EdgeDetector, Direction, EnginePhase,
    TAKER_FEE_MULTIPLIER,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 열거형 & 데이터 클래스
# ═══════════════════════════════════════════════════

class ExitAction(Enum):
    """청산 판정."""
    HOLD = "HOLD"       # 유지
    CLOSE = "CLOSE"     # 청산


@dataclass
class OpenPosition:
    """
    보유 중인 포지션.

    Attributes:
        ticker:       Kalshi 마켓 ticker
        direction:    진입 방향 (BUY_YES / BUY_NO)
        contracts:    보유 계약 수
        entry_price:  평균 진입가 (0~1 스케일, 달러)
        p_true_at_entry: 진입 시 P_true
        ev_at_entry:  진입 시 EV_adj
        match_id:     경기 ID (경기별 그룹핑용)
        market_type:  "home_win" / "away_win" / "draw"
    """
    ticker: str = ""
    direction: Direction = Direction.BUY_YES
    contracts: int = 0
    entry_price: float = 0.0
    p_true_at_entry: float = 0.0
    ev_at_entry: float = 0.0
    match_id: str = ""
    market_type: str = ""


@dataclass
class ExitDecision:
    """
    청산 판정 결과.

    Attributes:
        ticker:      마켓 ticker
        action:      HOLD / CLOSE
        trigger:     트리거 이유
        current_ev:  현재 EV
        exit_price:  청산 시 가격 (센트)
        pnl_estimate: 예상 실현 P&L (달러)
    """
    ticker: str = ""
    action: ExitAction = ExitAction.HOLD
    trigger: str = ""
    current_ev: float = 0.0
    exit_price_cents: int = 0
    pnl_estimate: float = 0.0


# ═══════════════════════════════════════════════════
# Exit Manager
# ═══════════════════════════════════════════════════

class ExitManager:
    """
    포지션 청산 관리자.

    매 틱마다 모든 보유 포지션을 재평가하여
    3가지 트리거 중 하나라도 발동하면 청산 시그널 발행.

    Args:
        exit_threshold:    Edge 소멸 임계값 (달러, 기본 0.005 = 0.5¢)
        entry_threshold:   진입 시 사용한 θ_entry (역전 판단용, 기본 0.02)
        expiry_minutes:    만기 평가 시작 시점 (종료 전 n분, 기본 3)
        fee_multiplier:    수수료 계수 (기본 0.07 = taker)
        match_duration:    정규 시간 (분, 기본 90)
    """

    def __init__(
        self,
        exit_threshold: float = 0.005,
        entry_threshold: float = 0.02,
        expiry_minutes: int = 3,
        fee_multiplier: float = TAKER_FEE_MULTIPLIER,
        match_duration: int = 90,
    ):
        self.exit_threshold = exit_threshold
        self.entry_threshold = entry_threshold
        self.expiry_minutes = expiry_minutes
        self.fee_multiplier = fee_multiplier
        self.match_duration = match_duration

    # ─── 수수료 계산 ─────────────────────────────

    def _fee(self, price: float) -> float:
        """계약당 수수료 (달러)."""
        return self.fee_multiplier * price * (1.0 - price)

    # ─── 트리거 1: Edge 소멸 ─────────────────────

    def check_edge_erosion(
        self,
        pos: OpenPosition,
        p_true_cons: float,
        market_price: float,
    ) -> Optional[str]:
        """
        Edge 소멸 체크.

        현재 EV가 θ_exit 미만이면 청산.

        BUY_YES: EV = P_cons - P_ask - fee
        BUY_NO:  EV = (1-P_cons) - no_ask - fee
        """
        fee = self._fee(market_price)

        if pos.direction == Direction.BUY_YES:
            ev = p_true_cons - market_price - fee
        else:  # BUY_NO
            ev = (1.0 - p_true_cons) - market_price - fee

        if ev < self.exit_threshold:
            return f"edge_erosion(EV={ev*100:.2f}¢<{self.exit_threshold*100:.1f}¢)"
        return None

    # ─── 트리거 2: Edge 역전 ─────────────────────

    def check_edge_reversal(
        self,
        pos: OpenPosition,
        p_true_cons: float,
        yes_bid_cents: Optional[int],
    ) -> Optional[str]:
        """
        Edge 역전 체크.

        BUY_YES 보유 중: P_cons < P_bid(달러) - θ_entry → 역전
          (모델이 시장보다 θ이상 낮게 평가 = 진입 논리 완전 반전)
        BUY_NO 보유 중: (1-P_cons) < (1-P_ask)(달러) - θ_entry → 역전
          즉, P_cons > P_ask(달러) + θ_entry
        """
        if yes_bid_cents is None:
            return None

        if pos.direction == Direction.BUY_YES:
            bid_price = yes_bid_cents / 100.0
            if p_true_cons < bid_price - self.entry_threshold:
                return (
                    f"edge_reversal(P_cons={p_true_cons:.3f}"
                    f"<bid={bid_price:.2f}-θ={self.entry_threshold:.2f})"
                )
        else:  # BUY_NO
            # yes_bid → yes_ask = (100 - yes_bid) 는 no_bid 쪽
            # No 보유 중 역전 = P_cons가 높아져서 No가 불리해짐
            yes_ask_price = (100 - yes_bid_cents) / 100.0 if yes_bid_cents < 100 else 0.99
            if p_true_cons > yes_ask_price + self.entry_threshold:
                return (
                    f"edge_reversal(P_cons={p_true_cons:.3f}"
                    f">ask={yes_ask_price:.2f}+θ={self.entry_threshold:.2f})"
                )
        return None

    # ─── 트리거 3: 만기 평가 ─────────────────────

    def check_expiry_evaluation(
        self,
        pos: OpenPosition,
        p_true_cons: float,
        yes_bid_cents: Optional[int],
        minute: int,
    ) -> Optional[str]:
        """
        만기 평가 (경기 종료 n분 전).

        E[Hold] = P_cons × (1 - entry) - (1-P_cons) × entry
                (정산 시 수수료 없음)
        E[Exit] = (bid - entry) - fee_exit  (이익일 때)
                = (bid - entry)             (손실일 때, 수수료 0)

        E[Hold] > E[Exit] → 만기 보유
        그렇지 않으면 → 청산
        """
        if minute < self.match_duration - self.expiry_minutes:
            return None  # 아직 만기 평가 시점이 아님

        if yes_bid_cents is None:
            return None

        entry = pos.entry_price

        if pos.direction == Direction.BUY_YES:
            # Hold to expiry: Yes 정산
            e_hold = p_true_cons * (1.0 - entry) - (1.0 - p_true_cons) * entry
            # Exit now: sell at bid
            bid_price = yes_bid_cents / 100.0
            exit_pnl = bid_price - entry
            fee_exit = self._fee(bid_price) if exit_pnl > 0 else 0.0
            e_exit = exit_pnl - fee_exit
        else:  # BUY_NO
            # Hold: No 정산, No의 진짜 확률 = 1 - P_cons
            e_hold = (1.0 - p_true_cons) * (1.0 - entry) - p_true_cons * entry
            # Exit: sell No at no_bid = 1 - yes_ask
            # But we need no_bid. Approximate: no_bid ≈ 1 - (yes_ask)
            # For simplicity, use the yes_bid to estimate no market
            no_bid_price = (100 - yes_bid_cents) / 100.0 if yes_bid_cents < 100 else 0.01
            exit_pnl = no_bid_price - entry
            fee_exit = self._fee(no_bid_price) if exit_pnl > 0 else 0.0
            e_exit = exit_pnl - fee_exit

        if e_hold >= e_exit:
            return None  # 만기 보유가 유리

        return (
            f"expiry_eval(min={minute},E[hold]={e_hold*100:.2f}¢"
            f",E[exit]={e_exit*100:.2f}¢)"
        )

    # ─── 단일 포지션 평가 ────────────────────────

    def evaluate_position(
        self,
        pos: OpenPosition,
        p_true: float,
        sigma_mc: float,
        yes_bid_cents: Optional[int],
        yes_ask_cents: Optional[int],
        minute: int = 0,
        engine_phase: EnginePhase = EnginePhase.FIRST_HALF,
        z: float = 1.645,
    ) -> ExitDecision:
        """
        단일 포지션 청산 여부 판정.

        트리거 우선순위: 역전 > 소멸 > 만기
        (역전이 가장 긴급)
        """
        decision = ExitDecision(ticker=pos.ticker)

        # 보수적 확률
        p_cons = max(0.001, min(0.999, p_true - z * sigma_mc))

        # 현재 시장 가격 (exit 방향)
        if pos.direction == Direction.BUY_YES:
            market_price = (yes_ask_cents / 100.0) if yes_ask_cents else None
        else:
            market_price = ((100 - yes_bid_cents) / 100.0) if yes_bid_cents else None

        # ── 경기 종료 → 무조건 만기 보유 ────────
        if engine_phase == EnginePhase.FINISHED:
            decision.trigger = "match_finished(hold_to_settlement)"
            return decision

        # ── 하프타임 → 청산 안 함 (주문 불가) ────
        if engine_phase == EnginePhase.HALFTIME:
            decision.trigger = "halftime(no_trading)"
            return decision

        # ── 트리거 2: Edge 역전 (최우선) ─────────
        trigger = self.check_edge_reversal(pos, p_cons, yes_bid_cents)
        if trigger:
            exit_cents = yes_bid_cents if pos.direction == Direction.BUY_YES else (100 - yes_bid_cents)
            decision.action = ExitAction.CLOSE
            decision.trigger = trigger
            decision.exit_price_cents = exit_cents or 0
            decision.pnl_estimate = self._estimate_pnl(pos, exit_cents)
            return decision

        # ── 트리거 1: Edge 소멸 ──────────────────
        if market_price is not None:
            trigger = self.check_edge_erosion(pos, p_cons, market_price)
            if trigger:
                exit_cents = yes_bid_cents if pos.direction == Direction.BUY_YES else (100 - yes_bid_cents)
                decision.action = ExitAction.CLOSE
                decision.trigger = trigger
                decision.exit_price_cents = exit_cents or 0
                decision.pnl_estimate = self._estimate_pnl(pos, exit_cents)
                # 현재 EV 기록
                fee = self._fee(market_price)
                if pos.direction == Direction.BUY_YES:
                    decision.current_ev = p_cons - market_price - fee
                else:
                    decision.current_ev = (1.0 - p_cons) - market_price - fee
                return decision

        # ── 트리거 3: 만기 평가 ──────────────────
        trigger = self.check_expiry_evaluation(pos, p_cons, yes_bid_cents, minute)
        if trigger:
            exit_cents = yes_bid_cents if pos.direction == Direction.BUY_YES else (100 - yes_bid_cents)
            decision.action = ExitAction.CLOSE
            decision.trigger = trigger
            decision.exit_price_cents = exit_cents or 0
            decision.pnl_estimate = self._estimate_pnl(pos, exit_cents)
            return decision

        return decision  # HOLD

    # ─── 일괄 평가 ───────────────────────────────

    def evaluate_all(
        self,
        positions: List[OpenPosition],
        market_data: Dict[str, dict],
        engine_phase: EnginePhase = EnginePhase.FIRST_HALF,
        minute: int = 0,
        z: float = 1.645,
    ) -> List[ExitDecision]:
        """
        모든 보유 포지션 일괄 평가.

        Args:
            positions:   보유 포지션 목록
            market_data: {ticker: {p_true, sigma_mc, yes_bid_cents, yes_ask_cents}}
            engine_phase: 현재 엔진 상태
            minute:      현재 경기 시간 (분)
            z:           보수성 계수

        Returns:
            ExitDecision 목록
        """
        decisions = []
        for pos in positions:
            data = market_data.get(pos.ticker, {})
            dec = self.evaluate_position(
                pos=pos,
                p_true=data.get("p_true", pos.p_true_at_entry),
                sigma_mc=data.get("sigma_mc", 0.0),
                yes_bid_cents=data.get("yes_bid_cents"),
                yes_ask_cents=data.get("yes_ask_cents"),
                minute=minute,
                engine_phase=engine_phase,
                z=z,
            )
            decisions.append(dec)
        return decisions

    # ─── P&L 추정 ────────────────────────────────

    def _estimate_pnl(self, pos: OpenPosition, exit_cents: Optional[int]) -> float:
        """
        예상 실현 P&L (달러).

        P&L = contracts × (exit_price - entry_price) - fee
        """
        if exit_cents is None:
            return 0.0
        exit_price = exit_cents / 100.0
        raw_pnl = pos.contracts * (exit_price - pos.entry_price)
        # 이익일 때만 수수료 (Kalshi 정산은 무료지만 중간 매도는 수수료 부과)
        fee = self._fee(exit_price) * pos.contracts if raw_pnl > 0 else 0.0
        return raw_pnl - fee

    # ─── 유틸리티 ────────────────────────────────

    @staticmethod
    def format_decision(d: ExitDecision) -> str:
        """사람이 읽을 수 있는 포맷."""
        if d.action == ExitAction.HOLD:
            reason = d.trigger if d.trigger else "no_trigger"
            return f"HOLD [{d.ticker}] {reason}"
        return (
            f"CLOSE [{d.ticker}] {d.trigger}  "
            f"exit@{d.exit_price_cents}¢  "
            f"est_pnl=${d.pnl_estimate:.2f}"
        )

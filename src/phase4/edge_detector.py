"""
Phase 4 Step 4.2: Edge Detector (edge_detector.py)

Phase 3의 P_true와 Kalshi 호가창의 P_kalshi를 비교하여
수수료를 감안한 양의 기댓값(Positive EV)이 있는지 판별.

핵심 공식:
  1. P_cons = P_true - z × σ_MC        (보수적 하한)
  2. fee = multiplier × P × (1-P)       (Kalshi 수수료)
  3. EV_buy_yes = P_cons - P_ask - fee  (Yes 매수 EV)
  4. EV_buy_no  = (1-P_cons) - P_no_ask - fee_no  (No 매수 EV)

시그널: BUY_YES / BUY_NO / HOLD

Kalshi 수수료 구조 (2025~2026):
  - Taker: roundup(0.07 × C × P × (1-P))
  - Maker: roundup(0.0175 × C × P × (1-P))
  - 거래 시 부과, 정산 시 무료
  - P*(1-P) → 50¢ 근처에서 최대, 양 극단에서 최소
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 상수 & 열거형
# ═══════════════════════════════════════════════════

class Direction(Enum):
    """거래 방향."""
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


class EnginePhase(Enum):
    """Phase 3 엔진 상태."""
    FIRST_HALF = "FIRST_HALF"
    HALFTIME = "HALFTIME"
    SECOND_HALF = "SECOND_HALF"
    FINISHED = "FINISHED"


# Kalshi 수수료 계수
TAKER_FEE_MULTIPLIER = 0.07
MAKER_FEE_MULTIPLIER = 0.0175


# ═══════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════

@dataclass
class EdgeSignal:
    """
    엣지 탐색 결과.

    모든 확률/가격은 0~1 스케일 (예: 0.45 = 45¢).
    EV는 달러 단위 (예: 0.03 = 3¢).
    """
    direction: Direction = Direction.HOLD
    ev_adj: float = 0.0          # 수수료 반영 기댓값 (달러)
    p_true: float = 0.0          # 모델 원시 확률
    p_true_cons: float = 0.0     # 보수적 확률
    p_kalshi: float = 0.0        # 시장 가격 (거래 상대방)
    sigma_mc: float = 0.0        # MC 표준오차
    fee_per_contract: float = 0.0  # 계약당 수수료 (달러)
    ticker: str = ""
    market_type: str = ""        # "home_win" / "away_win" / "draw"
    reason: str = ""             # HOLD 사유

    @property
    def ev_cents(self) -> float:
        """EV를 센트로."""
        return self.ev_adj * 100

    @property
    def edge_pct(self) -> float:
        """P_cons - P_kalshi 차이 (%)."""
        if self.direction == Direction.BUY_YES:
            return (self.p_true_cons - self.p_kalshi) * 100
        elif self.direction == Direction.BUY_NO:
            return ((1 - self.p_true_cons) - self.p_kalshi) * 100
        return 0.0


@dataclass
class MarketSnapshot:
    """
    단일 마켓의 현재 상태.

    Phase 3 엔진 + Kalshi 호가창에서 수집.
    """
    ticker: str = ""
    market_type: str = ""        # "home_win" / "away_win" / "draw"
    p_true: float = 0.0          # Phase 3 P_true
    sigma_mc: float = 0.0        # Phase 3 σ_MC
    yes_ask_cents: Optional[int] = None   # Yes 매수가 (센트)
    yes_bid_cents: Optional[int] = None   # Yes 매도가 (센트)
    yes_depth: float = 0.0       # Yes 쪽 총 물량
    no_depth: float = 0.0        # No 쪽 총 물량


# ═══════════════════════════════════════════════════
# Edge Detector
# ═══════════════════════════════════════════════════

class EdgeDetector:
    """
    Fee-Adjusted Edge 판별기.

    Phase 3의 P_true와 Kalshi 호가창을 비교하여
    양방향(Buy Yes / Buy No) 기댓값을 계산하고
    필터 조건을 적용하여 시그널 발행.

    Args:
        z:               보수성 계수 (기본 1.645 = 90% 보수적)
        entry_threshold:  최소 엣지 (달러, 기본 0.02 = 2¢)
        fee_multiplier:   수수료 계수 (기본 0.07 = taker)
        min_depth:        최소 유동성 (계약 수, 기본 20)
    """

    def __init__(
        self,
        z: float = 1.645,
        entry_threshold: float = 0.02,
        fee_multiplier: float = TAKER_FEE_MULTIPLIER,
        min_depth: float = 20.0,
    ):
        self.z = z
        self.entry_threshold = entry_threshold
        self.fee_multiplier = fee_multiplier
        self.min_depth = min_depth

    # ─── 핵심 계산 ─────────────────────────────────

    def conservative_p(self, p_true: float, sigma_mc: float) -> float:
        """
        보수적 P_true 하한.

        P_cons = P_true - z × σ_MC
        [0.001, 0.999] 범위로 클램핑.
        """
        p_cons = p_true - self.z * sigma_mc
        return max(0.001, min(0.999, p_cons))

    def fee_per_contract(self, price: float) -> float:
        """
        Kalshi 계약당 수수료 (달러).

        fee = multiplier × P × (1-P)
        P는 0~1 스케일 (달러).

        예: taker, 45¢ → 0.07 × 0.45 × 0.55 = 0.01733
        """
        return self.fee_multiplier * price * (1.0 - price)

    def ev_buy_yes(self, p_cons: float, ask_price: float) -> float:
        """
        Yes 매수 기댓값 (달러).

        Cost = P_ask + fee  (거래 시 부과)
        Win:  $1 수령 → net = 1 - P_ask - fee
        Lose: 전액 손실 → net = -(P_ask + fee)

        EV = P_cons × (1 - P_ask - fee) - (1 - P_cons) × (P_ask + fee)
           = P_cons - P_ask - fee
        """
        fee = self.fee_per_contract(ask_price)
        return p_cons - ask_price - fee

    def ev_buy_no(self, p_cons: float, no_ask_price: float) -> float:
        """
        No 매수 기댓값 (달러).

        No의 진짜 확률 = 1 - P_cons
        No의 시장 가격 = no_ask_price (= 1 - yes_bid / 100)

        EV = (1 - P_cons) - no_ask_price - fee
        """
        fee = self.fee_per_contract(no_ask_price)
        return (1.0 - p_cons) - no_ask_price - fee

    # ─── 단일 마켓 평가 ───────────────────────────

    def evaluate_market(
        self,
        snapshot: MarketSnapshot,
        cooldown: bool = False,
        ob_freeze: bool = False,
        engine_phase: EnginePhase = EnginePhase.FIRST_HALF,
    ) -> EdgeSignal:
        """
        단일 마켓에 대해 양방향 엣지를 평가.

        필터 순서:
          1. engine_phase 활성?
          2. cooldown / ob_freeze?
          3. 유동성 충분?
          4. 호가 존재?
          5. EV > threshold?

        Returns:
            EdgeSignal (BUY_YES / BUY_NO / HOLD)
        """
        base = EdgeSignal(
            ticker=snapshot.ticker,
            market_type=snapshot.market_type,
            p_true=snapshot.p_true,
            sigma_mc=snapshot.sigma_mc,
        )

        # ── 필터 1: 엔진 상태 ────────────────────
        active_phases = {EnginePhase.FIRST_HALF, EnginePhase.SECOND_HALF}
        if engine_phase not in active_phases:
            base.reason = f"engine_phase={engine_phase.value}"
            return base

        # ── 필터 2: 쿨다운 / OB 동결 ────────────
        if cooldown:
            base.reason = "cooldown"
            return base
        if ob_freeze:
            base.reason = "ob_freeze"
            return base

        # ── 필터 3: 유동성 ────────────────────────
        if snapshot.yes_depth < self.min_depth or snapshot.no_depth < self.min_depth:
            base.reason = f"low_liquidity(yes={snapshot.yes_depth:.0f},no={snapshot.no_depth:.0f})"
            return base

        # ── 보수적 확률 ──────────────────────────
        p_cons = self.conservative_p(snapshot.p_true, snapshot.sigma_mc)
        base.p_true_cons = p_cons

        # ── Buy Yes 평가 ─────────────────────────
        ev_yes = None
        if snapshot.yes_ask_cents is not None and snapshot.yes_ask_cents > 0:
            ask_price = snapshot.yes_ask_cents / 100.0
            ev_yes = self.ev_buy_yes(p_cons, ask_price)

        # ── Buy No 평가 ──────────────────────────
        ev_no = None
        if snapshot.yes_bid_cents is not None and snapshot.yes_bid_cents > 0:
            no_ask_price = (100 - snapshot.yes_bid_cents) / 100.0
            ev_no = self.ev_buy_no(p_cons, no_ask_price)

        # ── 양방향 중 더 큰 쪽 선택 ──────────────
        best_dir = Direction.HOLD
        best_ev = 0.0
        best_price = 0.0
        best_fee = 0.0

        if ev_yes is not None and ev_yes > self.entry_threshold:
            best_dir = Direction.BUY_YES
            best_ev = ev_yes
            best_price = snapshot.yes_ask_cents / 100.0
            best_fee = self.fee_per_contract(best_price)

        if ev_no is not None and ev_no > self.entry_threshold:
            if ev_no > best_ev:
                best_dir = Direction.BUY_NO
                best_ev = ev_no
                best_price = (100 - snapshot.yes_bid_cents) / 100.0
                best_fee = self.fee_per_contract(best_price)

        if best_dir == Direction.HOLD:
            ev_y_str = f"{ev_yes*100:.2f}¢" if ev_yes is not None else "N/A"
            ev_n_str = f"{ev_no*100:.2f}¢" if ev_no is not None else "N/A"
            base.reason = f"ev_below_threshold(yes={ev_y_str},no={ev_n_str},θ={self.entry_threshold*100:.1f}¢)"
            return base

        return EdgeSignal(
            direction=best_dir,
            ev_adj=best_ev,
            p_true=snapshot.p_true,
            p_true_cons=p_cons,
            p_kalshi=best_price,
            sigma_mc=snapshot.sigma_mc,
            fee_per_contract=best_fee,
            ticker=snapshot.ticker,
            market_type=snapshot.market_type,
        )

    # ─── 복수 마켓 일괄 평가 ──────────────────────

    def scan_markets(
        self,
        snapshots: List[MarketSnapshot],
        cooldown: bool = False,
        ob_freeze: bool = False,
        engine_phase: EnginePhase = EnginePhase.FIRST_HALF,
    ) -> List[EdgeSignal]:
        """
        여러 마켓을 동시에 평가하여 시그널 목록 반환.

        같은 경기의 home_win / away_win / draw 3개 마켓을
        동시에 평가. EV > 0인 것만 반환.
        """
        signals = []
        for snap in snapshots:
            sig = self.evaluate_market(snap, cooldown, ob_freeze, engine_phase)
            if sig.direction != Direction.HOLD:
                signals.append(sig)
        signals.sort(key=lambda s: s.ev_adj, reverse=True)
        return signals

    # ─── 유틸리티 ────────────────────────────────

    @staticmethod
    def format_signal(sig: EdgeSignal) -> str:
        """시그널을 사람이 읽을 수 있는 포맷으로."""
        if sig.direction == Direction.HOLD:
            return f"HOLD [{sig.ticker}] {sig.reason}"
        return (
            f"{sig.direction.value} [{sig.ticker}] "
            f"EV={sig.ev_cents:+.2f}¢  "
            f"P_cons={sig.p_true_cons:.3f} vs P_mkt={sig.p_kalshi:.3f}  "
            f"edge={sig.edge_pct:+.1f}%  "
            f"fee={sig.fee_per_contract*100:.2f}¢  "
            f"σ={sig.sigma_mc:.4f}"
        )

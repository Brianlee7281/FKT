"""
Phase 4 Step 4.5: 주문 실행 & 리스크 관리 (execution_engine.py)

Phase 4 전체 파이프라인의 오케스트레이터.
포트폴리오 상태 관리, 주문 실행, 거래 로그 기록.

역할:
  1. 포트폴리오 상태 추적 (잔고, 포지션, 경기별/전체 노출)
  2. 진입 주문: EdgeSignal → PositionSizer → Limit Order 제출
  3. 청산 주문: ExitDecision → 공격적 지정가 제출
  4. 거래 로그: 모든 주문의 전 필드 기록 (사후 분석용)
  5. Paper Trading 모드: 실제 API 호출 없이 가상 체결

사용법:
  engine = ExecutionEngine(kalshi_client, bankroll=10000, paper=True)
  engine.process_entry(signal, sizing_result)
  engine.process_exit(decision)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .edge_detector import Direction, EnginePhase, EdgeSignal, TAKER_FEE_MULTIPLIER
from .position_sizer import PositionSizer, SizingResult
from .exit_manager import ExitAction, ExitDecision, OpenPosition

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 열거형 & 데이터 클래스
# ═══════════════════════════════════════════════════

class OrderType(Enum):
    """주문 유형."""
    ENTRY = "ENTRY"
    EXIT_EDGE_EROSION = "EXIT_EDGE_EROSION"
    EXIT_EDGE_REVERSAL = "EXIT_EDGE_REVERSAL"
    EXIT_EXPIRY_EVAL = "EXIT_EXPIRY_EVAL"
    EXIT_MANUAL = "EXIT_MANUAL"


class FillStatus(Enum):
    """체결 상태."""
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    UNFILLED = "UNFILLED"
    PAPER = "PAPER"          # 종이 거래


@dataclass
class TradeRecord:
    """
    거래 로그 레코드.

    Step 4.6 사후 분석의 핵심 입력.
    """
    timestamp: str = ""
    match_id: str = ""
    ticker: str = ""
    direction: str = ""          # BUY_YES / BUY_NO / SELL_YES / SELL_NO
    order_type: str = ""         # ENTRY / EXIT_*
    quantity_ordered: int = 0
    quantity_filled: int = 0
    limit_price_cents: int = 0
    fill_price_cents: int = 0
    p_true: float = 0.0
    p_true_cons: float = 0.0
    p_kalshi: float = 0.0
    ev_adj: float = 0.0
    f_kelly: float = 0.0
    f_invest: float = 0.0
    sigma_mc: float = 0.0
    fee_per_contract: float = 0.0
    fill_status: str = ""        # FILLED / PARTIAL / UNFILLED / PAPER
    bankroll_before: float = 0.0
    bankroll_after: float = 0.0
    pnl: float = 0.0
    notes: str = ""


@dataclass
class PortfolioState:
    """
    실시간 포트폴리오 상태.

    모든 리스크 계산의 기반.
    """
    bankroll: float = 0.0                          # 현금 잔고 (달러)
    positions: Dict[str, OpenPosition] = field(default_factory=dict)
    # 경기별 노출: {match_id: 총 노출액}
    match_exposure: Dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0
    trade_log: List[TradeRecord] = field(default_factory=list)
    peak_bankroll: float = 0.0                     # 최고점 (drawdown용)


# ═══════════════════════════════════════════════════
# Execution Engine
# ═══════════════════════════════════════════════════

class ExecutionEngine:
    """
    Phase 4 주문 실행 엔진.

    Args:
        kalshi_client:   KalshiClient 인스턴스 (paper=True면 None 가능)
        initial_bankroll: 초기 잔고 (달러)
        paper:           종이 거래 모드
        order_timeout:   주문 타임아웃 (초, 기본 5)
        aggressive_offset: 진입 시 ask+n¢ (기본 1)
        exit_offset:     청산 시 bid-n¢ (기본 1)
    """

    def __init__(
        self,
        kalshi_client: Any = None,
        initial_bankroll: float = 10000.0,
        paper: bool = True,
        order_timeout: float = 5.0,
        aggressive_offset: int = 1,
        exit_offset: int = 1,
    ):
        self.client = kalshi_client
        self.paper = paper
        self.order_timeout = order_timeout
        self.aggressive_offset = aggressive_offset
        self.exit_offset = exit_offset

        self.portfolio = PortfolioState(
            bankroll=initial_bankroll,
            peak_bankroll=initial_bankroll,
        )

    # ─── 포트폴리오 조회 ─────────────────────────

    @property
    def bankroll(self) -> float:
        return self.portfolio.bankroll

    @property
    def positions(self) -> Dict[str, OpenPosition]:
        return self.portfolio.positions

    @property
    def trade_log(self) -> List[TradeRecord]:
        return self.portfolio.trade_log

    def get_match_exposure(self, match_id: str) -> float:
        """특정 경기의 현재 노출액."""
        return self.portfolio.match_exposure.get(match_id, 0.0)

    def get_total_exposure(self) -> float:
        """전체 포트폴리오 노출액."""
        return self.portfolio.total_exposure

    def get_drawdown(self) -> float:
        """현재 드로다운 (%)."""
        if self.portfolio.peak_bankroll <= 0:
            return 0.0
        return (1.0 - self.portfolio.bankroll / self.portfolio.peak_bankroll) * 100

    # ─── 진입 주문 처리 ──────────────────────────

    def process_entry(
        self,
        signal: EdgeSignal,
        sizing: SizingResult,
        match_id: str = "",
    ) -> TradeRecord:
        """
        진입 주문 실행.

        1. 주문 가격 결정 (ask + offset)
        2. API 제출 (또는 paper 기록)
        3. 포지션/잔고 업데이트
        4. 거래 로그 기록

        Args:
            signal:   EdgeSignal (from edge_detector)
            sizing:   SizingResult (from position_sizer)
            match_id: 경기 ID

        Returns:
            TradeRecord
        """
        if sizing.contracts <= 0:
            return self._log_trade(
                ticker=signal.ticker, direction=signal.direction.value,
                order_type=OrderType.ENTRY.value,
                fill_status=FillStatus.UNFILLED.value,
                notes=f"sizing=0({sizing.reason})",
                signal=signal, sizing=sizing, match_id=match_id,
            )

        # 주문 가격 (센트)
        limit_cents = int(signal.p_kalshi * 100) + self.aggressive_offset

        # 체결 시뮬레이션 or 실제 API
        if self.paper:
            fill_qty = sizing.contracts
            fill_price = int(signal.p_kalshi * 100)  # paper: ask 가격 체결
            fill_status = FillStatus.PAPER
        else:
            fill_qty, fill_price, fill_status = self._execute_live_order(
                ticker=signal.ticker,
                direction=signal.direction,
                count=sizing.contracts,
                limit_cents=limit_cents,
            )

        if fill_qty <= 0:
            return self._log_trade(
                ticker=signal.ticker, direction=signal.direction.value,
                order_type=OrderType.ENTRY.value,
                quantity_ordered=sizing.contracts,
                limit_price_cents=limit_cents,
                fill_status=FillStatus.UNFILLED.value,
                notes="order_unfilled",
                signal=signal, sizing=sizing, match_id=match_id,
            )

        # 포지션 업데이트
        cost = fill_qty * fill_price / 100.0  # 달러
        fee = self._calc_fee(fill_price / 100.0) * fill_qty
        total_cost = cost + fee
        bankroll_before = self.portfolio.bankroll
        self.portfolio.bankroll -= total_cost

        # 포지션 추가/업데이트
        pos_key = signal.ticker
        if pos_key in self.portfolio.positions:
            existing = self.portfolio.positions[pos_key]
            # 평균 진입가 갱신
            total_qty = existing.contracts + fill_qty
            existing.entry_price = (
                (existing.entry_price * existing.contracts + fill_price / 100.0 * fill_qty)
                / total_qty
            )
            existing.contracts = total_qty
        else:
            self.portfolio.positions[pos_key] = OpenPosition(
                ticker=signal.ticker,
                direction=signal.direction,
                contracts=fill_qty,
                entry_price=fill_price / 100.0,
                p_true_at_entry=signal.p_true,
                ev_at_entry=signal.ev_adj,
                match_id=match_id,
                market_type=signal.market_type,
            )

        # 노출 업데이트
        self._update_exposure(match_id, cost)

        # Peak 업데이트
        if self.portfolio.bankroll > self.portfolio.peak_bankroll:
            self.portfolio.peak_bankroll = self.portfolio.bankroll

        return self._log_trade(
            ticker=signal.ticker, direction=signal.direction.value,
            order_type=OrderType.ENTRY.value,
            quantity_ordered=sizing.contracts,
            quantity_filled=fill_qty,
            limit_price_cents=limit_cents,
            fill_price_cents=fill_price,
            fill_status=fill_status.value,
            bankroll_before=bankroll_before,
            bankroll_after=self.portfolio.bankroll,
            pnl=-total_cost,
            signal=signal, sizing=sizing, match_id=match_id,
        )

    # ─── 청산 주문 처리 ──────────────────────────

    def process_exit(
        self,
        decision: ExitDecision,
        pos: OpenPosition,
    ) -> TradeRecord:
        """
        청산 주문 실행.

        1. 공격적 지정가 (bid - offset)
        2. API 제출 (또는 paper 기록)
        3. 포지션/잔고 업데이트
        4. 실현 P&L 계산

        Args:
            decision: ExitDecision (from exit_manager)
            pos:      보유 포지션

        Returns:
            TradeRecord
        """
        if decision.action != ExitAction.CLOSE:
            return self._log_trade(
                ticker=pos.ticker,
                direction=f"SELL_{pos.direction.value.split('_')[1]}",
                order_type="EXIT_HOLD",
                fill_status=FillStatus.UNFILLED.value,
                notes="hold_decision",
            )

        # 청산 가격
        exit_cents = max(1, decision.exit_price_cents - self.exit_offset)

        # 주문 유형 분류
        order_type = self._classify_exit_type(decision.trigger)

        # 체결
        if self.paper:
            fill_qty = pos.contracts
            fill_price = decision.exit_price_cents  # paper: bid 가격 체결
            fill_status = FillStatus.PAPER
        else:
            fill_qty, fill_price, fill_status = self._execute_live_exit(
                ticker=pos.ticker,
                direction=pos.direction,
                count=pos.contracts,
                limit_cents=exit_cents,
            )

        if fill_qty <= 0:
            return self._log_trade(
                ticker=pos.ticker,
                direction=f"SELL_{pos.direction.value.split('_')[1]}",
                order_type=order_type,
                quantity_ordered=pos.contracts,
                limit_price_cents=exit_cents,
                fill_status=FillStatus.UNFILLED.value,
                notes="exit_unfilled",
            )

        # P&L 계산
        fill_price_dollar = fill_price / 100.0
        raw_pnl = fill_qty * (fill_price_dollar - pos.entry_price)
        fee = self._calc_fee(fill_price_dollar) * fill_qty if raw_pnl > 0 else 0.0
        realized_pnl = raw_pnl - fee

        bankroll_before = self.portfolio.bankroll
        # 청산 시: 매도 대금 수령
        self.portfolio.bankroll += fill_qty * fill_price_dollar - fee

        # 포지션 업데이트
        remaining = pos.contracts - fill_qty
        if remaining <= 0:
            self.portfolio.positions.pop(pos.ticker, None)
        else:
            self.portfolio.positions[pos.ticker].contracts = remaining

        # 노출 업데이트
        self._update_exposure(pos.match_id, -fill_qty * pos.entry_price)

        # Peak 업데이트
        if self.portfolio.bankroll > self.portfolio.peak_bankroll:
            self.portfolio.peak_bankroll = self.portfolio.bankroll

        return self._log_trade(
            ticker=pos.ticker,
            direction=f"SELL_{pos.direction.value.split('_')[1]}",
            order_type=order_type,
            quantity_ordered=pos.contracts,
            quantity_filled=fill_qty,
            limit_price_cents=exit_cents,
            fill_price_cents=fill_price,
            fill_status=fill_status.value,
            bankroll_before=bankroll_before,
            bankroll_after=self.portfolio.bankroll,
            pnl=realized_pnl,
            notes=decision.trigger,
        )

    # ─── 경기 종료 정산 ──────────────────────────

    def settle_match(
        self,
        match_id: str,
        outcomes: Dict[str, bool],
    ) -> List[TradeRecord]:
        """
        경기 종료 시 자동 정산.

        Args:
            match_id:  경기 ID
            outcomes:  {ticker: True(Yes승)/False(No승)}

        Returns:
            정산 TradeRecord 목록
        """
        records = []
        tickers_to_remove = []

        for ticker, pos in self.portfolio.positions.items():
            if pos.match_id != match_id:
                continue
            if ticker not in outcomes:
                continue

            won = outcomes[ticker]
            bankroll_before = self.portfolio.bankroll

            if pos.direction == Direction.BUY_YES:
                if won:
                    # Yes 승리: $1 수령
                    payout = pos.contracts * 1.0
                    cost_basis = pos.contracts * pos.entry_price
                    pnl = payout - cost_basis  # 수수료는 이미 진입 시 처리
                else:
                    pnl = -pos.contracts * pos.entry_price
                    payout = 0
            else:  # BUY_NO
                if not won:
                    payout = pos.contracts * 1.0
                    cost_basis = pos.contracts * pos.entry_price
                    pnl = payout - cost_basis
                else:
                    pnl = -pos.contracts * pos.entry_price
                    payout = 0

            self.portfolio.bankroll += payout if payout > 0 else 0

            records.append(self._log_trade(
                ticker=ticker,
                direction=f"SETTLE_{pos.direction.value}",
                order_type="SETTLEMENT",
                quantity_filled=pos.contracts,
                fill_price_cents=100 if won and pos.direction == Direction.BUY_YES else 0,
                fill_status="SETTLED",
                bankroll_before=bankroll_before,
                bankroll_after=self.portfolio.bankroll,
                pnl=pnl,
                notes=f"outcome={'YES' if won else 'NO'}",
                match_id=pos.match_id,
            ))
            tickers_to_remove.append(ticker)

        for t in tickers_to_remove:
            self.portfolio.positions.pop(t, None)

        # 경기별 노출 제거
        self.portfolio.match_exposure.pop(match_id, None)
        self._recalc_total_exposure()

        # Peak 업데이트
        if self.portfolio.bankroll > self.portfolio.peak_bankroll:
            self.portfolio.peak_bankroll = self.portfolio.bankroll

        return records

    # ─── 실시간 리스크 대시보드 ───────────────────

    def risk_dashboard(self) -> Dict[str, Any]:
        """현재 리스크 상태."""
        return {
            "bankroll": round(self.portfolio.bankroll, 2),
            "peak_bankroll": round(self.portfolio.peak_bankroll, 2),
            "drawdown_pct": round(self.get_drawdown(), 2),
            "open_positions": len(self.portfolio.positions),
            "total_exposure": round(self.portfolio.total_exposure, 2),
            "match_exposures": {
                k: round(v, 2) for k, v in self.portfolio.match_exposure.items()
            },
            "total_trades": len(self.portfolio.trade_log),
            "mode": "PAPER" if self.paper else "LIVE",
        }

    # ─── 내부 헬퍼 ───────────────────────────────

    def _calc_fee(self, price: float) -> float:
        """Kalshi taker 수수료."""
        return TAKER_FEE_MULTIPLIER * price * (1.0 - price)

    def _update_exposure(self, match_id: str, delta: float):
        """경기별/전체 노출 업데이트."""
        if not match_id:
            return
        current = self.portfolio.match_exposure.get(match_id, 0.0)
        self.portfolio.match_exposure[match_id] = max(0.0, current + delta)
        self._recalc_total_exposure()

    def _recalc_total_exposure(self):
        """전체 노출 재계산."""
        self.portfolio.total_exposure = sum(self.portfolio.match_exposure.values())

    def _classify_exit_type(self, trigger: str) -> str:
        """트리거 문자열 → OrderType."""
        if "reversal" in trigger:
            return OrderType.EXIT_EDGE_REVERSAL.value
        elif "erosion" in trigger:
            return OrderType.EXIT_EDGE_EROSION.value
        elif "expiry" in trigger:
            return OrderType.EXIT_EXPIRY_EVAL.value
        return OrderType.EXIT_MANUAL.value

    def _log_trade(self, **kwargs) -> TradeRecord:
        """거래 로그 기록."""
        signal = kwargs.pop("signal", None)
        sizing = kwargs.pop("sizing", None)
        match_id = kwargs.pop("match_id", "")

        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            match_id=match_id,
            **kwargs,
        )

        if signal:
            record.p_true = signal.p_true
            record.p_true_cons = signal.p_true_cons
            record.p_kalshi = signal.p_kalshi
            record.ev_adj = signal.ev_adj
            record.sigma_mc = signal.sigma_mc
            record.fee_per_contract = signal.fee_per_contract

        if sizing:
            record.f_kelly = sizing.f_kelly
            record.f_invest = sizing.f_invest

        self.portfolio.trade_log.append(record)
        return record

    # ─── 실제 API 주문 (라이브 모드) ─────────────

    def _execute_live_order(
        self,
        ticker: str,
        direction: Direction,
        count: int,
        limit_cents: int,
    ) -> tuple[int, int, FillStatus]:
        """
        실제 Kalshi API 주문 제출 + 타임아웃 처리.

        Returns:
            (fill_qty, fill_price_cents, fill_status)
        """
        if self.client is None:
            logger.error("Live mode but no client!")
            return 0, 0, FillStatus.UNFILLED

        try:
            side = "yes" if direction == Direction.BUY_YES else "no"
            resp = self.client.buy_yes(ticker, count, limit_cents) \
                if direction == Direction.BUY_YES \
                else self.client.buy_no(ticker, count, limit_cents)

            order_id = resp.order_id if resp else None
            if not order_id:
                return 0, 0, FillStatus.UNFILLED

            # 타임아웃 대기 (동기 — 실전에서는 asyncio 전환)
            deadline = time.time() + self.order_timeout
            while time.time() < deadline:
                time.sleep(0.5)
                # TODO: 주문 상태 확인 API 호출
                # status = self.client.get_order_status(order_id)
                # if status.filled: break

            # 미체결분 취소
            # self.client.cancel_order(order_id)

            # TODO: 실제 체결 수량/가격 조회
            return count, limit_cents, FillStatus.FILLED

        except Exception as e:
            logger.error(f"Live order failed: {e}")
            return 0, 0, FillStatus.UNFILLED

    def _execute_live_exit(
        self,
        ticker: str,
        direction: Direction,
        count: int,
        limit_cents: int,
    ) -> tuple[int, int, FillStatus]:
        """실제 Kalshi API 청산 주문."""
        if self.client is None:
            return 0, 0, FillStatus.UNFILLED

        try:
            # 청산 = 반대 방향 매도
            if direction == Direction.BUY_YES:
                resp = self.client.sell_yes(ticker, count, limit_cents)
            else:
                resp = self.client.sell_no(ticker, count, limit_cents)

            # 타임아웃 + 취소 로직은 진입과 동일
            return count, limit_cents, FillStatus.FILLED

        except Exception as e:
            logger.error(f"Live exit failed: {e}")
            return 0, 0, FillStatus.UNFILLED

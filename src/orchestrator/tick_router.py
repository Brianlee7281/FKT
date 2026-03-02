"""
오케스트레이터 — 틱 라우터 (tick_router.py)

Phase 3의 매 틱 출력(TickSnapshot)을 받아서
Phase 4 파이프라인(엣지 탐색 → 사이징 → 진입/청산)을 실행.

이 모듈이 전체 시스템의 "접합부":
  Phase 3 (확률 엔진)  →  TickRouter  →  Phase 4 (거래 실행)

매 틱 사이클:
  ① TickSnapshot에서 P_true, σ_MC, engine_phase 추출
  ② 3개 시장(home/away/draw)에 대해 EdgeDetector 실행
  ③ 시그널 있으면 → PositionSizer → ExecutionEngine.process_entry()
  ④ 기존 포지션 있으면 → ExitManager → ExecutionEngine.process_exit()

사용법:
  router = TickRouter(execution_engine, edge_detector, sizer, exit_mgr)
  router.on_tick(snapshot, orderbooks, match_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.phase4.edge_detector import (
    EdgeDetector, EdgeSignal, MarketSnapshot,
    Direction, EnginePhase,
)
from src.phase4.position_sizer import PositionSizer, SizingResult
from src.phase4.exit_manager import ExitManager, ExitAction, ExitDecision, OpenPosition
from src.phase4.execution_engine import ExecutionEngine, TradeRecord

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════

@dataclass
class OrderbookSnapshot:
    """단일 마켓의 호가창 스냅샷."""
    ticker: str = ""
    yes_ask_cents: Optional[int] = None
    yes_bid_cents: Optional[int] = None
    yes_depth: int = 0            # ask 쪽 물량
    no_depth: int = 0             # bid 쪽 물량


@dataclass
class TickResult:
    """틱 처리 결과."""
    tick: int = 0
    minute: float = 0.0
    signals_evaluated: int = 0
    entries: List[TradeRecord] = field(default_factory=list)
    exits: List[TradeRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# Phase 3 engine_phase → Phase 4 EnginePhase 변환
PHASE_MAP = {
    "FIRST_HALF": EnginePhase.FIRST_HALF,
    "SECOND_HALF": EnginePhase.SECOND_HALF,
    "HALFTIME": EnginePhase.HALFTIME,
    "FINISHED": EnginePhase.FINISHED,
    "WAITING": EnginePhase.HALFTIME,      # 대기 = 거래 불가
    "EXTRA_TIME": EnginePhase.SECOND_HALF, # 연장 = 2H와 동일 취급
}

# PricingResult 확률 필드 → market_type 매핑
PROB_FIELDS = {
    "home_win": "home_win",
    "away_win": "away_win",
    "draw": "draw",
}


# ═══════════════════════════════════════════════════
# Tick Router
# ═══════════════════════════════════════════════════

class TickRouter:
    """
    매 틱 Phase 3 → Phase 4 파이프라인 실행.

    Args:
        engine:    ExecutionEngine (포트폴리오 + 주문 실행)
        detector:  EdgeDetector (엣지 탐색)
        sizer:     PositionSizer (Kelly + 리스크)
        exit_mgr:  ExitManager (청산 판정)
        entry_cooldown_after_exit: 청산 후 재진입 금지 틱 수 (기본 100 ≈ 5분)
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        detector: EdgeDetector,
        sizer: PositionSizer,
        exit_mgr: ExitManager,
        entry_cooldown_after_exit: int = 100,
    ):
        self.engine = engine
        self.detector = detector
        self.sizer = sizer
        self.exit_mgr = exit_mgr
        self._tick_count = 0
        self.entry_cooldown_after_exit = entry_cooldown_after_exit
        self._exit_cooldown: Dict[str, int] = {}  # ticker → 재진입 허용 틱

    def on_tick(
        self,
        snapshot: Any,
        orderbooks: Dict[str, OrderbookSnapshot],
        kalshi_tickers: Dict[str, str],
        match_id: str = "",
    ) -> TickResult:
        """
        매 틱 호출되는 메인 라우팅 함수.

        Args:
            snapshot:       Phase 3 TickSnapshot
            orderbooks:     {market_type: OrderbookSnapshot}
                           market_type ∈ {"home_win", "away_win", "draw"}
            kalshi_tickers: {market_type: kalshi_ticker}
            match_id:       경기 ID

        Returns:
            TickResult
        """
        self._tick_count += 1
        result = TickResult(tick=self._tick_count, minute=getattr(snapshot, "minute", 0))

        # Phase 3 → Phase 4 변환
        engine_phase = PHASE_MAP.get(
            getattr(snapshot, "engine_phase", ""), EnginePhase.HALFTIME
        )
        orders_allowed = getattr(snapshot, "orders_allowed", False)
        pricing = getattr(snapshot, "pricing", None)

        if pricing is None:
            return result

        sigma_mc = getattr(pricing, "sigma_mc", 0.0)

        # ── ① 기존 포지션 청산 평가 ──────────────
        self._evaluate_exits(
            result, pricing, sigma_mc, orderbooks,
            kalshi_tickers, engine_phase, result.minute,
        )

        # ── ② 신규 진입 탐색 ─────────────────────
        if orders_allowed and engine_phase in (EnginePhase.FIRST_HALF, EnginePhase.SECOND_HALF):
            self._evaluate_entries(
                result, pricing, sigma_mc, orderbooks,
                kalshi_tickers, engine_phase, match_id,
            )

        return result

    # ─── 진입 평가 ───────────────────────────────

    def _evaluate_entries(
        self,
        result: TickResult,
        pricing: Any,
        sigma_mc: float,
        orderbooks: Dict[str, OrderbookSnapshot],
        kalshi_tickers: Dict[str, str],
        engine_phase: EnginePhase,
        match_id: str,
    ):
        """3개 시장 × 양방향 진입 평가."""
        for market_type, prob_field in PROB_FIELDS.items():
            ticker = kalshi_tickers.get(market_type)
            if not ticker:
                continue

            ob = orderbooks.get(market_type)
            if ob is None:
                continue

            p_true = getattr(pricing, prob_field, 0.0)
            if p_true <= 0 or p_true >= 1:
                continue

            # 이미 같은 방향 포지션이 있으면 스킵 (중복 진입 방지)
            if ticker in self.engine.positions:
                continue

            # 재진입 쿨다운 체크
            if ticker in self._exit_cooldown:
                if self._tick_count < self._exit_cooldown[ticker]:
                    continue
                else:
                    del self._exit_cooldown[ticker]  # 쿨다운 만료

            # MarketSnapshot 구성
            ms = MarketSnapshot(
                ticker=ticker,
                market_type=market_type,
                p_true=p_true,
                sigma_mc=sigma_mc,
                yes_ask_cents=ob.yes_ask_cents or 0,
                yes_bid_cents=ob.yes_bid_cents or 0,
                yes_depth=ob.yes_depth,
                no_depth=ob.no_depth,
            )

            # EdgeDetector 평가
            signal = self.detector.evaluate_market(
                ms,
                engine_phase=engine_phase,
                cooldown=False,      # orders_allowed가 이미 cooldown 체크 포함
                ob_freeze=False,
            )
            result.signals_evaluated += 1

            if signal.direction == Direction.HOLD:
                continue

            # PositionSizer
            sizing = self.sizer.size(
                ev_adj=signal.ev_adj,
                p_kalshi=signal.p_kalshi,
                bankroll=self.engine.bankroll,
                match_exposure=self.engine.get_match_exposure(match_id),
                total_exposure=self.engine.get_total_exposure(),
            )

            if sizing.contracts <= 0:
                continue

            # 주문 실행
            try:
                record = self.engine.process_entry(signal, sizing, match_id)
                result.entries.append(record)
                # entry_tick 설정 (min_hold 체크용)
                if ticker in self.engine.positions:
                    self.engine.positions[ticker].entry_tick = self._tick_count
                logger.info(
                    f"ENTRY [{ticker}] {signal.direction.value} "
                    f"{sizing.contracts}x EV={signal.ev_adj*100:.1f}¢"
                )
            except Exception as e:
                result.errors.append(f"entry_error({ticker}): {e}")
                logger.error(f"Entry error: {e}")

    # ─── 청산 평가 ───────────────────────────────

    def _evaluate_exits(
        self,
        result: TickResult,
        pricing: Any,
        sigma_mc: float,
        orderbooks: Dict[str, OrderbookSnapshot],
        kalshi_tickers: Dict[str, str],
        engine_phase: EnginePhase,
        minute: float,
    ):
        """기존 포지션 청산 평가."""
        # 현재 경기의 포지션만 필터
        active_tickers = set(kalshi_tickers.values())
        positions_to_check = [
            (ticker, pos)
            for ticker, pos in self.engine.positions.items()
            if ticker in active_tickers
        ]

        for ticker, pos in positions_to_check:
            # market_type 역매핑
            market_type = pos.market_type
            prob_field = PROB_FIELDS.get(market_type)
            if not prob_field:
                continue

            ob = orderbooks.get(market_type)
            p_true = getattr(pricing, prob_field, 0.0)

            decision = self.exit_mgr.evaluate_position(
                pos=pos,
                p_true=p_true,
                sigma_mc=sigma_mc,
                yes_bid_cents=ob.yes_bid_cents if ob else None,
                yes_ask_cents=ob.yes_ask_cents if ob else None,
                minute=int(minute),
                engine_phase=engine_phase,
                current_tick=self._tick_count,
            )

            if decision.action == ExitAction.CLOSE:
                try:
                    record = self.engine.process_exit(decision, pos)
                    result.exits.append(record)
                    # 재진입 쿨다운 설정
                    self._exit_cooldown[ticker] = (
                        self._tick_count + self.entry_cooldown_after_exit
                    )
                    logger.info(
                        f"EXIT [{ticker}] {decision.trigger} "
                        f"pnl=${record.pnl:+.2f} "
                        f"(re-entry blocked for {self.entry_cooldown_after_exit} ticks)"
                    )
                except Exception as e:
                    result.errors.append(f"exit_error({ticker}): {e}")
                    logger.error(f"Exit error: {e}")

    # ─── 유틸리티 ────────────────────────────────

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @staticmethod
    def format_result(r: TickResult) -> str:
        """틱 결과 포맷."""
        parts = [f"tick={r.tick} min={r.minute:.1f}"]
        parts.append(f"eval={r.signals_evaluated}")
        if r.entries:
            parts.append(f"entries={len(r.entries)}")
        if r.exits:
            parts.append(f"exits={len(r.exits)}")
        if r.errors:
            parts.append(f"errors={len(r.errors)}")
        return " | ".join(parts)
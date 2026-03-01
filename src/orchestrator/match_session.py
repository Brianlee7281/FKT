"""
오케스트레이터 — 매치 세션 (match_session.py)

단일 경기의 전체 라이프사이클 관리:
  초기화 → 라이브 거래 → 경기 종료 → 정산 → 사후 분석

Phase 3 엔진의 on_tick 콜백으로 TickRouter를 연결하고,
경기 종료 시 자동 정산 및 PostMatchAnalyzer 호출.

사용법:
  session = MatchSession(
      match_id="GS_12345",
      kalshi_tickers={"home_win": "KX...", "away_win": "KX...", "draw": "KX..."},
      tick_router=router,
  )
  # Phase 3 엔진의 on_tick 콜백으로 등록
  engine = LiveTradingEngine(on_tick=session.on_tick, ...)
  await engine.run(match_id)
  # 경기 종료 후
  report = session.finalize(outcomes={"KX...-HW": True, ...})
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.tick_router import TickRouter, TickResult, OrderbookSnapshot
from src.phase4.post_match import PostMatchAnalyzer, TradeOutcome, MatchReport

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════

class SessionState:
    """매치 세션 상태."""
    WAITING = "WAITING"
    LIVE = "LIVE"
    FINISHED = "FINISHED"
    SETTLED = "SETTLED"


@dataclass
class SessionStats:
    """세션 통계."""
    total_ticks: int = 0
    total_entries: int = 0
    total_exits: int = 0
    total_signals_evaluated: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


# ═══════════════════════════════════════════════════
# Match Session
# ═══════════════════════════════════════════════════

class MatchSession:
    """
    단일 경기 거래 세션.

    Phase 3 엔진과 Phase 4 파이프라인을 연결하는 어댑터.

    Args:
        match_id:       Goalserve fixture ID
        kalshi_tickers: {market_type: kalshi_ticker}
        tick_router:    TickRouter 인스턴스
        orderbook_fn:   호가창 조회 함수 (Optional)
                       signature: (ticker) → OrderbookSnapshot
    """

    def __init__(
        self,
        match_id: str,
        kalshi_tickers: Dict[str, str],
        tick_router: TickRouter,
        orderbook_fn: Optional[Callable] = None,
    ):
        self.match_id = match_id
        self.kalshi_tickers = kalshi_tickers
        self.tick_router = tick_router
        self.orderbook_fn = orderbook_fn

        self.state = SessionState.WAITING
        self.stats = SessionStats()
        self.tick_results: List[TickResult] = []

        # 최근 호가창 캐시 (orderbook_fn 없을 때 외부에서 업데이트)
        self._orderbook_cache: Dict[str, OrderbookSnapshot] = {}

    # ─── Phase 3 on_tick 콜백 ────────────────────

    def on_tick(self, snapshot: Any) -> Optional[TickResult]:
        """
        Phase 3 엔진의 on_tick 콜백.

        LiveTradingEngine(on_tick=session.on_tick) 형태로 등록.

        Args:
            snapshot: Phase 3 TickSnapshot

        Returns:
            TickResult (또는 None)
        """
        if self.state == SessionState.SETTLED:
            return None

        if self.state == SessionState.WAITING:
            self.state = SessionState.LIVE
            self.stats.start_time = time.time()

        # 호가창 조회
        orderbooks = self._get_orderbooks()

        # TickRouter 실행
        result = self.tick_router.on_tick(
            snapshot=snapshot,
            orderbooks=orderbooks,
            kalshi_tickers=self.kalshi_tickers,
            match_id=self.match_id,
        )

        # 통계 업데이트
        self.stats.total_ticks += 1
        self.stats.total_entries += len(result.entries)
        self.stats.total_exits += len(result.exits)
        self.stats.total_signals_evaluated += result.signals_evaluated
        self.stats.errors.extend(result.errors)
        self.tick_results.append(result)

        # 경기 종료 감지
        engine_phase = getattr(snapshot, "engine_phase", "")
        if engine_phase == "FINISHED":
            self.state = SessionState.FINISHED
            self.stats.end_time = time.time()

        return result

    # ─── 호가창 관리 ─────────────────────────────

    def update_orderbook(self, market_type: str, ob: OrderbookSnapshot):
        """외부에서 호가창 업데이트 (REST 폴링 등)."""
        self._orderbook_cache[market_type] = ob

    def _get_orderbooks(self) -> Dict[str, OrderbookSnapshot]:
        """현재 호가창 반환."""
        if self.orderbook_fn:
            orderbooks = {}
            for market_type, ticker in self.kalshi_tickers.items():
                try:
                    ob = self.orderbook_fn(ticker)
                    if ob:
                        orderbooks[market_type] = ob
                except Exception as e:
                    logger.debug(f"Orderbook fetch failed for {ticker}: {e}")
            return orderbooks
        return self._orderbook_cache

    # ─── 경기 종료 정산 ──────────────────────────

    def finalize(
        self,
        outcomes: Dict[str, bool],
        analyzer: Optional[PostMatchAnalyzer] = None,
    ) -> Optional[MatchReport]:
        """
        경기 종료 후 정산 및 사후 분석.

        Args:
            outcomes:  {kalshi_ticker: True(Yes승)/False(No승)}
            analyzer:  PostMatchAnalyzer (None이면 정산만)

        Returns:
            MatchReport (analyzer 있으면)
        """
        if self.state == SessionState.SETTLED:
            return None

        # ExecutionEngine 정산
        settle_records = self.tick_router.engine.settle_match(
            self.match_id, outcomes,
        )

        self.state = SessionState.SETTLED
        self.stats.end_time = time.time()

        logger.info(
            f"Match {self.match_id} settled: "
            f"{len(settle_records)} positions, "
            f"entries={self.stats.total_entries}, "
            f"exits={self.stats.total_exits}"
        )

        if not analyzer:
            return None

        # TradeOutcome 변환
        trade_outcomes = []
        for rec in settle_records:
            trade_outcomes.append(TradeOutcome(
                ticker=rec.ticker,
                match_id=self.match_id,
                direction=rec.direction,
                entry_price=rec.fill_price_cents / 100.0 if rec.fill_price_cents else 0,
                p_true_at_entry=rec.p_true,
                ev_adj_at_entry=rec.ev_adj,
                outcome=1 if outcomes.get(rec.ticker, False) else 0,
                pnl=rec.pnl,
                contracts=rec.quantity_filled,
            ))

        # 진입 로그에서 추가 정보 보충
        for entry_rec in self.tick_router.engine.trade_log:
            if (entry_rec.match_id == self.match_id and
                    entry_rec.order_type == "ENTRY" and
                    entry_rec.quantity_filled > 0):
                # 해당 ticker의 TradeOutcome 찾아서 보충
                for to in trade_outcomes:
                    if to.ticker == entry_rec.ticker and to.entry_price == 0:
                        to.entry_price = entry_rec.fill_price_cents / 100.0
                        to.fill_price = entry_rec.fill_price_cents / 100.0
                        to.signal_price = entry_rec.p_kalshi
                        to.p_true_at_entry = entry_rec.p_true
                        to.ev_adj_at_entry = entry_rec.ev_adj

        return analyzer.analyze_match(self.match_id, trade_outcomes)

    # ─── 유틸리티 ────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """세션 요약."""
        duration = self.stats.end_time - self.stats.start_time \
            if self.stats.end_time > 0 else 0
        return {
            "match_id": self.match_id,
            "state": self.state,
            "ticks": self.stats.total_ticks,
            "entries": self.stats.total_entries,
            "exits": self.stats.total_exits,
            "signals_evaluated": self.stats.total_signals_evaluated,
            "errors": len(self.stats.errors),
            "duration_sec": round(duration, 1),
        }

    @staticmethod
    def format_summary(s: Dict) -> str:
        return (
            f"Match [{s['match_id']}] {s['state']}  "
            f"ticks={s['ticks']} entries={s['entries']} "
            f"exits={s['exits']} eval={s['signals_evaluated']} "
            f"errs={s['errors']} {s['duration_sec']}s"
        )

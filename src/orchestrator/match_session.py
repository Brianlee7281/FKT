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

        모든 거래(조기 청산 + 만기 정산)를 PostMatch에 반영.

        Args:
            outcomes:  {kalshi_ticker: True(Yes승)/False(No승)}
            analyzer:  PostMatchAnalyzer (None이면 정산만)

        Returns:
            MatchReport (analyzer 있으면)
        """
        if self.state == SessionState.SETTLED:
            return None

        # ExecutionEngine 정산 (잔여 포지션)
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

        # ── 전체 거래 기록에서 TradeOutcome 구성 ──
        # trade_log에서 이 경기의 ENTRY만 추출
        trade_log = self.tick_router.engine.trade_log
        entries = [
            t for t in trade_log
            if t.match_id == self.match_id and t.order_type == "ENTRY"
        ]

        # ENTRY별로 대응하는 EXIT/SETTLEMENT 찾기
        exits = [
            t for t in trade_log
            if t.match_id == self.match_id and t.order_type != "ENTRY"
        ]

        # ticker별 exit 매핑 (같은 ticker의 exit들을 순서대로)
        from collections import defaultdict
        exit_by_ticker: Dict[str, list] = defaultdict(list)
        for ex in exits:
            exit_by_ticker[ex.ticker].append(ex)

        trade_outcomes = []
        used_exit_idx: Dict[str, int] = defaultdict(int)

        for entry_rec in entries:
            if entry_rec.quantity_filled <= 0:
                continue

            ticker = entry_rec.ticker
            outcome_val = 1 if outcomes.get(ticker, False) else 0

            # 대응 exit 찾기
            exit_list = exit_by_ticker.get(ticker, [])
            idx = used_exit_idx[ticker]
            exit_rec = exit_list[idx] if idx < len(exit_list) else None
            used_exit_idx[ticker] = idx + 1

            # 실현 P&L = entry P&L + exit P&L
            entry_pnl = entry_rec.pnl  # 음수 (비용)
            exit_pnl = exit_rec.pnl if exit_rec else 0.0
            total_pnl = entry_pnl + exit_pnl

            trade_outcomes.append(TradeOutcome(
                ticker=ticker,
                match_id=self.match_id,
                direction=entry_rec.direction,
                entry_price=entry_rec.fill_price_cents / 100.0 if entry_rec.fill_price_cents else 0,
                fill_price=entry_rec.fill_price_cents / 100.0 if entry_rec.fill_price_cents else 0,
                signal_price=entry_rec.p_kalshi,
                p_true_at_entry=entry_rec.p_true,
                ev_adj_at_entry=entry_rec.ev_adj,
                outcome=outcome_val,
                pnl=total_pnl,
                contracts=entry_rec.quantity_filled,
            ))

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
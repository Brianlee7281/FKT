"""
Kalshi 호가창 실시간 폴링 (orderbook_poller.py)

백그라운드 asyncio 태스크로 Kalshi 호가를 3~5초 간격 폴링하여
MatchSession 캐시에 주입한다.

구조:
  KalshiClient.get_orderbook(ticker) → OrderBook
      ↓ 변환
  OrderbookSnapshot (tick_router.py 포맷)
      ↓ 주입
  MatchSession.update_orderbook(market_type, snapshot)

사용법:
  poller = KalshiOrderbookPoller(
      kalshi_client=client,
      session=match_session,
      poll_interval=3.0,
  )
  task = asyncio.create_task(poller.run())
  ...
  poller.stop()
  await task
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from src.phase4.kalshi_client import KalshiClient, OrderBook
from src.orchestrator.tick_router import OrderbookSnapshot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 변환: KalshiClient.OrderBook → OrderbookSnapshot
# ═══════════════════════════════════════════════════

def orderbook_to_snapshot(ticker: str, ob: OrderBook) -> OrderbookSnapshot:
    """
    KalshiClient의 OrderBook을 TickRouter가 사용하는
    OrderbookSnapshot으로 변환한다.

    OrderBook:
      yes_bids: Yes 매수 호가 (높은 가격순)
      no_bids:  No 매수 호가 (높은 가격순)
      → best_yes_ask = 100 - best_no_bid
      → best_yes_bid = yes_bids[0].price_cents

    OrderbookSnapshot:
      yes_ask_cents: Yes 최저 매도가 (= 100 - best_no_bid)
      yes_bid_cents: Yes 최고 매수가
      yes_depth:     Yes ask 쪽 물량 (No bid 총량)
      no_depth:      No ask 쪽 물량 (Yes bid 총량)
    """
    return OrderbookSnapshot(
        ticker=ticker,
        yes_ask_cents=ob.best_yes_ask,
        yes_bid_cents=ob.best_yes_bid,
        yes_depth=int(ob.no_total_depth),   # Yes를 사려면 No bid을 소비
        no_depth=int(ob.yes_total_depth),    # No를 사려면 Yes bid을 소비
    )


# ═══════════════════════════════════════════════════
# 폴링 통계
# ═══════════════════════════════════════════════════

@dataclass
class PollerStats:
    """폴러 실행 통계."""
    polls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_poll_time: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    @property
    def success_rate(self) -> float:
        if self.polls == 0:
            return 0.0
        return self.successes / self.polls


# ═══════════════════════════════════════════════════
# Kalshi Orderbook Poller
# ═══════════════════════════════════════════════════

class KalshiOrderbookPoller:
    """
    Kalshi 호가창 주기적 폴링.

    백그라운드 asyncio 태스크로 실행되어 지정된 티커들의
    호가를 주기적으로 조회하고, 변환 후 콜백 또는
    MatchSession 캐시에 주입한다.

    Args:
        kalshi_client:  인증된 KalshiClient
        tickers:        {market_type: kalshi_ticker}
                        예: {"home_win": "SOCCER-...", "away_win": "SOCCER-...", "draw": "SOCCER-..."}
        poll_interval:  폴링 간격 (초, 기본 3.0)
        on_update:      호가 갱신 콜백
                        signature: (market_type: str, snapshot: OrderbookSnapshot) -> None
    """

    def __init__(
        self,
        kalshi_client: KalshiClient,
        tickers: Dict[str, str],
        poll_interval: float = 3.0,
        on_update: Optional[Callable[[str, OrderbookSnapshot], None]] = None,
    ):
        self.client = kalshi_client
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.on_update = on_update

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.stats = PollerStats()

        # 최신 스냅샷 캐시 (외부에서 직접 접근 가능)
        self.latest: Dict[str, OrderbookSnapshot] = {}

    # ─── 실행 제어 ─────────────────────────────────

    async def run(self):
        """폴링 루프 실행. asyncio.create_task()로 백그라운드 실행."""
        self._running = True
        logger.info(
            f"KalshiOrderbookPoller: 시작 "
            f"({len(self.tickers)}개 티커, {self.poll_interval}초 간격)"
        )

        while self._running:
            await self._poll_all()
            await asyncio.sleep(self.poll_interval)

        logger.info("KalshiOrderbookPoller: 중지")

    def stop(self):
        """폴링 루프 중지 신호."""
        self._running = False

    async def start_background(self) -> asyncio.Task:
        """백그라운드 태스크로 시작. 반환된 task를 나중에 cancel 가능."""
        self._task = asyncio.create_task(self.run())
        return self._task

    async def stop_and_wait(self):
        """중지 신호 + 태스크 완료 대기."""
        self.stop()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=self.poll_interval + 2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    # ─── 폴링 로직 ─────────────────────────────────

    async def _poll_all(self):
        """모든 티커의 호가창을 한 번 폴링."""
        self.stats.polls += 1
        self.stats.last_poll_time = time.time()

        for market_type, ticker in self.tickers.items():
            if not self._running:
                break
            await self._poll_one(market_type, ticker)

    async def _poll_one(self, market_type: str, ticker: str):
        """단일 티커 호가창 폴링 + 변환 + 콜백."""
        t0 = time.time()

        try:
            ob = await self.client.get_orderbook(ticker, depth=10)
            latency_ms = (time.time() - t0) * 1000

            # 통계 업데이트
            self.stats.successes += 1
            self.stats.total_latency_ms += latency_ms
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)

            # 변환
            snapshot = orderbook_to_snapshot(ticker, ob)

            # 캐시 저장
            self.latest[market_type] = snapshot

            # 콜백 호출
            if self.on_update:
                self.on_update(market_type, snapshot)

            logger.debug(
                f"  [{market_type}] {ticker}: "
                f"bid={snapshot.yes_bid_cents} ask={snapshot.yes_ask_cents} "
                f"spread={self._calc_spread(snapshot)} "
                f"({latency_ms:.0f}ms)"
            )

        except Exception as e:
            self.stats.failures += 1
            logger.warning(
                f"KalshiOrderbookPoller: {ticker} 폴링 실패 — "
                f"{type(e).__name__}: {e}"
            )

    @staticmethod
    def _calc_spread(snap: OrderbookSnapshot) -> Optional[int]:
        if snap.yes_ask_cents is not None and snap.yes_bid_cents is not None:
            return snap.yes_ask_cents - snap.yes_bid_cents
        return None

    # ─── MatchSession 연동 헬퍼 ────────────────────

    def make_orderbook_fn(self) -> Callable[[str], Optional[OrderbookSnapshot]]:
        """
        MatchSession의 orderbook_fn 파라미터로 사용할 수 있는 함수를 반환.

        주의: 이 함수는 동기(sync)이며, 캐시된 최신 스냅샷을 반환한다.
        폴링 루프가 백그라운드에서 캐시를 업데이트해야 한다.
        """
        def _fn(ticker: str) -> Optional[OrderbookSnapshot]:
            # ticker → market_type 역매핑
            for market_type, t in self.tickers.items():
                if t == ticker:
                    return self.latest.get(market_type)
            return None
        return _fn

    def bind_to_session(self, session) -> None:
        """
        MatchSession에 이 폴러를 바인딩.
        on_update 콜백을 session.update_orderbook에 연결.
        """
        self.on_update = session.update_orderbook
        logger.info("KalshiOrderbookPoller: MatchSession에 바인딩 완료")

    # ─── 디버그 ────────────────────────────────────

    def format_stats(self) -> str:
        """통계 요약 문자열."""
        s = self.stats
        return (
            f"polls={s.polls} success={s.successes} fail={s.failures} "
            f"avg_lat={s.avg_latency_ms:.0f}ms max_lat={s.max_latency_ms:.0f}ms "
            f"rate={s.success_rate:.1%}"
        )

    def format_latest(self) -> str:
        """최신 호가 요약."""
        parts = []
        for market_type, snap in self.latest.items():
            spread = self._calc_spread(snap)
            parts.append(
                f"{market_type}: bid={snap.yes_bid_cents} "
                f"ask={snap.yes_ask_cents} "
                f"spread={spread} "
                f"depth=({snap.yes_depth}/{snap.no_depth})"
            )
        return " | ".join(parts) if parts else "(no data)"

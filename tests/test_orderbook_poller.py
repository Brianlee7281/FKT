"""
orderbook_poller.py 단위 테스트.

실행:
  python -m pytest tests/test_orderbook_poller.py -v
  python tests/test_orderbook_poller.py
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.phase4.kalshi_client import OrderBook, OrderBookLevel
from src.orchestrator.orderbook_poller import (
    KalshiOrderbookPoller,
    orderbook_to_snapshot,
    PollerStats,
)
from src.orchestrator.tick_router import OrderbookSnapshot


def test_all():

    # ──────────────────────────────────────────────
    # T1: orderbook_to_snapshot — 기본 변환
    # ──────────────────────────────────────────────
    ob = OrderBook(
        yes_bids=[
            OrderBookLevel(price_cents=42, quantity=100),
            OrderBookLevel(price_cents=40, quantity=50),
        ],
        no_bids=[
            OrderBookLevel(price_cents=60, quantity=200),
            OrderBookLevel(price_cents=58, quantity=80),
        ],
    )
    snap = orderbook_to_snapshot("SOCCER-TEST", ob)

    assert snap.ticker == "SOCCER-TEST"
    assert snap.yes_bid_cents == 42       # best yes bid
    assert snap.yes_ask_cents == 40       # 100 - best_no_bid(60) = 40
    assert snap.yes_depth == 280          # no_total_depth (Yes 사려면 No bid 소비)
    assert snap.no_depth == 150           # yes_total_depth (No 사려면 Yes bid 소비)
    print("✅ T1: orderbook_to_snapshot — 기본 변환")

    # ──────────────────────────────────────────────
    # T2: 빈 호가창 변환
    # ──────────────────────────────────────────────
    empty_ob = OrderBook()
    snap2 = orderbook_to_snapshot("EMPTY", empty_ob)
    assert snap2.yes_bid_cents is None
    assert snap2.yes_ask_cents is None
    assert snap2.yes_depth == 0
    assert snap2.no_depth == 0
    print("✅ T2: 빈 호가창 변환 — None 반환")

    # ──────────────────────────────────────────────
    # T3: 한쪽만 있는 호가창
    # ──────────────────────────────────────────────
    one_side = OrderBook(
        yes_bids=[OrderBookLevel(price_cents=50, quantity=30)],
        no_bids=[],
    )
    snap3 = orderbook_to_snapshot("ONE-SIDE", one_side)
    assert snap3.yes_bid_cents == 50
    assert snap3.yes_ask_cents is None  # no_bids 없으면 ask 계산 불가
    assert snap3.no_depth == 30
    assert snap3.yes_depth == 0
    print("✅ T3: 한쪽 호가만 있을 때 — ask None")

    # ──────────────────────────────────────────────
    # T4: PollerStats 계산
    # ──────────────────────────────────────────────
    stats = PollerStats(polls=10, successes=8, failures=2, total_latency_ms=4000, max_latency_ms=800)
    assert stats.avg_latency_ms == 500.0
    assert stats.success_rate == 0.8
    print("✅ T4: PollerStats — 평균/성공률")

    # ──────────────────────────────────────────────
    # T5: PollerStats 빈 상태
    # ──────────────────────────────────────────────
    empty_stats = PollerStats()
    assert empty_stats.avg_latency_ms == 0.0
    assert empty_stats.success_rate == 0.0
    print("✅ T5: PollerStats 빈 상태")

    # ──────────────────────────────────────────────
    # T6: KalshiOrderbookPoller 초기화
    # ──────────────────────────────────────────────
    mock_client = MagicMock()
    tickers = {
        "home_win": "SOCCER-HW",
        "away_win": "SOCCER-AW",
        "draw": "SOCCER-DR",
    }
    poller = KalshiOrderbookPoller(
        kalshi_client=mock_client,
        tickers=tickers,
        poll_interval=3.0,
    )
    assert len(poller.tickers) == 3
    assert poller.poll_interval == 3.0
    assert poller._running is False
    print("✅ T6: KalshiOrderbookPoller 초기화")

    # ──────────────────────────────────────────────
    # T7: make_orderbook_fn — 캐시 기반 조회
    # ──────────────────────────────────────────────
    poller.latest["home_win"] = OrderbookSnapshot(
        ticker="SOCCER-HW",
        yes_ask_cents=55,
        yes_bid_cents=52,
        yes_depth=100,
        no_depth=80,
    )
    fn = poller.make_orderbook_fn()

    result = fn("SOCCER-HW")
    assert result is not None
    assert result.yes_ask_cents == 55

    result2 = fn("SOCCER-UNKNOWN")
    assert result2 is None
    print("✅ T7: make_orderbook_fn — 캐시 기반 조회")

    # ──────────────────────────────────────────────
    # T8: bind_to_session
    # ──────────────────────────────────────────────
    mock_session = MagicMock()
    mock_session.update_orderbook = MagicMock()
    poller.bind_to_session(mock_session)
    assert poller.on_update == mock_session.update_orderbook
    print("✅ T8: bind_to_session — 콜백 연결")

    # ──────────────────────────────────────────────
    # T9: _poll_one — 성공 케이스
    # ──────────────────────────────────────────────
    async def _test_poll_one():
        mock_client2 = AsyncMock()
        mock_client2.get_orderbook = AsyncMock(return_value=OrderBook(
            yes_bids=[OrderBookLevel(45, 100)],
            no_bids=[OrderBookLevel(58, 150)],
        ))

        updates = []
        poller2 = KalshiOrderbookPoller(
            kalshi_client=mock_client2,
            tickers={"home_win": "SOCCER-HW"},
            on_update=lambda mt, snap: updates.append((mt, snap)),
        )

        await poller2._poll_one("home_win", "SOCCER-HW")

        assert poller2.stats.successes == 1
        assert "home_win" in poller2.latest
        assert poller2.latest["home_win"].yes_bid_cents == 45
        assert len(updates) == 1
        assert updates[0][0] == "home_win"

    asyncio.run(_test_poll_one())
    print("✅ T9: _poll_one — 성공 (폴링+변환+콜백)")

    # ──────────────────────────────────────────────
    # T10: _poll_one — 실패 케이스
    # ──────────────────────────────────────────────
    async def _test_poll_fail():
        mock_client3 = AsyncMock()
        mock_client3.get_orderbook = AsyncMock(side_effect=Exception("timeout"))

        poller3 = KalshiOrderbookPoller(
            kalshi_client=mock_client3,
            tickers={"home_win": "SOCCER-HW"},
        )

        await poller3._poll_one("home_win", "SOCCER-HW")

        assert poller3.stats.failures == 1
        assert poller3.stats.successes == 0
        assert "home_win" not in poller3.latest

    asyncio.run(_test_poll_fail())
    print("✅ T10: _poll_one — 실패 처리 (에러 카운트)")

    # ──────────────────────────────────────────────
    # T11: format_stats
    # ──────────────────────────────────────────────
    poller4 = KalshiOrderbookPoller(
        kalshi_client=MagicMock(),
        tickers={"home_win": "T1"},
    )
    poller4.stats = PollerStats(polls=5, successes=4, failures=1, total_latency_ms=2000, max_latency_ms=700)
    s = poller4.format_stats()
    assert "polls=5" in s
    assert "80.0%" in s
    print("✅ T11: format_stats — 포맷 정상")

    # ──────────────────────────────────────────────
    # T12: stop 시그널
    # ──────────────────────────────────────────────
    poller5 = KalshiOrderbookPoller(
        kalshi_client=MagicMock(),
        tickers={"hw": "T1"},
    )
    assert poller5._running is False
    poller5.stop()
    assert poller5._running is False
    print("✅ T12: stop 시그널")

    # ──────────────────────────────────────────────
    print()
    print("═" * 50)
    print("  ALL 12 TESTS PASSED ✅")
    print("═" * 50)


if __name__ == "__main__":
    test_all()

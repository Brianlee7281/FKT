"""
Kalshi API 연결 테스트.

사용법:
  python test_kalshi_live.py                 # 잔고 + 축구 마켓 검색
  python test_kalshi_live.py TICKER          # 특정 마켓 호가창 조회

.env 파일 필요:
  KALSHI_API_KEY=your-api-key-id
  KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
"""

import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

# .env 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.phase4.kalshi_client import KalshiClient


async def test_connection():
    """기본 연결 테스트: 잔고 조회."""
    print("=" * 50)
    print("  Kalshi API 연결 테스트")
    print("=" * 50)

    client = KalshiClient()
    print(f"\n  API Key: {client.api_key[:8]}...")
    print(f"  Key Path: {client.private_key_path}")
    print(f"  Base URL: {client.base_url}")

    try:
        await client.connect()
        balance = await client.get_balance()
        print(f"\n  ✅ 연결 성공!")
        print(f"  💰 잔고: ${balance / 100:.2f}")
    except FileNotFoundError:
        print(f"\n  ❌ 키 파일 없음: {client.private_key_path}")
        print("  .env에 KALSHI_PRIVATE_KEY_PATH 설정 확인")
        return
    except Exception as e:
        print(f"\n  ❌ 연결 실패: {type(e).__name__}: {e}")
        return

    # 축구 관련 마켓 검색
    print(f"\n{'─'*50}")
    print("  축구/Soccer 마켓 검색 중...")
    try:
        markets = await client.search_markets(status="open", limit=200)
        soccer_markets = [
            m for m in markets
            if any(kw in (m.get("title", "") + m.get("ticker", "")).lower()
                   for kw in ["soccer", "football", "epl", "premier", "la liga",
                              "champions", "goal", "match"])
        ]

        if soccer_markets:
            print(f"  축구 마켓 {len(soccer_markets)}개 발견:\n")
            for m in soccer_markets[:10]:
                print(f"  [{m['ticker']}]")
                print(f"    {m['title']}")
                print(f"    Yes: {m.get('yes_price', '?')}¢  Vol: {m.get('volume', '?')}")
                print()
        else:
            print("  축구 마켓 없음 (시즌 외이거나 다른 카테고리)")
            print(f"  총 {len(markets)}개 마켓 중 상위 5개:")
            for m in markets[:5]:
                print(f"    [{m['ticker']}] {m['title']}")
    except Exception as e:
        print(f"  마켓 검색 실패: {type(e).__name__}: {e}")

    await client.disconnect()


async def test_orderbook(ticker: str):
    """특정 마켓 호가창 조회."""
    print(f"\n{'='*50}")
    print(f"  호가창: {ticker}")
    print(f"{'='*50}")

    client = KalshiClient()
    try:
        await client.connect()

        # 마켓 정보
        market = await client.get_market(ticker)
        print(f"\n  {market.get('title', ticker)}")
        print(f"  Status: {market.get('status', '?')}")
        print(f"  Yes Price: {market.get('yes_price', '?')}¢")
        print(f"  Volume: {market.get('volume', '?')}")

        # 호가창
        ob = await client.get_orderbook(ticker, depth=5)
        print(f"\n  {'─'*40}")
        print(f"  Best Yes Bid: {ob.best_yes_bid}¢")
        print(f"  Best Yes Ask: {ob.best_yes_ask}¢")
        print(f"  Spread: {ob.spread}¢")
        print(f"  Yes Depth: {ob.yes_total_depth} contracts")
        print(f"  No Depth: {ob.no_total_depth} contracts")
        print(f"  Liquidity OK: {ob.liquidity_ok()}")

        print(f"\n  Yes Bids (매수 호가):")
        for lvl in ob.yes_bids[:5]:
            print(f"    {lvl.price_cents}¢ × {lvl.quantity}")

        print(f"\n  No Bids (= Yes Asks):")
        for lvl in ob.no_bids[:5]:
            ask = 100 - lvl.price_cents
            print(f"    {ask}¢ (No bid {lvl.price_cents}¢) × {lvl.quantity}")

        # VWAP 테스트
        for qty in [10, 50, 100]:
            vwap = ob.vwap_yes_buy(qty)
            if vwap:
                print(f"\n  VWAP {qty}계약 매수: {vwap:.2f}¢")

    except Exception as e:
        print(f"\n  ❌ 오류: {type(e).__name__}: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(test_orderbook(sys.argv[1]))
    else:
        asyncio.run(test_connection())
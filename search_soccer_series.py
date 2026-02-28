"""
Kalshi 축구 단일 경기 마켓 검색 (series ticker 기반).
"""

import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.phase4.kalshi_client import KalshiClient


SOCCER_SERIES = [
    "KXEPLGAME",
    "KXSERIEAGAME",
    "KXUCLGAME",
    "KXLALIGAGAME",
    "KXBUNDESLIGAGAME",
    "KXLIGUE1GAME",
    "KXSUPERLIGGAME",
    "KXARGPREMDIVGAME",
    "KXEREDIVISIEGAME",
    "KXLIGAMXGAME",
    "KXCONFLEAGUEGAME",
    "KXEUROPALEAGUEGAME",
    "KXEFLCHAMPGAME",
]


async def main():
    client = KalshiClient()
    await client.connect()

    print("=" * 60)
    print("  Kalshi 축구 단일 경기 마켓")
    print("=" * 60)

    total = 0
    for series in SOCCER_SERIES:
        try:
            markets = await client.search_markets(series_ticker=series, status="open")
            if markets:
                total += len(markets)
                print(f"\n  📋 {series}: {len(markets)}개 마켓")
                for m in markets[:5]:
                    ticker = m.get("ticker", "?")
                    title = m.get("title", "?")
                    yes_p = m.get("yes_price", "?")
                    vol = m.get("volume", 0)
                    print(f"    [{ticker}]")
                    print(f"      {title}")
                    print(f"      Yes: {yes_p}¢  Vol: {vol}")

                    # 호가창 첫 번째 마켓만
                    if m == markets[0]:
                        try:
                            ob = await client.get_orderbook(ticker, depth=3)
                            print(f"      OB: bid={ob.best_yes_bid}¢ ask={ob.best_yes_ask}¢ "
                                  f"spread={ob.spread}¢ "
                                  f"depth=({ob.yes_total_depth}/{ob.no_total_depth})")
                        except Exception:
                            pass

                if len(markets) > 5:
                    print(f"    ... 외 {len(markets)-5}개")
        except Exception as e:
            pass  # 시리즈가 없으면 무시

    print(f"\n{'='*60}")
    print(f"  총 단일 경기 마켓: {total}개")
    print(f"{'='*60}")

    await client.disconnect()


asyncio.run(main())

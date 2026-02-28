"""
Kalshi 축구 단일 경기 마켓 탐색.
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


async def main():
    client = KalshiClient()
    await client.connect()

    print("=" * 60)
    print("  Kalshi 축구 마켓 탐색")
    print("=" * 60)

    # 1. 전체 마켓에서 축구 관련 키워드 검색
    soccer_keywords = [
        "premier league", "epl", "la liga", "serie a", "bundesliga",
        "champions league", "arsenal", "liverpool", "manchester",
        "chelsea", "barcelona", "real madrid", "soccer", "football",
        "goal", "win", "draw",
    ]

    all_markets = []
    cursor = ""
    for i in range(5):  # 최대 500개
        params = {"status": "open", "limit": "200"}
        if cursor:
            params["cursor"] = cursor
        data = await client._get("/markets", params=params)
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor", "")
        if not cursor or not markets:
            break

    print(f"\n  총 오픈 마켓: {len(all_markets)}개")

    # 축구 관련 필터 (파레이 제외)
    single_soccer = []
    parlay_soccer = []
    for m in all_markets:
        title = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
        ticker = m.get("ticker", "").lower()
        event = m.get("event_ticker", "").lower()

        is_soccer = any(kw in title or kw in ticker or kw in event for kw in soccer_keywords)
        if not is_soccer:
            continue

        is_parlay = "parlay" in title or "KXMVE" in m.get("ticker", "") or "multi" in ticker
        if is_parlay:
            parlay_soccer.append(m)
        else:
            single_soccer.append(m)

    print(f"  축구 단일 마켓: {len(single_soccer)}개")
    print(f"  축구 파레이 마켓: {len(parlay_soccer)}개")

    if single_soccer:
        print(f"\n{'─'*60}")
        print("  축구 단일 마켓:")
        for m in single_soccer[:20]:
            print(f"\n  [{m['ticker']}]")
            print(f"    Title: {m.get('title', '?')}")
            print(f"    Event: {m.get('event_ticker', '?')}")
            print(f"    Yes: {m.get('yes_price', '?')}¢  Vol: {m.get('volume', '?')}")
    else:
        print("\n  ⚠️ 단일 축구 마켓 없음")
        print("  파레이만 있는 경우, 개별 경기 마켓이 Kalshi에 없을 수 있음")

    # 2. 이벤트 레벨에서도 검색
    print(f"\n{'─'*60}")
    print("  이벤트 레벨 검색 (series)...")

    series_keywords = ["soccer", "football", "epl", "premier", "champions"]
    for kw in series_keywords:
        try:
            data = await client._get("/series", {"limit": "20"})
            series_list = data.get("series", [])
            matches = [s for s in series_list if kw in s.get("ticker", "").lower()
                       or kw in s.get("title", "").lower()]
            if matches:
                print(f"\n  '{kw}' 시리즈:")
                for s in matches[:5]:
                    print(f"    [{s['ticker']}] {s.get('title', '?')}")
        except Exception as e:
            print(f"    {kw}: {type(e).__name__}")
            break

    # 3. 카테고리/서브카테고리 확인
    print(f"\n{'─'*60}")
    print("  전체 시리즈 (sports 관련)...")
    try:
        data = await client._get("/series", {"limit": "200"})
        series_list = data.get("series", [])
        sports = [s for s in series_list
                  if any(kw in (s.get("ticker", "") + s.get("title", "")).lower()
                         for kw in ["sport", "soccer", "football", "nfl", "nba",
                                    "epl", "premier", "goal", "match"])]
        print(f"  스포츠 시리즈: {len(sports)}개")
        for s in sports[:20]:
            print(f"    [{s['ticker']}] {s.get('title', '?')} (cat: {s.get('category', '?')})")
    except Exception as e:
        print(f"  시리즈 조회 실패: {type(e).__name__}: {e}")

    await client.disconnect()


asyncio.run(main())

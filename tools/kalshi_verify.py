#!/usr/bin/env python3
"""
Kalshi 호가창 실시간 폴링 검증 도구.

사용법:
  # 1단계: 연결 + 잔고 확인
  python tools/kalshi_verify.py --check

  # 2단계: 축구 마켓 검색
  python tools/kalshi_verify.py --soccer

  # 3단계: 특정 마켓 호가창 1회 조회
  python tools/kalshi_verify.py --orderbook TICKER-NAME

  # 4단계: 특정 마켓들 실시간 폴링 (3초 간격)
  python tools/kalshi_verify.py --poll TICKER1 TICKER2 TICKER3

  # 5단계: 특정 이벤트의 모든 마켓 호가 모니터링
  python tools/kalshi_verify.py --event EVENT_TICKER

환경변수:
  KALSHI_API_KEY=<api_key_id>
  KALSHI_PRIVATE_KEY_PATH=<path_to_pem>
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.phase4.kalshi_client import KalshiClient, OrderBook
from src.orchestrator.orderbook_poller import orderbook_to_snapshot


# ═══════════════════════════════════════════════════
# 환경 확인
# ═══════════════════════════════════════════════════

def check_env():
    """필수 환경변수 확인."""
    api_key = os.getenv("KALSHI_API_KEY", "")
    pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

    if not api_key:
        # .env 파일 시도
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("KALSHI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("KALSHI_PRIVATE_KEY_PATH="):
                        pem_path = line.split("=", 1)[1].strip().strip('"').strip("'")

    if not api_key:
        print("❌ KALSHI_API_KEY 미설정")
        print("   export KALSHI_API_KEY=<your_api_key_id>")
        sys.exit(1)

    if not pem_path:
        # 기본 위치들 시도
        for p in ["./kalshi_private_key.pem", "./kalshi.pem", "../kalshi_private_key.pem"]:
            if os.path.exists(p):
                pem_path = p
                break

    if not pem_path or not os.path.exists(pem_path):
        print(f"❌ PEM 파일을 찾을 수 없음: {pem_path or '(미설정)'}")
        print("   export KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem")
        sys.exit(1)

    return api_key, pem_path


# ═══════════════════════════════════════════════════
# 1단계: 연결 테스트
# ═══════════════════════════════════════════════════

async def check_connection():
    """API 인증 + 잔고 확인."""
    print("\n" + "=" * 60)
    print("  KALSHI API 연결 테스트")
    print("=" * 60)

    api_key, pem_path = check_env()
    print(f"\n  API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"  PEM:     {pem_path}")

    client = KalshiClient(api_key=api_key, private_key_path=pem_path)

    try:
        await client.connect()
    except FileNotFoundError:
        print(f"\n  ❌ PEM 파일 없음: {pem_path}")
        return
    except Exception as e:
        print(f"\n  ❌ 연결 실패: {type(e).__name__}: {e}")
        return

    try:
        t0 = time.time()
        balance = await client.get_balance()
        latency = (time.time() - t0) * 1000
        print(f"\n  ✅ 인증 성공 ({latency:.0f}ms)")
        print(f"  잔고: ${balance / 100:.2f}")

        # 포지션 확인
        positions = await client.get_positions()
        print(f"  포지션: {len(positions)}개")

        # 레이턴시 측정
        print(f"\n  ── 레이턴시 측정 (3회) ──")
        latencies = []
        for i in range(3):
            t0 = time.time()
            await client.get_balance()
            lat = (time.time() - t0) * 1000
            latencies.append(lat)
            print(f"  {i+1}회: {lat:.0f}ms")
        avg = sum(latencies) / len(latencies)
        print(f"  평균: {avg:.0f}ms")

    finally:
        await client.disconnect()

    print(f"\n  {'='*60}")
    print(f"  연결 테스트 완료 ✅")
    print(f"  {'='*60}")


# ═══════════════════════════════════════════════════
# 2단계: 축구 마켓 검색
# ═══════════════════════════════════════════════════

async def search_soccer():
    """현재 열린 축구 마켓 검색."""
    print("\n" + "=" * 60)
    print("  KALSHI 축구 마켓 검색")
    print("=" * 60)

    api_key, pem_path = check_env()
    client = KalshiClient(api_key=api_key, private_key_path=pem_path)
    await client.connect()

    try:
        # 축구 시리즈 검색
        soccer_keywords = ["SOCCER", "FOOTBALL", "EPL", "MLS", "LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1"]

        all_markets = []
        for kw in soccer_keywords:
            try:
                markets = await client.search_markets(series_ticker=kw, limit=100)
                all_markets.extend(markets)
            except Exception:
                pass

        # 중복 제거
        seen = set()
        unique = []
        for m in all_markets:
            tid = m.get("ticker", "")
            if tid not in seen:
                seen.add(tid)
                unique.append(m)

        if not unique:
            print("\n  축구 마켓 없음. 직접 검색:")
            print("  python tools/kalshi_verify.py --event <EVENT_TICKER>")

            # 대안: 전체 오픈 마켓에서 soccer/football 키워드 탐색
            print("\n  전체 마켓에서 축구 키워드 탐색 중...")
            try:
                all_open = await client.search_markets(limit=200)
                soccer_matches = [
                    m for m in all_open
                    if any(kw.lower() in (m.get("ticker", "") + m.get("title", "")).lower()
                           for kw in ["soccer", "football", "mls", "epl", "premier", "liga"])
                ]
                if soccer_matches:
                    unique = soccer_matches
                    print(f"  {len(soccer_matches)}개 발견!")
                else:
                    print("  발견 안 됨. Kalshi 웹사이트에서 축구 마켓 확인 필요.")
            except Exception as e:
                print(f"  검색 실패: {e}")
            if not unique:
                return

        # 이벤트별 그룹핑
        by_event = {}
        for m in unique:
            event = m.get("event_ticker", "")
            if event not in by_event:
                by_event[event] = []
            by_event[event].append(m)

        print(f"\n  축구 마켓: {len(unique)}개 ({len(by_event)} 이벤트)")

        for event, markets in sorted(by_event.items()):
            print(f"\n  📋 Event: {event}")
            print(f"  {'─'*55}")
            for m in markets[:10]:  # 이벤트당 최대 10개
                ticker = m.get("ticker", "")
                title = m.get("title", m.get("subtitle", ""))[:40]
                yes_bid = m.get("yes_bid", "?")
                yes_ask = m.get("yes_ask", "?")
                status = m.get("status", "?")
                print(f"    [{status:>6}] {ticker:<35} {title}")
                if yes_bid or yes_ask:
                    print(f"             bid={yes_bid} ask={yes_ask}")

    finally:
        await client.disconnect()


# ═══════════════════════════════════════════════════
# 3단계: 호가창 조회
# ═══════════════════════════════════════════════════

async def show_orderbook(ticker: str):
    """특정 마켓 호가창 상세 조회."""
    print(f"\n{'='*60}")
    print(f"  호가창: {ticker}")
    print(f"{'='*60}")

    api_key, pem_path = check_env()
    client = KalshiClient(api_key=api_key, private_key_path=pem_path)
    await client.connect()

    try:
        # 마켓 정보
        try:
            market = await client.get_market(ticker)
            title = market.get("title", market.get("subtitle", ""))
            status = market.get("status", "?")
            print(f"\n  제목:   {title}")
            print(f"  상태:   {status}")
        except Exception as e:
            print(f"\n  마켓 정보 조회 실패: {e}")

        # 호가창
        t0 = time.time()
        ob = await client.get_orderbook(ticker, depth=10)
        latency = (time.time() - t0) * 1000

        print(f"\n  ── Yes Bids (매수 호가) ── ({latency:.0f}ms)")
        if ob.yes_bids:
            for lvl in ob.yes_bids[:5]:
                bar = "█" * int(lvl.quantity / 5)
                print(f"    {lvl.price_cents:>3}¢  {lvl.quantity:>6.0f}  {bar}")
        else:
            print("    (없음)")

        print(f"\n  ── No Bids (매수 호가) ──")
        if ob.no_bids:
            for lvl in ob.no_bids[:5]:
                bar = "█" * int(lvl.quantity / 5)
                print(f"    {lvl.price_cents:>3}¢  {lvl.quantity:>6.0f}  {bar}")
        else:
            print("    (없음)")

        # 요약
        print(f"\n  ── 요약 ──")
        print(f"  Best Yes Bid: {ob.best_yes_bid}¢")
        print(f"  Best Yes Ask: {ob.best_yes_ask}¢")
        print(f"  Spread:       {ob.spread}¢")
        print(f"  Yes Depth:    {ob.yes_total_depth:.0f}")
        print(f"  No Depth:     {ob.no_total_depth:.0f}")

        # OrderbookSnapshot 변환 테스트
        snap = orderbook_to_snapshot(ticker, ob)
        print(f"\n  ── OrderbookSnapshot 변환 ──")
        print(f"  yes_ask_cents: {snap.yes_ask_cents}")
        print(f"  yes_bid_cents: {snap.yes_bid_cents}")
        print(f"  yes_depth:     {snap.yes_depth}")
        print(f"  no_depth:      {snap.no_depth}")

    finally:
        await client.disconnect()


# ═══════════════════════════════════════════════════
# 4단계: 실시간 폴링
# ═══════════════════════════════════════════════════

async def poll_tickers(tickers: list, interval: float = 3.0):
    """여러 티커를 실시간 폴링."""
    print(f"\n{'='*60}")
    print(f"  호가창 실시간 폴링 ({len(tickers)}개, {interval}초)")
    print(f"  Ctrl+C로 종료")
    print(f"{'='*60}")

    api_key, pem_path = check_env()
    client = KalshiClient(api_key=api_key, private_key_path=pem_path)
    await client.connect()

    poll_count = 0
    try:
        while True:
            poll_count += 1
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n  [{ts}] Poll #{poll_count}")
            print(f"  {'─'*55}")

            for ticker in tickers:
                try:
                    t0 = time.time()
                    ob = await client.get_orderbook(ticker, depth=5)
                    lat = (time.time() - t0) * 1000

                    spread = ob.spread or "?"
                    print(
                        f"  {ticker:<35} "
                        f"bid={ob.best_yes_bid or '?':>3} "
                        f"ask={ob.best_yes_ask or '?':>3} "
                        f"spd={spread:>2} "
                        f"depth=({ob.yes_total_depth:.0f}/{ob.no_total_depth:.0f}) "
                        f"({lat:.0f}ms)"
                    )
                except Exception as e:
                    print(f"  {ticker:<35} ❌ {e}")

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        await client.disconnect()
        print(f"\n  종료 (총 {poll_count}회 폴링)")


# ═══════════════════════════════════════════════════
# 5단계: 이벤트별 마켓 모니터링
# ═══════════════════════════════════════════════════

async def monitor_event(event_ticker: str, interval: float = 5.0):
    """특정 이벤트의 모든 마켓 호가 모니터링."""
    print(f"\n{'='*60}")
    print(f"  이벤트 모니터링: {event_ticker}")
    print(f"{'='*60}")

    api_key, pem_path = check_env()
    client = KalshiClient(api_key=api_key, private_key_path=pem_path)
    await client.connect()

    try:
        # 이벤트 내 마켓 검색
        markets = await client.search_markets(event_ticker=event_ticker, limit=50)
        if not markets:
            print(f"\n  ❌ '{event_ticker}' 이벤트에 마켓 없음")
            return

        tickers = [m["ticker"] for m in markets if m.get("status") == "open"]
        print(f"\n  오픈 마켓: {len(tickers)}개")
        for m in markets:
            ticker = m.get("ticker", "")
            title = m.get("title", m.get("subtitle", ""))[:45]
            status = m.get("status", "?")
            print(f"    [{status:>6}] {ticker}")
            print(f"             {title}")

        if not tickers:
            print("\n  오픈 마켓 없음. 모든 마켓이 닫혀있음.")
            return

        # 폴링
        print(f"\n  폴링 시작 ({interval}초 간격, Ctrl+C 종료)...")
        await poll_tickers(tickers, interval=interval)

    finally:
        await client.disconnect()


# ═══════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Kalshi 호가창 검증")
    parser.add_argument("--check", action="store_true", help="연결 + 잔고 확인")
    parser.add_argument("--soccer", action="store_true", help="축구 마켓 검색")
    parser.add_argument("--orderbook", type=str, metavar="TICKER", help="호가창 조회")
    parser.add_argument("--poll", nargs="+", metavar="TICKER", help="실시간 폴링")
    parser.add_argument("--event", type=str, metavar="EVENT_TICKER", help="이벤트 마켓 모니터링")
    parser.add_argument("--interval", type=float, default=3.0, help="폴링 간격 (초)")
    args = parser.parse_args()

    if not any([args.check, args.soccer, args.orderbook, args.poll, args.event]):
        parser.print_help()
        return

    if args.check:
        asyncio.run(check_connection())
    elif args.soccer:
        asyncio.run(search_soccer())
    elif args.orderbook:
        asyncio.run(show_orderbook(args.orderbook))
    elif args.poll:
        asyncio.run(poll_tickers(args.poll, interval=args.interval))
    elif args.event:
        asyncio.run(monitor_event(args.event, interval=args.interval))


if __name__ == "__main__":
    main()

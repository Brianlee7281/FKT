#!/usr/bin/env python3
"""
Goalserve 라이브 데이터 연동 검증 도구.

사용법:
  # 1단계: API 연결 테스트
  python tools/goalserve_verify.py --check

  # 2단계: 현재 라이브 경기 목록
  python tools/goalserve_verify.py --live

  # 3단계: 오늘 전체 경기 목록 (FT 포함)
  python tools/goalserve_verify.py --all

  # 4단계: 특정 경기 이벤트 실시간 수신
  python tools/goalserve_verify.py --watch <match_id>

  # 5단계: 특정 경기 JSON 원본 덤프
  python tools/goalserve_verify.py --dump <match_id>

  # 6단계: 완료된 경기(FT)로 이벤트 파싱 정확도 검증
  python tools/goalserve_verify.py --validate-ft

환경변수:
  GOALSERVE_API_KEY=<your_api_key>
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from src.phase3.goalserve import GoalserveSource, GOALSERVE_BASE_URL
from src.phase3.event_source import EventType


# ═══════════════════════════════════════════════════════════
# API 키 로드
# ═══════════════════════════════════════════════════════════

def get_api_key() -> str:
    """환경변수 또는 .env 파일에서 API 키 로드."""
    key = os.getenv("GOALSERVE_API_KEY", "")
    if not key:
        # .env 파일 시도
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOALSERVE_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not key:
        print("❌ GOALSERVE_API_KEY가 설정되지 않았습니다.")
        print("   export GOALSERVE_API_KEY=<your_key>")
        print("   또는 .env 파일에 GOALSERVE_API_KEY=<your_key> 추가")
        sys.exit(1)
    return key


# ═══════════════════════════════════════════════════════════
# 1단계: API 연결 테스트
# ═══════════════════════════════════════════════════════════

async def check_connection(api_key: str):
    """API 연결 + 응답 구조 검증."""
    print("\n" + "=" * 60)
    print("  GOALSERVE API 연결 테스트")
    print("=" * 60)

    url = f"{GOALSERVE_BASE_URL}/{api_key}/soccernew/live?json=1"
    print(f"\n  URL: {url[:60]}...")

    async with httpx.AsyncClient(timeout=10.0) as client:
        t0 = time.time()
        try:
            resp = await client.get(url)
            latency = (time.time() - t0) * 1000
        except Exception as e:
            print(f"\n  ❌ 연결 실패: {type(e).__name__}: {e}")
            return False

    print(f"\n  ✅ HTTP {resp.status_code} ({latency:.0f}ms)")
    print(f"  Content-Type: {resp.headers.get('content-type', 'N/A')}")
    print(f"  Content-Length: {len(resp.content):,} bytes")

    # JSON 파싱
    try:
        data = resp.json()
    except Exception as e:
        print(f"\n  ❌ JSON 파싱 실패: {e}")
        print(f"  응답 본문: {resp.text[:200]}")
        return False

    # 구조 확인
    if "scores" not in data:
        print(f"\n  ❌ 'scores' 키 없음. 키 목록: {list(data.keys())}")
        # 에러 메시지 확인
        if "error" in data:
            print(f"  에러: {data['error']}")
        return False

    categories = data.get("scores", {}).get("category", [])
    if isinstance(categories, dict):
        categories = [categories]

    total_matches = 0
    live_matches = 0
    ft_matches = 0
    leagues = set()

    for cat in categories:
        league = cat.get("@name", "")
        leagues.add(league)
        matches = cat.get("matches", {}).get("match", [])
        if isinstance(matches, dict):
            matches = [matches]
        for m in matches:
            total_matches += 1
            status = m.get("@status", "")
            if status == "FT":
                ft_matches += 1
            elif status not in ("", "Postp.", "Canc.", "Awarded"):
                live_matches += 1

    print(f"\n  ✅ JSON 구조 정상")
    print(f"  리그 수:       {len(leagues)}")
    print(f"  전체 경기 수:  {total_matches}")
    print(f"  라이브:        {live_matches}")
    print(f"  종료(FT):      {ft_matches}")

    # 응답 시간 체크 (3회)
    print(f"\n  ── 레이턴시 측정 (3회) ──")
    latencies = []
    for i in range(3):
        t0 = time.time()
        resp2 = await client.get(url) if False else await httpx.AsyncClient(timeout=10).get(url)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        print(f"  {i+1}회: {lat:.0f}ms")
        await asyncio.sleep(0.5)

    avg_lat = sum(latencies) / len(latencies)
    print(f"  평균: {avg_lat:.0f}ms")

    if avg_lat > 3000:
        print(f"\n  ⚠️  레이턴시가 높습니다 (>{avg_lat:.0f}ms). 폴링 간격 조정 필요.")
    else:
        print(f"\n  ✅ 레이턴시 양호 ({avg_lat:.0f}ms < 3000ms)")

    print(f"\n  {'='*60}")
    print(f"  연결 테스트 완료 ✅")
    print(f"  {'='*60}")
    return True


# ═══════════════════════════════════════════════════════════
# 2단계: 라이브 경기 목록
# ═══════════════════════════════════════════════════════════

async def list_matches(api_key: str, live_only: bool = True):
    """현재 경기 목록 출력."""
    url = f"{GOALSERVE_BASE_URL}/{api_key}/soccernew/live?json=1"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        data = resp.json()

    if live_only:
        matches = GoalserveSource.find_all_live_matches(data)
        title = "라이브 경기"
    else:
        matches = GoalserveSource.find_all_matches(data)
        title = "전체 경기"

    if not matches:
        print(f"\n  현재 {title} 없음")
        if live_only:
            print("  '--all' 옵션으로 FT 포함 전체 목록 확인 가능")
        return

    # 리그별 그룹핑
    by_league = {}
    for m in matches:
        league = m["league"]
        if league not in by_league:
            by_league[league] = []
        by_league[league].append(m)

    print(f"\n{'='*70}")
    print(f"  {title}: {len(matches)}개 ({len(by_league)} 리그)")
    print(f"{'='*70}")

    for league, league_matches in sorted(by_league.items()):
        print(f"\n  📋 {league}")
        print(f"  {'─'*60}")
        for m in league_matches:
            status_icon = "🔴" if m["status"] not in ("FT", "HT") else ("⏸️" if m["status"] == "HT" else "✅")
            timer_str = f"{m['timer']}'" if m["timer"] else m["status"]
            print(f"  {status_icon} [{m['match_id']:>8}] {m['home']:>20} {m['score']:^5} {m['away']:<20} ({timer_str})")

    print(f"\n  특정 경기 이벤트 수신:")
    print(f"  python tools/goalserve_verify.py --watch <match_id>")
    print(f"  python tools/goalserve_verify.py --dump <match_id>")


# ═══════════════════════════════════════════════════════════
# 3단계: 특정 경기 이벤트 실시간 수신
# ═══════════════════════════════════════════════════════════

async def watch_match(api_key: str, match_id: str):
    """특정 경기의 이벤트를 실시간으로 수신."""
    print(f"\n{'='*60}")
    print(f"  경기 {match_id} 이벤트 수신 중... (Ctrl+C로 종료)")
    print(f"{'='*60}")

    source = GoalserveSource(api_key=api_key, poll_interval=3.0)
    await source.connect(match_id)

    print(f"\n  {source._home_team_name} vs {source._away_team_name}")
    print(f"  현재: {source._prev_home_goals}-{source._prev_away_goals} ({source._prev_status})")
    print(f"  기존 이벤트: {len(source._seen_event_ids)}개 (무시됨)")
    print(f"\n  {'─'*50}")
    print(f"  새 이벤트 대기 중 (3초 폴링)...")
    print(f"  {'─'*50}\n")

    event_count = 0
    poll_count = 0

    try:
        async for event in source.listen():
            event_count += 1
            ts = datetime.now().strftime("%H:%M:%S")

            icon = {
                EventType.GOAL: "⚽",
                EventType.RED_CARD: "🟥",
                EventType.HALFTIME: "⏸️",
                EventType.SECOND_HALF_START: "▶️",
                EventType.STOPPAGE_ENTERED: "⏱️",
                EventType.MATCH_END: "🏁",
            }.get(event.event_type, "❓")

            print(f"  [{ts}] {icon} {event.event_type.value} "
                  f"{'['+event.team+'] ' if event.team else ''}"
                  f"@ {event.minute:.0f}min")
            if event.raw:
                for k, v in event.raw.items():
                    if v:
                        print(f"           {k}: {v}")

            if event.event_type == EventType.MATCH_END:
                break

    except KeyboardInterrupt:
        pass
    finally:
        await source.disconnect()
        print(f"\n  수신 완료: {event_count}개 이벤트")


# ═══════════════════════════════════════════════════════════
# 4단계: JSON 원본 덤프
# ═══════════════════════════════════════════════════════════

async def dump_match(api_key: str, match_id: str):
    """특정 경기의 원본 JSON을 출력."""
    url = f"{GOALSERVE_BASE_URL}/{api_key}/soccernew/live?json=1"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        data = resp.json()

    match_data = GoalserveSource._find_match(data, match_id)

    if not match_data:
        print(f"\n  ❌ match_id={match_id} 못 찾음")
        print(f"  '--all' 옵션으로 사용 가능한 경기 확인")
        return

    print(f"\n{'='*60}")
    print(f"  경기 {match_id} JSON 덤프")
    print(f"{'='*60}\n")
    print(json.dumps(match_data, indent=2, ensure_ascii=False))

    # 이벤트 요약
    events_data = match_data.get("events")
    if events_data:
        event_list = events_data.get("event", [])
        if isinstance(event_list, dict):
            event_list = [event_list]
        print(f"\n  ── 이벤트 요약 ({len(event_list)}개) ──")
        for evt in event_list:
            etype = evt.get("@type", "?")
            team = evt.get("@team", "?")
            minute = evt.get("@minute", "?")
            extra = evt.get("@extra_min", "")
            player = evt.get("@player", "")
            result = evt.get("@result", "")
            extra_str = f"+{extra}" if extra else ""
            print(f"  {minute}{extra_str}' [{team:>12}] {etype:<12} {player} {result}")


# ═══════════════════════════════════════════════════════════
# 5단계: FT 경기로 이벤트 파싱 검증
# ═══════════════════════════════════════════════════════════

async def validate_ft_matches(api_key: str):
    """종료된(FT) 경기의 이벤트를 파싱하여 정확도 검증."""
    url = f"{GOALSERVE_BASE_URL}/{api_key}/soccernew/live?json=1"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        data = resp.json()

    all_matches = GoalserveSource.find_all_matches(data)
    ft_matches = [m for m in all_matches if m["status"] == "FT"]

    if not ft_matches:
        print("\n  FT 경기 없음. 경기가 끝난 후에 다시 시도하세요.")
        return

    print(f"\n{'='*70}")
    print(f"  FT 경기 이벤트 파싱 검증 ({len(ft_matches)}개)")
    print(f"{'='*70}")

    passed = 0
    failed = 0

    for m_info in ft_matches[:10]:  # 최대 10개
        mid = m_info["match_id"]
        match_data = GoalserveSource._find_match(data, mid)
        if not match_data:
            continue

        home = match_data.get("localteam", {}).get("@name", "?")
        away = match_data.get("visitorteam", {}).get("@name", "?")
        h_goals = int(match_data.get("localteam", {}).get("@goals", "0"))
        a_goals = int(match_data.get("visitorteam", {}).get("@goals", "0"))

        # GoalserveSource로 이벤트 파싱
        source = GoalserveSource(api_key="test")
        events_data = match_data.get("events")
        parsed_goals_home = 0
        parsed_goals_away = 0
        parsed_reds = 0
        parse_errors = []

        if events_data:
            event_list = events_data.get("event", [])
            if isinstance(event_list, dict):
                event_list = [event_list]

            for evt in event_list:
                try:
                    normalized = source._parse_event(evt)
                    if normalized:
                        if normalized.event_type == EventType.GOAL:
                            if normalized.team == "home":
                                parsed_goals_home += 1
                            elif normalized.team == "away":
                                parsed_goals_away += 1
                        elif normalized.event_type == EventType.RED_CARD:
                            parsed_reds += 1
                except Exception as e:
                    parse_errors.append(str(e))

        # 자책골 보정: 자책골은 상대팀 점수에 기여하지만
        # Goalserve에서 @team이 자책골을 넣은 팀이므로 반전 필요
        # 실제 스코어와 파싱된 골 수 비교
        # (자책골은 raw["own_goal"]로 처리됨)

        # 검증
        ok = True
        issues = []

        # 골 수는 자책골 때문에 정확히 일치 안 할 수 있음
        # 대신 총 골 수가 맞는지 확인
        total_parsed = parsed_goals_home + parsed_goals_away
        total_actual = h_goals + a_goals

        if total_parsed != total_actual:
            ok = False
            issues.append(f"골 수 불일치: 파싱={total_parsed} vs 실제={total_actual}")

        if parse_errors:
            ok = False
            issues.append(f"파싱 에러 {len(parse_errors)}건")

        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n  {status} [{mid}] {home} {h_goals}-{a_goals} {away}")
        print(f"     파싱: 골 {total_parsed}개 (H:{parsed_goals_home} A:{parsed_goals_away}), 퇴장 {parsed_reds}개")
        if issues:
            for issue in issues:
                print(f"     ⚠️  {issue}")

    print(f"\n  {'─'*60}")
    print(f"  결과: {passed} 통과 / {failed} 실패 (총 {passed+failed})")
    print(f"  {'='*70}")


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Goalserve 라이브 데이터 검증")
    parser.add_argument("--check", action="store_true", help="API 연결 테스트")
    parser.add_argument("--live", action="store_true", help="라이브 경기 목록")
    parser.add_argument("--all", action="store_true", help="전체 경기 목록 (FT 포함)")
    parser.add_argument("--watch", type=str, metavar="MATCH_ID", help="특정 경기 이벤트 수신")
    parser.add_argument("--dump", type=str, metavar="MATCH_ID", help="특정 경기 JSON 덤프")
    parser.add_argument("--validate-ft", action="store_true", help="FT 경기 파싱 검증")
    args = parser.parse_args()

    if not any([args.check, args.live, args.all, args.watch, args.dump, args.validate_ft]):
        parser.print_help()
        return

    api_key = get_api_key()

    if args.check:
        asyncio.run(check_connection(api_key))
    elif args.live:
        asyncio.run(list_matches(api_key, live_only=True))
    elif args.all:
        asyncio.run(list_matches(api_key, live_only=False))
    elif args.watch:
        asyncio.run(watch_match(api_key, args.watch))
    elif args.dump:
        asyncio.run(dump_match(api_key, args.dump))
    elif args.validate_ft:
        asyncio.run(validate_ft_matches(api_key))


if __name__ == "__main__":
    main()

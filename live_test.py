"""
Goalserve 라이브 폴링 테스트.

사용법:
  python live_test.py                    # 모든 라이브 경기 목록 출력
  python live_test.py 6678380            # 특정 경기 이벤트 수신
"""

import asyncio
import sys
import logging
import httpx
from src.phase3.goalserve import GoalserveSource

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

API_KEY = "59e68707a50347e8fa7508de75e150c9"


async def list_live_matches():
    """현재 라이브 경기 목록 출력."""
    url = f"http://www.goalserve.com/getfeed/{API_KEY}/soccernew/live?json=1"
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(3):
            try:
                resp = await client.get(url)
                data = resp.json()
                break
            except Exception as e:
                print(f"  요청 실패 ({attempt+1}/3): {type(e).__name__}")
                if attempt == 2:
                    print("  3회 실패. 나중에 다시 시도해줘.")
                    return
                await asyncio.sleep(3)

    matches = GoalserveSource.find_all_live_matches(data)

    if not matches:
        print("현재 라이브 경기 없음")
        return

    print(f"\n{'='*60}")
    print(f"  라이브 경기: {len(matches)}개")
    print(f"{'='*60}\n")

    for m in matches:
        print(f"  [{m['match_id']}] {m['home']} {m['score']} {m['away']}")
        print(f"           {m['league']} | {m['status']}분")
        print()

    print("특정 경기 수신:")
    print(f"  python live_test.py <match_id>")


async def watch_match(match_id: str):
    """특정 경기의 이벤트를 실시간으로 수신."""
    print(f"\n경기 {match_id} 수신 중... (Ctrl+C로 종료)\n")

    source = GoalserveSource(api_key=API_KEY, poll_interval=3.0)
    await source.connect(match_id)

    print(f"  {source._home_team_name} vs {source._away_team_name}")
    print(f"  현재: {source._prev_home_goals}-{source._prev_away_goals} (timer={source._prev_timer}, status={source._prev_status})")
    print(f"  기존 이벤트: {len(source._seen_event_ids)}개 (무시됨)")
    print(f"\n{'─'*50}\n")

    last_minute = ""
    try:
        while not source._match_end_sent:
            match_data = await source._poll_match()

            if match_data:
                timer = match_data.get("@timer", "?")
                status = match_data.get("@status", "?")
                home_g = match_data.get("localteam", {}).get("@goals", "0")
                away_g = match_data.get("visitorteam", {}).get("@goals", "0")

                # 분이 바뀌었을 때만 표시
                if timer != last_minute:
                    tag = ""
                    if status == "HT":
                        tag = " ⏸ 하프타임"
                    elif status == "FT":
                        tag = " 🏁 경기종료"
                    print(f"  ⏱ {timer}분  [{home_g}-{away_g}]{tag}")
                    last_minute = timer

                # 이벤트 diff
                async for event in source._diff(match_data):
                    dm = event.raw.get("display_minute", "")
                    if dm:
                        print(f"  ⚡ {event.event_type.value} [{event.team}] @ {dm}분")
                    else:
                        print(f"  ⚡ {event}")

            await asyncio.sleep(source.poll_interval)

    except KeyboardInterrupt:
        pass
    finally:
        await source.disconnect()
        print("\n종료")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(watch_match(sys.argv[1]))
    else:
        asyncio.run(list_live_matches())
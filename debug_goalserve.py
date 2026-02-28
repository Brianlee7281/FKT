"""
Goalserve 폴링 디버깅 스크립트.

사용법:
  python debug_goalserve.py
"""

import asyncio
import httpx

API_KEY = "59e68707a50347e8fa7508de75e150c9"
URL = f"http://www.goalserve.com/getfeed/{API_KEY}/soccernew/live?json=1"


async def main():
    print(f"URL: {URL}")
    print()

    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1차 요청
        print("=== 1차 요청 ===")
        try:
            resp = await client.get(URL)
            print(f"  Status: {resp.status_code}")
            print(f"  Size: {len(resp.content)} bytes")
            data = resp.json()
            cats = data.get("scores", {}).get("category", [])
            if isinstance(cats, dict):
                cats = [cats]
            total = sum(
                len(cat.get("matches", {}).get("match", []))
                if isinstance(cat.get("matches", {}).get("match", []), list)
                else 1
                for cat in cats
            )
            print(f"  Categories: {len(cats)}, Total matches: ~{total}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

        # 3초 대기
        print("\n  3초 대기...")
        await asyncio.sleep(3)

        # 2차 요청
        print("\n=== 2차 요청 ===")
        try:
            resp = await client.get(URL)
            print(f"  Status: {resp.status_code}")
            print(f"  Size: {len(resp.content)} bytes")

            # 특정 경기 찾기
            data = resp.json()
            cats = data.get("scores", {}).get("category", [])
            if isinstance(cats, dict):
                cats = [cats]

            test_ids = ["6678380", "6700989"]
            for mid in test_ids:
                found = False
                for cat in cats:
                    matches = cat.get("matches", {}).get("match", [])
                    if isinstance(matches, dict):
                        matches = [matches]
                    for m in matches:
                        if m.get("@id") == mid:
                            print(f"\n  Match {mid}: {m['localteam']['@name']} vs {m['visitorteam']['@name']}")
                            print(f"    status={m['@status']} timer={m['@timer']}")
                            print(f"    score={m['localteam']['@goals']}-{m['visitorteam']['@goals']}")
                            found = True
                            break
                    if found:
                        break
                if not found:
                    print(f"\n  Match {mid}: NOT FOUND in live feed")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

        # 3차 — 빠르게 연속 요청
        print("\n=== 3차 요청 (연속) ===")
        try:
            resp = await client.get(URL)
            print(f"  Status: {resp.status_code}")
            print(f"  Size: {len(resp.content)} bytes")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")


asyncio.run(main())
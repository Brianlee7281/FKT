"""
Replay Paper Trading 배치 도구.

1. DB에서 거래 가능한 경기 목록 조회
2. 여러 경기를 한 번에 시뮬레이션
3. 결과 요약 테이블 출력

사용법:
  # 1단계: 경기 목록 확인
  python tools/replay_batch.py --db data/kalshi_football.db --list

  # 2단계: 특정 리그만
  python tools/replay_batch.py --db data/kalshi_football.db --list --league 39

  # 3단계: 경기 여러 개 시뮬레이션
  python tools/replay_batch.py --db data/kalshi_football.db --fixtures 1035068,1035069,1035070

  # 4단계: 리그 전체 (최대 N경기)
  python tools/replay_batch.py --db data/kalshi_football.db --league 39 --max 20

  # 5단계: 전체 DB (최대 N경기)
  python tools/replay_batch.py --db data/kalshi_football.db --all --max 50
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import sqlite3
from typing import Dict, List


def list_fixtures(db_path: str, league_id: int = None, limit: int = 100) -> List[Dict]:
    """DB에서 이벤트가 있는 경기 목록 조회."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            m.fixture_id,
            m.league_id,
            m.league_name,
            m.season,
            m.match_date,
            m.home_team_name,
            m.away_team_name,
            m.home_goals_ft,
            m.away_goals_ft,
            m.home_goals_ht,
            m.away_goals_ht,
            m.elapsed_minutes,
            COUNT(CASE WHEN e.event_type = 'Goal' THEN 1 END) as goal_count,
            COUNT(CASE WHEN e.event_type = 'Card'
                  AND e.event_detail IN ('Red Card', 'Second Yellow card')
                  THEN 1 END) as red_count,
            COUNT(e.id) as total_events
        FROM matches m
        LEFT JOIN match_events e ON m.fixture_id = e.fixture_id
            AND (e.event_type = 'Goal'
                 OR (e.event_type = 'Card'
                     AND e.event_detail IN ('Red Card', 'Second Yellow card')))
        WHERE m.home_goals_ft IS NOT NULL
    """
    params = []
    if league_id:
        query += " AND m.league_id = ?"
        params.append(league_id)

    query += """
        GROUP BY m.fixture_id
        HAVING goal_count > 0
        ORDER BY m.match_date DESC, m.fixture_id
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    fixtures = []
    for r in rows:
        result = f"{r['home_goals_ft']}-{r['away_goals_ft']}"
        if r['home_goals_ft'] > r['away_goals_ft']:
            outcome = "H"
        elif r['home_goals_ft'] < r['away_goals_ft']:
            outcome = "A"
        else:
            outcome = "D"

        fixtures.append({
            "fixture_id": r["fixture_id"],
            "league_id": r["league_id"],
            "league": r["league_name"] or "?",
            "season": r["season"],
            "date": r["match_date"],
            "home": r["home_team_name"],
            "away": r["away_team_name"],
            "ft": result,
            "ht": f"{r['home_goals_ht']}-{r['away_goals_ht']}",
            "outcome": outcome,
            "goals": r["goal_count"],
            "reds": r["red_count"],
            "elapsed": r["elapsed_minutes"],
        })

    return fixtures


def list_leagues(db_path: str) -> List[Dict]:
    """DB에 있는 리그 목록."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT league_id, league_name, season,
               COUNT(*) as match_count
        FROM matches
        WHERE home_goals_ft IS NOT NULL
        GROUP BY league_id, season
        ORDER BY match_count DESC
    """).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "season": r[2], "matches": r[3]} for r in rows]


def print_fixtures(fixtures: List[Dict]):
    """경기 목록 출력."""
    print(f"\n{'─'*100}")
    print(f"  {'ID':>9}  {'Date':10}  {'League':20}  {'Home':20}  {'Away':20}  FT   HT   G  R  Min")
    print(f"{'─'*100}")
    for f in fixtures:
        print(
            f"  {f['fixture_id']:>9}  {f['date']:10}  {f['league'][:20]:20}  "
            f"{f['home'][:20]:20}  {f['away'][:20]:20}  "
            f"{f['ft']:4}  {f['ht']:4}  {f['goals']}  {f['reds']}  {f['elapsed'] or '?'}"
        )
    print(f"{'─'*100}")
    print(f"  Total: {len(fixtures)} fixtures")


async def run_batch(
    db_path: str,
    fixture_ids: List[str],
    bankroll: float = 5000,
    misprice_prob: float = 0.20,
    misprice_bias: float = 0.12,
    entry_threshold: float = 0.02,
    kelly_fraction: float = 0.25,
):
    """여러 경기 배치 시뮬레이션."""
    # 동적 임포트 (DB 없어도 --list는 돌아가게)
    from tests.test_replay_paper import run_replay_paper

    results = []
    for i, fid in enumerate(fixture_ids):
        print(f"\n{'═'*60}")
        print(f"  [{i+1}/{len(fixture_ids)}] fixture_id={fid}")
        print(f"{'═'*60}")

        try:
            r = await run_replay_paper(
                db_path=db_path,
                fixture_id=fid,
                bankroll=bankroll,
                misprice_prob=misprice_prob,
                misprice_bias=misprice_bias,
                entry_threshold=entry_threshold,
                kelly_fraction=kelly_fraction,
                seed=int(fid) % 10000,  # fixture_id 기반 시드 (재현 가능)
            )
            r["fixture_id"] = fid
            r["status"] = "OK"
            results.append(r)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({"fixture_id": fid, "status": "ERROR", "error": str(e)})

    # ── 요약 테이블 ──────────────────────────
    print(f"\n\n{'═'*80}")
    print(f"  BATCH RESULTS SUMMARY ({len(results)} matches)")
    print(f"{'═'*80}")
    print(f"  {'Fixture':>9}  {'Status':6}  {'Ticks':>5}  {'Trades':>6}  "
          f"{'Entries':>7}  {'Exits':>5}  {'P&L':>10}  {'DD':>6}")
    print(f"{'─'*80}")

    total_pnl = 0
    total_trades = 0
    ok_count = 0
    win_count = 0

    for r in results:
        fid = r["fixture_id"]
        if r["status"] == "ERROR":
            print(f"  {fid:>9}  {'ERROR':6}  {'─':>5}  {'─':>6}  {'─':>7}  {'─':>5}  {'─':>10}  {'─':>6}")
            continue

        ok_count += 1
        s = r["summary"]
        pnl = r["pnl"]
        total_pnl += pnl
        total_trades += r["trades"]
        if pnl > 0:
            win_count += 1

        dd_str = f"{r.get('summary', {}).get('drawdown', 0):.1f}%"

        print(
            f"  {fid:>9}  {'OK':6}  {s['ticks']:>5}  {r['trades']:>6}  "
            f"{s['entries']:>7}  {s['exits']:>5}  "
            f"${pnl:>+9.2f}  {dd_str:>6}"
        )

    print(f"{'─'*80}")
    if ok_count > 0:
        print(f"  Total: {ok_count} matches, {total_trades} trades, "
              f"P&L=${total_pnl:+.2f}, "
              f"Win Rate={win_count}/{ok_count} ({win_count/ok_count*100:.0f}%)")
        print(f"  Avg P&L/match: ${total_pnl/ok_count:+.2f}")
    print(f"{'═'*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay Paper Trading 배치 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 경기 목록 확인
  python tools/replay_batch.py --db data/kalshi_football.db --list

  # EPL만 (league_id=39)
  python tools/replay_batch.py --db data/kalshi_football.db --list --league 39

  # 특정 경기들
  python tools/replay_batch.py --db data/kalshi_football.db --fixtures 1035068,1035069

  # 리그 전체 (최대 20경기)
  python tools/replay_batch.py --db data/kalshi_football.db --league 39 --max 20 --run

  # 전체 DB
  python tools/replay_batch.py --db data/kalshi_football.db --all --max 50 --run
        """,
    )
    parser.add_argument("--db", required=True, help="SQLite DB path")
    parser.add_argument("--list", action="store_true", help="경기 목록만 출력")
    parser.add_argument("--leagues", action="store_true", help="리그 목록 출력")
    parser.add_argument("--league", type=int, help="리그 ID 필터")
    parser.add_argument("--fixtures", help="쉼표로 구분된 fixture_id 목록")
    parser.add_argument("--all", action="store_true", help="전체 DB 시뮬레이션")
    parser.add_argument("--run", action="store_true", help="시뮬레이션 실행")
    parser.add_argument("--max", type=int, default=20, help="최대 경기 수 (기본 20)")
    parser.add_argument("--bankroll", type=float, default=5000)
    parser.add_argument("--misprice", type=float, default=0.20)
    parser.add_argument("--bias", type=float, default=0.12)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--kelly", type=float, default=0.25)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"❌ DB 파일 없음: {args.db}")
        sys.exit(1)

    # 리그 목록
    if args.leagues:
        leagues = list_leagues(args.db)
        print(f"\n{'─'*60}")
        print(f"  {'ID':>5}  {'League':30}  {'Season':>6}  {'Matches':>7}")
        print(f"{'─'*60}")
        for lg in leagues:
            print(f"  {lg['id']:>5}  {lg['name'][:30]:30}  {lg['season']:>6}  {lg['matches']:>7}")
        print(f"{'─'*60}")
        return

    # 경기 목록
    if args.list:
        fixtures = list_fixtures(args.db, league_id=args.league, limit=args.max)
        print_fixtures(fixtures)

        # fixture_id 목록 출력 (복사용)
        ids = ",".join(str(f["fixture_id"]) for f in fixtures)
        print(f"\n  Copy-paste for --fixtures:")
        print(f"  {ids}")
        return

    # 시뮬레이션 실행
    if args.fixtures:
        fixture_ids = [fid.strip() for fid in args.fixtures.split(",")]
    elif args.all or args.run:
        fixtures = list_fixtures(args.db, league_id=args.league, limit=args.max)
        fixture_ids = [str(f["fixture_id"]) for f in fixtures]
        print(f"  선택된 경기: {len(fixture_ids)}개")
    else:
        parser.print_help()
        print("\n  💡 먼저 --list로 경기 목록을 확인하세요!")
        return

    asyncio.run(run_batch(
        db_path=args.db,
        fixture_ids=fixture_ids,
        bankroll=args.bankroll,
        misprice_prob=args.misprice,
        misprice_bias=args.bias,
        entry_threshold=args.threshold,
        kelly_fraction=args.kelly,
    ))


if __name__ == "__main__":
    main()

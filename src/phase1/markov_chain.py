"""
Step 1.2: 마르코프 체인 생성 행렬 Q 추정.

레드카드 발생률(상태 전이율)을 과거 데이터에서 추정하여 4×4 생성 행렬 Q를 구성한다.

상태 공간:
  0 = 11v11 (평상시)
  1 = 10v11 (홈 퇴장)
  2 = 11v10 (어웨이 퇴장)
  3 = 10v10 (양팀 퇴장)

추정 방법:
  q_ij = N_ij / (상태 i에서의 총 체류 시간)
  상태 3은 희소 → 가산 가정으로 합성

입력:
  - intervals 테이블 (preprocessor.py가 생성)
  - match_events 테이블 (api_football.py가 생성)

출력:
  - data/models/Q_matrix.json (전체 + 리그별)

사용법:
  python -m src.phase1.markov_chain
  python -m src.phase1.markov_chain --league 39    # 프리미어리그만
"""

import json
import sqlite3
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

DB_PATH = Path("data/kalshi_football.db")
OUTPUT_DIR = Path("data/models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/markov_chain.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    135: "Serie A",
    78:  "Bundesliga",
    61:  "Ligue 1",
}

# 허용 전이 (레드카드만)
# 0→1: 홈 퇴장,  0→2: 어웨이 퇴장
# 1→3: 어웨이 추가 퇴장,  2→3: 홈 추가 퇴장
VALID_TRANSITIONS = [(0, 1), (0, 2), (1, 3), (2, 3)]

STATE_NAMES = {0: "11v11", 1: "10v11(홈↓)", 2: "11v10(어↓)", 3: "10v10"}


# ─────────────────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────────────────

def collect_dwell_times(conn: sqlite3.Connection, league_id: Optional[int] = None) -> Dict[int, float]:
    """
    각 상태에서의 총 체류 시간(분)을 집계한다.

    intervals 테이블에서 is_halftime=0인 구간만 사용.
    이것이 q_ij 분모가 된다.
    """
    query = """
        SELECT i.state_X, SUM(i.duration)
        FROM intervals i
    """
    params = []

    if league_id is not None:
        query += """
            JOIN matches m ON i.fixture_id = m.fixture_id
            WHERE i.is_halftime = 0 AND m.league_id = ?
        """
        params.append(league_id)
    else:
        query += " WHERE i.is_halftime = 0"

    query += " GROUP BY i.state_X"

    cursor = conn.cursor()
    cursor.execute(query, params)

    dwell = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for state_x, total_dur in cursor.fetchall():
        dwell[state_x] = total_dur

    return dwell


def collect_transitions(conn: sqlite3.Connection, league_id: Optional[int] = None) -> Dict[Tuple[int, int], int]:
    """
    상태 전이 횟수를 집계한다.

    match_events에서 레드카드를 찾고,
    해당 시점의 이전 상태 → 이후 상태를 intervals에서 역추적한다.

    더 간단한 방법: intervals 테이블에서 같은 경기 내에서
    연속된 두 구간의 state_X가 다르면 전이가 발생한 것.
    """
    # 경기별로 구간을 시간순 정렬 → 연속된 구간의 X 비교
    query = """
        SELECT i.fixture_id, i.t_start, i.state_X
        FROM intervals i
    """
    params = []

    if league_id is not None:
        query += """
            JOIN matches m ON i.fixture_id = m.fixture_id
            WHERE i.is_halftime = 0 AND m.league_id = ?
        """
        params.append(league_id)
    else:
        query += " WHERE i.is_halftime = 0"

    query += " ORDER BY i.fixture_id, i.t_start"

    cursor = conn.cursor()
    cursor.execute(query, params)

    transitions = defaultdict(int)
    prev_fixture = None
    prev_state = None

    for fixture_id, t_start, state_x in cursor.fetchall():
        if fixture_id == prev_fixture and prev_state is not None:
            if state_x != prev_state:
                pair = (prev_state, state_x)
                transitions[pair] += 1

        prev_fixture = fixture_id
        prev_state = state_x

    return dict(transitions)


# ─────────────────────────────────────────────────────────
# Q 행렬 추정
# ─────────────────────────────────────────────────────────

def estimate_Q(
    dwell_times: Dict[int, float],
    transitions: Dict[Tuple[int, int], int],
    use_additivity: bool = True,
) -> np.ndarray:
    """
    4×4 생성 행렬 Q를 추정한다.

    q_ij = N_ij / T_i  (전이 횟수 / 체류 시간)

    상태 3(10v10)은 데이터가 극히 희소하므로,
    가산 가정(Additivity Assumption)을 적용:
      q_{1→3} ≈ q_{0→2}  (이미 홈 퇴장 상태에서 어웨이 추가 퇴장율 ≈ 평상시 어웨이 퇴장율)
      q_{2→3} ≈ q_{0→1}  (이미 어웨이 퇴장 상태에서 홈 추가 퇴장율 ≈ 평상시 홈 퇴장율)

    Args:
        dwell_times: 상태별 총 체류 시간 (분)
        transitions: (i,j) → 전이 횟수
        use_additivity: 상태 3 가산 가정 사용 여부

    Returns:
        Q: 4×4 생성 행렬 (단위: 1/분)
    """
    Q = np.zeros((4, 4))

    # ── 상태 0 → 1, 0 → 2: 직접 추정 (데이터 충분) ──
    T_0 = dwell_times.get(0, 0)
    if T_0 > 0:
        Q[0, 1] = transitions.get((0, 1), 0) / T_0
        Q[0, 2] = transitions.get((0, 2), 0) / T_0

    # ── 상태 1 → 3, 2 → 3 ──
    if use_additivity:
        # 가산 가정: 이미 한쪽이 퇴장된 상태에서 추가 퇴장율은
        # 평상시 퇴장율과 동일하다고 가정
        Q[1, 3] = Q[0, 2]   # 홈 퇴장 상태에서 어웨이 추가 퇴장율
        Q[2, 3] = Q[0, 1]   # 어웨이 퇴장 상태에서 홈 추가 퇴장율
    else:
        # 직접 추정 (데이터 부족 시 분산 큼)
        T_1 = dwell_times.get(1, 0)
        T_2 = dwell_times.get(2, 0)
        if T_1 > 0:
            Q[1, 3] = transitions.get((1, 3), 0) / T_1
        if T_2 > 0:
            Q[2, 3] = transitions.get((2, 3), 0) / T_2

    # ── 대각 성분: q_ii = -Σ_{j≠i} q_ij ──
    for i in range(4):
        Q[i, i] = -sum(Q[i, j] for j in range(4) if j != i)

    return Q


# ─────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────

def validate_Q(Q: np.ndarray, label: str = ""):
    """Q 행렬의 기본 속성 검증"""
    prefix = f"[{label}] " if label else ""
    ok = True

    # 1. 행 합 = 0
    row_sums = Q.sum(axis=1)
    if not np.allclose(row_sums, 0, atol=1e-10):
        logger.warning(f"{prefix}행 합이 0이 아님: {row_sums}")
        ok = False

    # 2. 비대각 원소 ≥ 0
    for i in range(4):
        for j in range(4):
            if i != j and Q[i, j] < 0:
                logger.warning(f"{prefix}q[{i},{j}] = {Q[i,j]:.6f} < 0")
                ok = False

    # 3. 대각 원소 ≤ 0
    for i in range(4):
        if Q[i, i] > 0:
            logger.warning(f"{prefix}q[{i},{i}] = {Q[i,i]:.6f} > 0")
            ok = False

    # 4. 경기당 퇴장 기대값이 합리적인지
    # 평균 경기 시간 ~93분 기준, 경기당 기대 퇴장 = (q01 + q02) × 93
    expected_reds = (Q[0, 1] + Q[0, 2]) * 93
    if expected_reds < 0.01 or expected_reds > 1.0:
        logger.warning(f"{prefix}경기당 기대 퇴장수 = {expected_reds:.3f} (범위 밖)")
        ok = False

    return ok


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def run(league_id: Optional[int] = None):
    """Q 행렬 추정 실행"""
    conn = sqlite3.connect(str(DB_PATH))

    try:
        results = {}

        # ── 전체 풀 추정 ─────────────────────────────────
        logger.info("📊 Step 1.2: Q 행렬 추정")
        logger.info("=" * 60)

        dwell = collect_dwell_times(conn)
        trans = collect_transitions(conn)
        Q_all = estimate_Q(dwell, trans)

        print_report("전체 (5대 리그 통합)", dwell, trans, Q_all)
        validate_Q(Q_all, "전체")

        results["all"] = {
            "Q": Q_all.tolist(),
            "dwell_minutes": dwell,
            "transitions": {f"{k[0]}->{k[1]}": v for k, v in trans.items()},
        }

        # ── 리그별 추정 ──────────────────────────────────
        leagues_to_process = {league_id: LEAGUES.get(league_id, f"League {league_id}")} if league_id else LEAGUES

        for lid, lname in leagues_to_process.items():
            dwell_l = collect_dwell_times(conn, lid)
            trans_l = collect_transitions(conn, lid)
            Q_l = estimate_Q(dwell_l, trans_l)

            print_report(lname, dwell_l, trans_l, Q_l)
            validate_Q(Q_l, lname)

            results[str(lid)] = {
                "league_name": lname,
                "Q": Q_l.tolist(),
                "dwell_minutes": dwell_l,
                "transitions": {f"{k[0]}->{k[1]}": v for k, v in trans_l.items()},
            }

        # ── 저장 ─────────────────────────────────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "Q_matrix.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ 저장 완료: {output_path}")

    finally:
        conn.close()


def print_report(
    label: str,
    dwell: Dict[int, float],
    trans: Dict[Tuple[int, int], int],
    Q: np.ndarray,
):
    """추정 결과를 보기 좋게 출력"""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    # 체류 시간
    print(f"\n  상태별 체류 시간:")
    total_minutes = sum(dwell.values())
    for s in range(4):
        t = dwell.get(s, 0)
        pct = t / total_minutes * 100 if total_minutes > 0 else 0
        print(f"    X={s} ({STATE_NAMES[s]}): {t:>10,.0f}분 ({pct:5.1f}%)")

    # 전이 횟수
    print(f"\n  관측된 전이 횟수:")
    for (i, j), n in sorted(trans.items()):
        print(f"    {STATE_NAMES[i]} → {STATE_NAMES[j]}: {n}회")

    # Q 행렬
    print(f"\n  Q 행렬 (단위: 1/분):")
    print(f"  {'':>12}", end="")
    for j in range(4):
        print(f"  {'X='+str(j):>10}", end="")
    print()

    for i in range(4):
        print(f"    X={i}      ", end="")
        for j in range(4):
            val = Q[i, j]
            if i == j:
                print(f"  {val:>10.6f}", end="")
            elif val > 0:
                print(f"  {val:>10.6f}", end="")
            else:
                print(f"  {'—':>10}", end="")
        print()

    # 해석 가능한 지표
    total_time_hours = total_minutes / 60
    n_matches_approx = total_minutes / 93  # 평균 93분/경기

    q_home_red = Q[0, 1]
    q_away_red = Q[0, 2]
    expected_per_match = (q_home_red + q_away_red) * 93
    avg_minutes_to_red = 1 / (q_home_red + q_away_red) if (q_home_red + q_away_red) > 0 else float('inf')

    print(f"\n  해석:")
    print(f"    총 플레이 시간: {total_time_hours:,.0f}시간 (~{n_matches_approx:,.0f}경기)")
    print(f"    홈 퇴장율: {q_home_red:.6f}/분 = 경기당 {q_home_red*93:.4f}회")
    print(f"    어웨이 퇴장율: {q_away_red:.6f}/분 = 경기당 {q_away_red*93:.4f}회")
    print(f"    경기당 기대 퇴장: {expected_per_match:.3f}회")
    print(f"    퇴장 간 평균 간격: {avg_minutes_to_red:.0f}분 ({avg_minutes_to_red/93:.1f}경기)")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1.2: 마르코프 체인 Q 행렬 추정",
    )
    parser.add_argument("--league", type=int, help="특정 리그만 (예: 39)")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()
    run(league_id=args.league)


if __name__ == "__main__":
    main()
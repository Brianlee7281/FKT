"""
engine.py 통합 테스트

ReplaySource + 인메모리 DB로 전체 경기를 엔진에 흘려보내
모든 컴포넌트가 올바르게 연결되는지 검증한다.

실행:
  python -m pytest tests/test_engine.py -v
  python tests/test_engine.py
"""

import asyncio
import math
import sqlite3
import tempfile
import os

import numpy as np
from scipy.linalg import expm

from src.phase3.engine import LiveTradingEngine, EngineParams, TickSnapshot
from src.phase3.replay import ReplaySource
from src.phase3.mu_calculator import build_gamma_array, build_delta_array, build_basis_bounds
from src.phase3.mc_core import build_Q_diag_and_off


# ═════════════════════════════════════════════════════════
# 테스트 DB + 파라미터 생성
# ═════════════════════════════════════════════════════════

def create_test_db(db_path: str) -> None:
    """테스트용 DB (Man City 2-1 Arsenal 시나리오)"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            fixture_id INTEGER PRIMARY KEY,
            league_id INTEGER NOT NULL,
            league_name TEXT,
            season INTEGER NOT NULL,
            match_date TEXT NOT NULL,
            round TEXT,
            home_team_id INTEGER NOT NULL,
            home_team_name TEXT NOT NULL,
            away_team_id INTEGER NOT NULL,
            away_team_name TEXT NOT NULL,
            home_goals_ft INTEGER,
            away_goals_ft INTEGER,
            home_goals_ht INTEGER,
            away_goals_ht INTEGER,
            elapsed_minutes INTEGER,
            venue_name TEXT,
            referee TEXT,
            downloaded_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS match_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER NOT NULL,
            event_minute INTEGER NOT NULL,
            event_extra INTEGER,
            event_type TEXT NOT NULL,
            event_detail TEXT,
            team_id INTEGER,
            team_name TEXT,
            player_id INTEGER,
            player_name TEXT,
            assist_id INTEGER,
            assist_name TEXT,
            comments TEXT,
            FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
        );
    """)

    # 경기: Man City 2-1 Arsenal
    c.execute("""
        INSERT INTO matches (fixture_id, league_id, league_name, season, match_date,
                             home_team_id, home_team_name, away_team_id, away_team_name,
                             home_goals_ft, away_goals_ft, home_goals_ht, away_goals_ht,
                             elapsed_minutes)
        VALUES (9001, 39, 'Premier League', 2024, '2024-09-22',
                50, 'Manchester City', 42, 'Arsenal', 2, 1, 1, 0, 95)
    """)

    events = [
        (9001, 23, None, "Goal", "Normal Goal", 50, "Manchester City", 1, "E. Haaland", None, None, None),
        (9001, 55, None, "Goal", "Normal Goal", 42, "Arsenal", 2, "B. Saka", None, None, None),
        (9001, 78, None, "Goal", "Normal Goal", 50, "Manchester City", 3, "K. De Bruyne", None, None, None),
    ]
    c.executemany("""
        INSERT INTO match_events (fixture_id, event_minute, event_extra, event_type, event_detail,
                                  team_id, team_name, player_id, player_name, assist_id, assist_name, comments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, events)

    # 경기 2: 레드카드 있는 경기 (Liverpool 2-1 Chelsea)
    c.execute("""
        INSERT INTO matches (fixture_id, league_id, league_name, season, match_date,
                             home_team_id, home_team_name, away_team_id, away_team_name,
                             home_goals_ft, away_goals_ft, home_goals_ht, away_goals_ht,
                             elapsed_minutes)
        VALUES (9002, 39, 'Premier League', 2024, '2024-09-29',
                33, 'Liverpool', 34, 'Chelsea', 2, 1, 0, 1, 93)
    """)

    events_2 = [
        (9002, 30, None, "Goal", "Normal Goal", 34, "Chelsea", 10, "C. Palmer", None, None, None),
        (9002, 38, None, "Card", "Red Card", 34, "Chelsea", 11, "M. Caicedo", None, None, None),
        (9002, 60, None, "Goal", "Normal Goal", 33, "Liverpool", 12, "M. Salah", None, None, None),
        (9002, 82, None, "Goal", "Normal Goal", 33, "Liverpool", 13, "D. Nunez", None, None, None),
    ]
    c.executemany("""
        INSERT INTO match_events (fixture_id, event_minute, event_extra, event_type, event_detail,
                                  team_id, team_name, player_id, player_name, assist_id, assist_name, comments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, events_2)

    # 경기 3: 0-0 무골
    c.execute("""
        INSERT INTO matches (fixture_id, league_id, league_name, season, match_date,
                             home_team_id, home_team_name, away_team_id, away_team_name,
                             home_goals_ft, away_goals_ft, home_goals_ht, away_goals_ht,
                             elapsed_minutes)
        VALUES (9003, 39, 'Premier League', 2024, '2024-10-06',
                47, 'Tottenham', 40, 'Aston Villa', 0, 0, 0, 0, 92)
    """)

    conn.commit()
    conn.close()


def make_engine_params() -> EngineParams:
    """테스트용 EngineParams 생성."""
    b = np.array([0.05, -0.02, 0.00, 0.03, -0.01, 0.08])
    basis_bounds = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 95.0])

    dt_bins = [15.0, 15.0, 17.0, 15.0, 15.0, 18.0]
    C_time = sum(math.exp(b[i]) * dt_bins[i] for i in range(6))

    mu_hat_H, mu_hat_A = 1.35, 1.10
    a_H = math.log(mu_hat_H) - math.log(C_time)
    a_A = math.log(mu_hat_A) - math.log(C_time)

    gamma = build_gamma_array(-0.25, 0.20)
    delta_H = build_delta_array([0.15, 0.08, -0.10, -0.18])
    delta_A = build_delta_array([-0.12, -0.06, 0.08, 0.15])

    Q = np.array([
        [-0.04,  0.02,  0.02,  0.00],
        [ 0.00, -0.02,  0.00,  0.02],
        [ 0.00,  0.00, -0.02,  0.02],
        [ 0.00,  0.00,  0.00,  0.00],
    ])
    Q_diag, Q_off = build_Q_diag_and_off(Q)
    P_grid = {dt: expm(Q * dt) for dt in range(101)}

    return EngineParams(
        a_H=a_H, a_A=a_A,
        b=b, gamma=gamma,
        delta_H=delta_H, delta_A=delta_A,
        P_grid=P_grid,
        Q_diag=Q_diag, Q_off=Q_off,
        T_exp=95.0,
        basis_bounds=basis_bounds,
        mu_H_prematch=mu_hat_H,
        mu_A_prematch=mu_hat_A,
        mc_simulations=5_000,       # 테스트용 적은 시뮬레이션
        cooldown_seconds=0.001,     # 테스트용 짧은 쿨다운
    )


# ═════════════════════════════════════════════════════════
# 테스트
# ═════════════════════════════════════════════════════════

def test_all():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        create_test_db(db_path)
        params = make_engine_params()

        # ──────────────────────────────────────────────
        # 테스트 1: 표준 경기 (9001, City 2-1 Arsenal)
        # ──────────────────────────────────────────────
        source = ReplaySource(db_path=db_path, speed=0.0)
        engine = LiveTradingEngine(event_source=source, params=params)
        snapshots = asyncio.run(engine.run("9001"))

        # 최종 스코어 확인
        last = snapshots[-1]
        assert last.S_H == 2 and last.S_A == 1, f"최종: {last.S_H}-{last.S_A}"
        assert last.engine_phase == "FINISHED"
        print(f"✅ T1: 표준 경기 — 최종 {last.S_H}-{last.S_A}, FINISHED")

        # ──────────────────────────────────────────────
        # 테스트 2: 틱 수 > 이벤트 수
        # ──────────────────────────────────────────────
        assert len(snapshots) >= 5  # 최소 골3 + 하프타임 + 종료
        print(f"✅ T2: 틱 수 = {len(snapshots)}")

        # ──────────────────────────────────────────────
        # 테스트 3: 하프타임 전환 존재
        # ──────────────────────────────────────────────
        phases = [s.engine_phase for s in snapshots]
        assert "HALFTIME" in phases
        assert "SECOND_HALF" in phases
        print("✅ T3: HALFTIME → SECOND_HALF 전환 확인")

        # ──────────────────────────────────────────────
        # 테스트 4: μ 단조 감소 (같은 상태 내에서)
        # ──────────────────────────────────────────────
        first_half_ticks = [
            s for s in snapshots
            if s.engine_phase == "FIRST_HALF" and s.event is None
        ]
        if len(first_half_ticks) >= 2:
            assert first_half_ticks[0].mu_H >= first_half_ticks[-1].mu_H
        print("✅ T4: μ 감소 추세 확인")

        # ──────────────────────────────────────────────
        # 테스트 5: 골 이벤트 시 ΔS 변화
        # ──────────────────────────────────────────────
        goal_ticks = [s for s in snapshots if s.event and "goal" in s.event.lower()]
        assert len(goal_ticks) == 3  # 3골
        # 첫 골: 0→1
        assert goal_ticks[0].delta_S == 1
        # 둘째 골 (어웨이): 1→0
        assert goal_ticks[1].delta_S == 0
        # 셋째 골: 0→1
        assert goal_ticks[2].delta_S == 1
        print("✅ T5: 골 이벤트 시 ΔS 변화 — 1, 0, 1")

        # ──────────────────────────────────────────────
        # 테스트 6: pricing 존재 (활성 틱에서)
        # ──────────────────────────────────────────────
        active_ticks = [
            s for s in snapshots
            if s.engine_phase in ("FIRST_HALF", "SECOND_HALF")
        ]
        for tick in active_ticks:
            assert tick.pricing is not None
            p = tick.pricing
            # 확률 합 ≈ 1
            total = p.home_win + p.draw + p.away_win
            assert abs(total - 1.0) < 0.05, f"확률 합 {total} ≠ 1"
        print(f"✅ T6: pricing 존재 — {len(active_ticks)}개 활성 틱")

        # ──────────────────────────────────────────────
        # 테스트 7: 마지막 틱 pricing — 2-1 확정
        # ──────────────────────────────────────────────
        # 경기 종료 직전 틱에서 μ ≈ 0이므로 현재 스코어가 거의 확정
        final_active = [
            s for s in snapshots
            if s.engine_phase in ("FIRST_HALF", "SECOND_HALF")
        ]
        if final_active:
            last_active = final_active[-1]
            if last_active.pricing:
                assert last_active.pricing.home_win > 0.5
                print(f"✅ T7: 마지막 활성 틱 — 홈승 {last_active.pricing.home_win:.3f}")

        # ──────────────────────────────────────────────
        # 테스트 8: summary() 출력
        # ──────────────────────────────────────────────
        s = engine.summary()
        assert "2-1" in s
        assert "골" in s or "이벤트" in s
        print(f"✅ T8: summary() 정상")

        # ──────────────────────────────────────────────
        # 테스트 9: 레드카드 경기 (9002, Liverpool 2-1 Chelsea)
        # ──────────────────────────────────────────────
        source2 = ReplaySource(db_path=db_path, speed=0.0)
        engine2 = LiveTradingEngine(event_source=source2, params=params)
        snapshots2 = asyncio.run(engine2.run("9002"))

        last2 = snapshots2[-1]
        assert last2.S_H == 2 and last2.S_A == 1
        assert last2.engine_phase == "FINISHED"

        # 레드카드 후 X 변화 확인
        red_ticks = [s for s in snapshots2 if s.event and "red_card" in s.event.lower()]
        assert len(red_ticks) == 1
        assert red_ticks[0].X == 2  # 어웨이 퇴장 → X=2
        print(f"✅ T9: 레드카드 경기 — X=2 (Chelsea 퇴장), 최종 {last2.S_H}-{last2.S_A}")

        # ──────────────────────────────────────────────
        # 테스트 10: X≠0일 때 MC 모드 사용
        # ──────────────────────────────────────────────
        mc_ticks = [
            s for s in snapshots2
            if s.pricing and s.pricing.mode == "monte_carlo"
        ]
        assert len(mc_ticks) > 0, "레드카드 후 MC 모드 사용해야 함"
        print(f"✅ T10: MC 모드 틱 = {len(mc_ticks)}개")

        # ──────────────────────────────────────────────
        # 테스트 11: 0-0 경기 (9003)
        # ──────────────────────────────────────────────
        source3 = ReplaySource(db_path=db_path, speed=0.0)
        engine3 = LiveTradingEngine(event_source=source3, params=params)
        snapshots3 = asyncio.run(engine3.run("9003"))

        last3 = snapshots3[-1]
        assert last3.S_H == 0 and last3.S_A == 0
        assert last3.engine_phase == "FINISHED"
        assert last3.X == 0  # 퇴장 없음
        print(f"✅ T11: 0-0 경기 — FINISHED, X=0")

        # ──────────────────────────────────────────────
        # 테스트 12: 0-0 경기에서 해석적 모드만 사용
        # ──────────────────────────────────────────────
        analytical_ticks = [
            s for s in snapshots3
            if s.engine_phase in ("FIRST_HALF", "SECOND_HALF")
            and s.pricing and s.pricing.mode == "analytical"
        ]
        active_ticks_3 = [
            s for s in snapshots3
            if s.engine_phase in ("FIRST_HALF", "SECOND_HALF")
        ]
        assert len(analytical_ticks) == len(active_ticks_3)
        print(f"✅ T12: 0-0 경기 — 활성 틱 전체 해석적 모드 ({len(analytical_ticks)}틱)")

        # ──────────────────────────────────────────────
        # 테스트 13: on_tick 콜백
        # ──────────────────────────────────────────────
        callback_log = []
        source4 = ReplaySource(db_path=db_path, speed=0.0)
        engine4 = LiveTradingEngine(
            event_source=source4,
            params=params,
            on_tick=lambda snap: callback_log.append(snap.tick),
        )
        asyncio.run(engine4.run("9003"))
        assert len(callback_log) > 0
        assert callback_log == list(range(len(callback_log)))  # 순차적
        print(f"✅ T13: on_tick 콜백 — {len(callback_log)}회 호출")

        # ──────────────────────────────────────────────
        # 테스트 14: 추가시간 관리 (stoppage)
        # ──────────────────────────────────────────────
        # engine.stoppage가 Phase C로 전환되었는지 확인
        # 9001: elapsed=95이므로 추가시간 존재
        assert engine.stoppage.first_half_stoppage_entered is True
        print("✅ T14: StoppageTimeManager 연동 확인")

        # ──────────────────────────────────────────────
        print()
        print("═" * 50)
        print("  ALL 14 TESTS PASSED ✅")
        print("═" * 50)

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_all()
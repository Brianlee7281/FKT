"""
mc_core.py 단위 테스트

실행:
  python -m pytest tests/test_mc_core.py -v
  python tests/test_mc_core.py
"""

import math
import time
import numpy as np
from scipy.linalg import expm

from src.phase3.mc_core import (
    mc_simulate_remaining,
    build_Q_diag_and_off,
    warmup,
    HAS_NUMBA,
)
from src.phase3.pricer import analytical_pricing, aggregate_mc_results


# ═════════════════════════════════════════════════════════
# 테스트 파라미터 (mu_calculator 테스트와 동일 구조)
# ═════════════════════════════════════════════════════════

def make_test_params():
    """현실적인 MC 파라미터 세트."""

    b = np.array([0.05, -0.02, 0.00, 0.03, -0.01, 0.08])
    basis_bounds = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 95.0])

    # C_time으로 a 역산 (Phase 2 방식)
    dt_bins = [15.0, 15.0, 17.0, 15.0, 15.0, 18.0]
    C_time = sum(math.exp(b[i]) * dt_bins[i] for i in range(6))
    a_H = math.log(1.35) - math.log(C_time)
    a_A = math.log(1.10) - math.log(C_time)

    gamma = np.array([0.0, -0.25, 0.20, -0.05])
    delta_H = np.array([0.15, 0.08, 0.0, -0.10, -0.18])
    delta_A = np.array([-0.12, -0.06, 0.0, 0.08, 0.15])

    Q = np.array([
        [-0.04,  0.02,  0.02,  0.00],
        [ 0.00, -0.02,  0.00,  0.02],
        [ 0.00,  0.00, -0.02,  0.02],
        [ 0.00,  0.00,  0.00,  0.00],
    ])
    Q_diag, Q_off = build_Q_diag_and_off(Q)

    return {
        "a_H": a_H, "a_A": a_A,
        "b": b, "gamma": gamma,
        "delta_H": delta_H, "delta_A": delta_A,
        "Q": Q, "Q_diag": Q_diag, "Q_off": Q_off,
        "basis_bounds": basis_bounds,
        "T_m": 95.0,
    }


# ═════════════════════════════════════════════════════════
# 테스트
# ═════════════════════════════════════════════════════════

def test_all():
    p = make_test_params()
    N = 50_000

    # ──────────────────────────────────────────────────
    # 테스트 1: build_Q_diag_and_off
    # ──────────────────────────────────────────────────
    Q_diag, Q_off = build_Q_diag_and_off(p["Q"])

    assert Q_diag.shape == (4,)
    assert Q_off.shape == (4, 4)
    assert Q_diag[0] == -0.04
    assert Q_diag[3] == 0.0   # 흡수 상태

    # Q_off 행 합 검증 (흡수 상태 제외)
    for i in range(3):
        assert abs(Q_off[i].sum() - 1.0) < 1e-10, f"Q_off[{i}] 행 합 ≠ 1"
    # 대각 성분 = 0
    for i in range(4):
        assert Q_off[i, i] == 0.0
    print("✅ T1: build_Q_diag_and_off — Q_diag, Q_off 정상")

    # ──────────────────────────────────────────────────
    # 테스트 2: 기본 실행 — 출력 형태 확인
    # ──────────────────────────────────────────────────
    results = mc_simulate_remaining(
        t_now=0.0, T_end=p["T_m"],
        S_H=0, S_A=0, state=0, score_diff=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        Q_diag=p["Q_diag"], Q_off=p["Q_off"],
        basis_bounds=p["basis_bounds"],
        N=N, seed=42,
    )

    assert results.shape == (N, 2)
    assert results.dtype == np.int32
    # 모든 스코어 >= 0
    assert np.all(results >= 0)
    print(f"✅ T2: 출력 shape=({N}, 2), dtype=int32, 모두 ≥ 0")

    # ──────────────────────────────────────────────────
    # 테스트 3: MC vs 해석적 교차 검증 (X=0, ΔS=0)
    # ──────────────────────────────────────────────────
    # X=0, ΔS=0이면 독립 푸아송이므로 해석적과 일치해야 함
    mc_result = aggregate_mc_results(results)
    an_result = analytical_pricing(
        mu_H=1.35, mu_A=1.10, S_H=0, S_A=0,
    )

    # 2% 이내 일치 (γ, δ의 영향으로 약간 차이 가능)
    # MC는 γ=0, δ=0 상태에서 시작하지만 퇴장 발생 가능성이 있어
    # 순수 독립 푸아송보다 약간 다를 수 있음
    diff_home = abs(mc_result.home_win - an_result.home_win)
    diff_over = abs(mc_result.over_25 - an_result.over_25)
    assert diff_home < 0.03, f"홈승 차이 {diff_home:.4f} > 0.03"
    assert diff_over < 0.03, f"Over2.5 차이 {diff_over:.4f} > 0.03"
    print(f"✅ T3: MC vs 해석적 — 홈승 차이={diff_home:.4f}, Over2.5 차이={diff_over:.4f}")

    # ──────────────────────────────────────────────────
    # 테스트 4: 평균 스코어 ≈ 기대 골
    # ──────────────────────────────────────────────────
    mean_H = results[:, 0].mean()
    mean_A = results[:, 1].mean()
    # 퇴장 효과로 인해 정확히 1.35, 1.10은 아니지만 근사
    assert 1.0 < mean_H < 1.7, f"mean_H={mean_H:.3f} 범위 초과"
    assert 0.8 < mean_A < 1.4, f"mean_A={mean_A:.3f} 범위 초과"
    print(f"✅ T4: 평균 스코어 — H={mean_H:.3f}, A={mean_A:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 5: 시드 재현성
    # ──────────────────────────────────────────────────
    r1 = mc_simulate_remaining(
        0.0, p["T_m"], 0, 0, 0, 0,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=1000, seed=123,
    )
    r2 = mc_simulate_remaining(
        0.0, p["T_m"], 0, 0, 0, 0,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=1000, seed=123,
    )
    assert np.array_equal(r1, r2)
    print("✅ T5: 시드 재현성 확인")

    # ──────────────────────────────────────────────────
    # 테스트 6: 다른 시드 → 다른 결과
    # ──────────────────────────────────────────────────
    r3 = mc_simulate_remaining(
        0.0, p["T_m"], 0, 0, 0, 0,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=1000, seed=456,
    )
    assert not np.array_equal(r1, r3)
    print("✅ T6: 다른 시드 → 다른 결과")

    # ──────────────────────────────────────────────────
    # 테스트 7: 현재 스코어 반영
    # ──────────────────────────────────────────────────
    r_10 = mc_simulate_remaining(
        60.0, p["T_m"], 1, 0, 0, 1,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=N, seed=42,
    )
    # 최소 1-0 이상이어야 함
    assert np.all(r_10[:, 0] >= 1)
    assert np.all(r_10[:, 1] >= 0)
    print("✅ T7: 현재 스코어 반영 — S_H ≥ 1 보장")

    # ──────────────────────────────────────────────────
    # 테스트 8: 홈 퇴장(X=1) → 홈 평균 골 감소
    # ──────────────────────────────────────────────────
    r_X0 = mc_simulate_remaining(
        30.0, p["T_m"], 0, 0, 0, 0,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=N, seed=42,
    )
    r_X1 = mc_simulate_remaining(
        30.0, p["T_m"], 0, 0, 1, 0,   # 홈 퇴장
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=N, seed=42,
    )
    mean_X0_H = r_X0[:, 0].mean()
    mean_X1_H = r_X1[:, 0].mean()
    assert mean_X1_H < mean_X0_H, "홈 퇴장 시 홈 평균 골 감소해야 함"
    print(f"✅ T8: 홈 퇴장 효과 — 평균 H: {mean_X0_H:.3f} → {mean_X1_H:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 9: t ≈ T → 추가 골 거의 없음
    # ──────────────────────────────────────────────────
    r_late = mc_simulate_remaining(
        94.0, 95.0, 2, 1, 0, 1,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=N, seed=42,
    )
    # 대부분 2-1 그대로
    pct_same = np.mean((r_late[:, 0] == 2) & (r_late[:, 1] == 1))
    assert pct_same > 0.95, f"경기 막판 원 스코어 비율: {pct_same:.3f}"
    print(f"✅ T9: 경기 막판 (94→95분) — 2-1 유지 비율={pct_same:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 10: t >= T → 현재 스코어 그대로
    # ──────────────────────────────────────────────────
    r_done = mc_simulate_remaining(
        95.0, 95.0, 3, 2, 0, 1,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=100, seed=42,
    )
    assert np.all(r_done[:, 0] == 3)
    assert np.all(r_done[:, 1] == 2)
    print("✅ T10: t >= T → 스코어 불변")

    # ──────────────────────────────────────────────────
    # 테스트 11: 성능 벤치마크
    # ──────────────────────────────────────────────────
    t0 = time.time()
    for _ in range(10):
        mc_simulate_remaining(
            30.0, p["T_m"], 1, 0, 1, 1,
            p["a_H"], p["a_A"], p["b"], p["gamma"],
            p["delta_H"], p["delta_A"],
            p["Q_diag"], p["Q_off"], p["basis_bounds"],
            N=N, seed=42,
        )
    elapsed = (time.time() - t0) / 10
    mode = "Numba" if HAS_NUMBA else "Python"
    print(f"✅ T11: 벤치마크 ({mode}) — N={N}, {elapsed*1000:.1f}ms/call")

    # ──────────────────────────────────────────────────
    # 테스트 12: warmup 실행
    # ──────────────────────────────────────────────────
    warmup_time = warmup()
    assert warmup_time >= 0
    print(f"✅ T12: warmup 완료 — {warmup_time:.2f}s")

    # ──────────────────────────────────────────────────
    # 테스트 13: 흡수 상태(X=3)에서 추가 퇴장 없음
    # ──────────────────────────────────────────────────
    # X=3(양쪽 퇴장)이면 Q_diag[3]=0 → 퇴장 이벤트 없음
    # 골만 발생해야 함
    r_abs = mc_simulate_remaining(
        0.0, p["T_m"], 0, 0, 3, 0,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=1000, seed=42,
    )
    assert r_abs.shape == (1000, 2)
    # 정상적으로 완료 (크래시 없음)
    print("✅ T13: 흡수 상태(X=3) — 정상 실행 (퇴장 없이 골만)")

    # ──────────────────────────────────────────────────
    # 테스트 14: 스코어차 효과 — 홈 2골 리드 시 방어 모드
    # ──────────────────────────────────────────────────
    r_even = mc_simulate_remaining(
        30.0, p["T_m"], 0, 0, 0, 0,
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=N, seed=42,
    )
    r_lead = mc_simulate_remaining(
        30.0, p["T_m"], 2, 0, 0, 2,   # 2-0 리드
        p["a_H"], p["a_A"], p["b"], p["gamma"],
        p["delta_H"], p["delta_A"],
        p["Q_diag"], p["Q_off"], p["basis_bounds"],
        N=N, seed=42,
    )
    # 잔여 골만 비교 (현재 스코어 빼기)
    remaining_even_H = r_even[:, 0].mean()
    remaining_lead_H = r_lead[:, 0].mean() - 2  # 현재 2골 빼기
    # δ_H(ΔS≥+2) = -0.18 → 리드 시 잔여 홈 골 감소
    assert remaining_lead_H < remaining_even_H, "리드 시 잔여 홈 골 감소해야 함"
    print(f"✅ T14: 스코어차 효과 — 잔여 홈 골: even={remaining_even_H:.3f}, lead={remaining_lead_H:.3f}")

    # ──────────────────────────────────────────────────
    print()
    print("═" * 50)
    print(f"  ALL 14 TESTS PASSED ✅ (mode: {mode})")
    print("═" * 50)


if __name__ == "__main__":
    test_all()
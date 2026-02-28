"""
mu_calculator.py 단위 테스트

실행:
  python -m pytest tests/test_mu_calculator.py -v
  python tests/test_mu_calculator.py
"""

import math
import math
import numpy as np
from scipy.linalg import expm

from src.phase3.mu_calculator import (
    build_gamma_array,
    build_delta_array,
    delta_index,
    get_basis_idx,
    build_basis_bounds,
    compute_remaining_mu,
    verify_kickoff_mu,
)


# ═════════════════════════════════════════════════════════
# 테스트 픽스처: 현실적인 파라미터
# ═════════════════════════════════════════════════════════

def make_test_params():
    """Phase 1/2에서 나올 수 있는 현실적인 파라미터 세트를 생성."""

    # 시간 기저함수 (Phase 1 NLL 결과 예시)
    b = np.array([0.05, -0.02, 0.00, 0.03, -0.01, 0.08])

    # 기저함수 경계 (전반 추가시간 2분, 총 ~95분)
    first_half_end = 47.0
    T_m = 95.0
    basis_bounds = build_basis_bounds(first_half_end, T_m)

    # C_time 계산 (Phase 2 역산 공식과 동일)
    dt_bins = [15.0, 15.0, 17.0, 15.0, 15.0, 18.0]
    C_time = sum(math.exp(b[i]) * dt_bins[i] for i in range(6))

    # Phase 2 역산: a = ln(μ̂) - ln(C_time)
    mu_hat_H = 1.35   # XGBoost 예측 홈 기대골
    mu_hat_A = 1.10   # XGBoost 예측 어웨이 기대골
    a_H = math.log(mu_hat_H) - math.log(C_time)
    a_A = math.log(mu_hat_A) - math.log(C_time)

    # 퇴장 패널티
    gamma_1 = -0.25    # 홈 퇴장 → 홈 약화
    gamma_2 = 0.20     # 원정 퇴장 → 홈 강화
    gamma = build_gamma_array(gamma_1, gamma_2)

    # 스코어차 효과 (Phase 1 NLL: 4개 값)
    delta_H_4 = [0.15, 0.08, -0.10, -0.18]   # ΔS: ≤-2, -1, +1, ≥+2
    delta_A_4 = [-0.12, -0.06, 0.08, 0.15]
    delta_H = build_delta_array(delta_H_4)
    delta_A = build_delta_array(delta_A_4)

    # Q matrix (현실적인 퇴장 비율)
    Q = np.array([
        [-0.04,  0.02,  0.02,  0.00],
        [ 0.00, -0.02,  0.00,  0.02],
        [ 0.00,  0.00, -0.02,  0.02],
        [ 0.00,  0.00,  0.00,  0.00],
    ])

    # P_grid 사전 계산
    P_grid = {}
    for dt in range(101):
        P_grid[dt] = expm(Q * dt)

    return {
        "a_H": a_H, "a_A": a_A,
        "b": b,
        "gamma": gamma, "gamma_1": gamma_1, "gamma_2": gamma_2,
        "delta_H": delta_H, "delta_A": delta_A,
        "delta_H_4": delta_H_4, "delta_A_4": delta_A_4,
        "Q": Q, "P_grid": P_grid,
        "basis_bounds": basis_bounds,
        "first_half_end": first_half_end,
        "T_m": T_m,
        "C_time": C_time,
        "mu_hat_H": mu_hat_H, "mu_hat_A": mu_hat_A,
    }


# ═════════════════════════════════════════════════════════
# 테스트
# ═════════════════════════════════════════════════════════

def test_all():
    p = make_test_params()

    # ──────────────────────────────────────────────────
    # 테스트 1: build_gamma_array
    # ──────────────────────────────────────────────────
    gamma = build_gamma_array(-0.25, 0.20)
    assert gamma[0] == 0.0         # 11v11
    assert gamma[1] == -0.25       # 홈 퇴장
    assert gamma[2] == 0.20        # 원정 퇴장
    assert abs(gamma[3] - (-0.05)) < 1e-10   # 양쪽 (합산)
    assert gamma.shape == (4,)
    print("✅ T1: build_gamma_array 정상")

    # ──────────────────────────────────────────────────
    # 테스트 2: build_delta_array
    # ──────────────────────────────────────────────────
    d = build_delta_array([0.15, 0.08, -0.10, -0.18])
    assert d.shape == (5,)
    assert d[0] == 0.15    # ΔS ≤ -2
    assert d[1] == 0.08    # ΔS = -1
    assert d[2] == 0.0     # ΔS = 0 (삽입됨)
    assert d[3] == -0.10   # ΔS = +1
    assert d[4] == -0.18   # ΔS ≥ +2
    print("✅ T2: build_delta_array — ΔS=0에 0.0 삽입 확인")

    # ──────────────────────────────────────────────────
    # 테스트 3: delta_index
    # ──────────────────────────────────────────────────
    assert delta_index(-3) == 0    # ≤ -2
    assert delta_index(-2) == 0
    assert delta_index(-1) == 1
    assert delta_index(0) == 2
    assert delta_index(1) == 3
    assert delta_index(2) == 4
    assert delta_index(5) == 4     # ≥ +2 (클램프)
    print("✅ T3: delta_index 매핑 정확")

    # ──────────────────────────────────────────────────
    # 테스트 4: basis_bounds 구성
    # ──────────────────────────────────────────────────
    bb = build_basis_bounds(47.0, 95.0)
    assert bb.shape == (7,)
    assert bb[0] == 0.0
    assert bb[1] == 15.0
    assert bb[2] == 30.0
    assert bb[3] == 47.0   # 전반 종료
    assert bb[4] == 62.0   # 47 + 15
    assert bb[5] == 77.0   # 47 + 30
    assert bb[6] == 95.0   # T_m
    print("✅ T4: basis_bounds [0, 15, 30, 47, 62, 77, 95]")

    # ──────────────────────────────────────────────────
    # 테스트 5: get_basis_idx
    # ──────────────────────────────────────────────────
    bb = p["basis_bounds"]
    assert get_basis_idx(0.0, bb) == 0
    assert get_basis_idx(10.0, bb) == 0
    assert get_basis_idx(15.0, bb) == 1
    assert get_basis_idx(29.9, bb) == 1
    assert get_basis_idx(30.0, bb) == 2
    assert get_basis_idx(47.0, bb) == 3   # 후반 시작
    assert get_basis_idx(61.9, bb) == 3
    assert get_basis_idx(62.0, bb) == 4
    assert get_basis_idx(77.0, bb) == 5
    assert get_basis_idx(94.0, bb) == 5
    assert get_basis_idx(99.0, bb) == 5   # T_m 이후도 마지막 빈
    print("✅ T5: get_basis_idx 경계 매핑 정확")

    # ──────────────────────────────────────────────────
    # 테스트 6: 킥오프 시 μ (t=0, X=0, ΔS=0)
    # ──────────────────────────────────────────────────
    mu_H, mu_A = compute_remaining_mu(
        t=0.0, T=p["T_m"], X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    # 양수여야 함
    assert mu_H > 0 and mu_A > 0
    # 현실적 범위 (0.5~3.0 기대 골)
    assert 0.5 < mu_H < 3.0, f"mu_H={mu_H} 범위 초과"
    assert 0.5 < mu_A < 3.0, f"mu_A={mu_A} 범위 초과"
    # a_H > a_A이므로 홈팀이 더 많이 넣어야 함
    assert mu_H > mu_A, f"mu_H={mu_H} <= mu_A={mu_A}"
    print(f"✅ T6: 킥오프 μ = ({mu_H:.4f}, {mu_A:.4f}) — 현실적 범위 내, μ_H > μ_A")

    # ──────────────────────────────────────────────────
    # 테스트 7: μ의 단조 감소 (시간이 지나면 μ 감소)
    # ──────────────────────────────────────────────────
    mu_30 = compute_remaining_mu(
        t=30.0, T=p["T_m"], X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    mu_60 = compute_remaining_mu(
        t=60.0, T=p["T_m"], X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    assert mu_H > mu_30[0] > mu_60[0] > 0
    assert mu_A > mu_30[1] > mu_60[1] > 0
    print(f"✅ T7: 단조 감소 — t=0:{mu_H:.3f} > t=30:{mu_30[0]:.3f} > t=60:{mu_60[0]:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 8: t >= T이면 (0, 0) 반환
    # ──────────────────────────────────────────────────
    mu_end = compute_remaining_mu(
        t=95.0, T=95.0, X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    assert mu_end == (0.0, 0.0)
    print("✅ T8: t >= T → (0, 0)")

    # ──────────────────────────────────────────────────
    # 테스트 9: 레드카드 효과 (홈팀 퇴장 → μ_H 감소, μ_A 변화)
    # ──────────────────────────────────────────────────
    mu_X0 = compute_remaining_mu(
        t=30.0, T=p["T_m"], X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    mu_X1 = compute_remaining_mu(
        t=30.0, T=p["T_m"], X=1, delta_S=0,  # 홈 퇴장
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    # γ₁ < 0이므로 홈 퇴장 → 홈 강도 감소
    assert mu_X1[0] < mu_X0[0], "홈 퇴장 시 mu_H 감소해야 함"
    print(f"✅ T9: 홈 퇴장 효과 — μ_H: {mu_X0[0]:.4f} → {mu_X1[0]:.4f} (감소)")

    # ──────────────────────────────────────────────────
    # 테스트 10: 스코어차 효과 (홈 리드 → δ_H < 0 → μ_H 감소)
    # ──────────────────────────────────────────────────
    mu_even = compute_remaining_mu(
        t=60.0, T=p["T_m"], X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    mu_lead = compute_remaining_mu(
        t=60.0, T=p["T_m"], X=0, delta_S=1,  # 홈 1-0 리드
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    # δ_H(+1) = -0.10 < 0 → 홈 리드 시 수비 전환 → μ_H 감소
    assert mu_lead[0] < mu_even[0], "홈 리드 시 mu_H 감소해야 함 (수비 전환)"
    # δ_A(+1) = +0.08 > 0 → 뒤진 팀 공격 강화 → μ_A 증가
    assert mu_lead[1] > mu_even[1], "홈 리드 시 mu_A 증가해야 함 (뒤진 팀 반격)"
    print(f"✅ T10: 스코어차 효과 — 홈 리드 시 μ_H↓({mu_even[0]:.4f}→{mu_lead[0]:.4f}), μ_A↑({mu_even[1]:.4f}→{mu_lead[1]:.4f})")

    # ──────────────────────────────────────────────────
    # 테스트 11: 킥오프 교차 검증 — Phase 2 μ̂와 근사 일치
    # ──────────────────────────────────────────────────
    # Phase 2 역산 공식은 X=0 고정 가정이므로, 마르코프 전이를 고려하는
    # compute_remaining_mu는 약간 다를 수 있음. 5% 이내 일치 검증.
    passed, actual_H, actual_A = verify_kickoff_mu(
        mu_H_expected=p["mu_hat_H"],
        mu_A_expected=p["mu_hat_A"],
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
        T_exp=p["T_m"],
        tolerance=p["mu_hat_H"] * 0.05,  # 5% 허용
    )
    assert passed, f"킥오프 교차 검증 실패: actual=({actual_H:.4f}, {actual_A:.4f})"
    print(f"✅ T11: verify_kickoff_mu — μ̂=({p['mu_hat_H']:.2f}, {p['mu_hat_A']:.2f}) vs actual=({actual_H:.4f}, {actual_A:.4f})")

    # ──────────────────────────────────────────────────
    # 테스트 12: 경기 막판 (t ≈ T) — μ가 0에 근접
    # ──────────────────────────────────────────────────
    mu_late = compute_remaining_mu(
        t=94.0, T=95.0, X=0, delta_S=0,
        a_H=p["a_H"], a_A=p["a_A"],
        b=p["b"], gamma=p["gamma"],
        delta_H=p["delta_H"], delta_A=p["delta_A"],
        P_grid=p["P_grid"], basis_bounds=p["basis_bounds"],
    )
    assert mu_late[0] < 0.1, f"경기 막판 mu_H가 너무 큼: {mu_late[0]}"
    assert mu_late[1] < 0.1, f"경기 막판 mu_A가 너무 큼: {mu_late[1]}"
    print(f"✅ T12: 경기 막판 μ ≈ 0 — ({mu_late[0]:.4f}, {mu_late[1]:.4f})")

    # ──────────────────────────────────────────────────
    # 테스트 13: P_grid 단위 행렬 확인 (dt=0일 때)
    # ──────────────────────────────────────────────────
    P0 = p["P_grid"][0]
    assert np.allclose(P0, np.eye(4), atol=1e-10)
    print("✅ T13: P_grid[0] = 단위 행렬")

    # ──────────────────────────────────────────────────
    # 테스트 14: P_grid 행 합 = 1 (확률의 성질)
    # ──────────────────────────────────────────────────
    for dt in [0, 10, 30, 50, 90]:
        row_sums = p["P_grid"][dt].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8), f"P_grid[{dt}] 행 합 ≠ 1"
    print("✅ T14: P_grid 행 합 = 1 (확률 보존)")

    # ──────────────────────────────────────────────────
    print()
    print("═" * 50)
    print("  ALL 14 TESTS PASSED ✅")
    print("═" * 50)


if __name__ == "__main__":
    test_all()
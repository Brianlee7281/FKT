"""
pricer.py 단위 테스트

실행:
  python -m pytest tests/test_pricer.py -v
  python tests/test_pricer.py
"""

import math
import numpy as np

from src.phase3.pricer import (
    poisson_pmf,
    analytical_pricing,
    aggregate_mc_results,
    price,
    PricingResult,
)


def test_all():

    # ──────────────────────────────────────────────────
    # 테스트 1: poisson_pmf 기본 검증
    # ──────────────────────────────────────────────────
    # P(X=0 | λ=1) = e^(-1) ≈ 0.3679
    assert abs(poisson_pmf(0, 1.0) - math.exp(-1)) < 1e-10
    # P(X=1 | λ=1) = e^(-1) ≈ 0.3679
    assert abs(poisson_pmf(1, 1.0) - math.exp(-1)) < 1e-10
    # Σ P(X=k) ≈ 1
    total = sum(poisson_pmf(k, 2.5) for k in range(20))
    assert abs(total - 1.0) < 1e-10
    # 엣지 케이스
    assert poisson_pmf(0, 0.0) == 1.0
    assert poisson_pmf(1, 0.0) == 0.0
    assert poisson_pmf(-1, 1.0) == 0.0
    print("✅ T1: poisson_pmf 정상")

    # ──────────────────────────────────────────────────
    # 테스트 2: 해석적 프라이싱 — 킥오프 (0-0, μ 현실적)
    # ──────────────────────────────────────────────────
    r = analytical_pricing(mu_H=1.3, mu_A=1.1, S_H=0, S_A=0)

    # 확률 합 = 1
    assert abs(r.home_win + r.draw + r.away_win - 1.0) < 1e-6
    assert abs(r.over_25 + r.under_25 - 1.0) < 1e-6
    print(f"  Match Odds: H={r.home_win:.3f} D={r.draw:.3f} A={r.away_win:.3f}")
    print(f"  Over 2.5: {r.over_25:.3f}")

    # 홈 기대골이 더 높으니 홈 승 > 어웨이 승
    assert r.home_win > r.away_win
    # 총 기대골 2.4 → Over 2.5 확률은 30~50% 범위
    assert 0.25 < r.over_25 < 0.60
    # 모드 확인
    assert r.mode == "analytical"
    assert r.sigma_mc == 0.0
    print("✅ T2: 킥오프 해석적 프라이싱 — 확률 합=1, 범위 정상")

    # ──────────────────────────────────────────────────
    # 테스트 3: 현재 스코어 반영 — 이미 1-0일 때
    # ──────────────────────────────────────────────────
    r0 = analytical_pricing(mu_H=0.5, mu_A=0.5, S_H=0, S_A=0)
    r1 = analytical_pricing(mu_H=0.5, mu_A=0.5, S_H=1, S_A=0)

    # 이미 1-0이면 홈 승 확률 ↑, Over 2.5 확률 ↑
    assert r1.home_win > r0.home_win
    assert r1.over_25 > r0.over_25
    print(f"✅ T3: 1-0 반영 — 홈승 {r0.home_win:.3f}→{r1.home_win:.3f}, Over2.5 {r0.over_25:.3f}→{r1.over_25:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 4: μ=0 — 경기 끝나면 현재 스코어가 최종
    # ──────────────────────────────────────────────────
    r_end = analytical_pricing(mu_H=0.0, mu_A=0.0, S_H=2, S_A=1)
    assert r_end.home_win > 0.99    # 2-1 확정
    assert r_end.draw < 0.01
    assert r_end.over_25 > 0.99     # 총 3골 → Over 2.5 확정
    print("✅ T4: μ=0 (경기 종료) — 2-1 확정, Over 2.5 확정")

    # ──────────────────────────────────────────────────
    # 테스트 5: μ=0, 0-0 — Under 2.5 확정
    # ──────────────────────────────────────────────────
    r_00 = analytical_pricing(mu_H=0.0, mu_A=0.0, S_H=0, S_A=0)
    assert r_00.draw > 0.99
    assert r_00.under_25 > 0.99
    assert r_00.over_25 < 0.01
    print("✅ T5: μ=0, 0-0 → 무승부, Under 2.5 확정")

    # ──────────────────────────────────────────────────
    # 테스트 6: Over/Under 관계 — over_15 >= over_25 >= over_35
    # ──────────────────────────────────────────────────
    r6 = analytical_pricing(mu_H=1.5, mu_A=1.2, S_H=0, S_A=0)
    assert r6.over_15 >= r6.over_25 >= r6.over_35
    print(f"✅ T6: Over 관계 — 1.5:{r6.over_15:.3f} ≥ 2.5:{r6.over_25:.3f} ≥ 3.5:{r6.over_35:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 7: MC 결과 집계 — 알려진 분포에서
    # ──────────────────────────────────────────────────
    np.random.seed(42)
    N = 100_000
    # 독립 푸아송으로 시뮬레이션 (해석적과 비교용)
    mu_H_test, mu_A_test = 1.3, 1.1
    h_sim = np.random.poisson(mu_H_test, N)
    a_sim = np.random.poisson(mu_A_test, N)
    final_scores = np.column_stack([h_sim, a_sim]).astype(np.int32)

    mc_result = aggregate_mc_results(final_scores)
    an_result = analytical_pricing(mu_H_test, mu_A_test, 0, 0)

    # MC vs 해석적 — 1% 이내 일치
    assert abs(mc_result.home_win - an_result.home_win) < 0.01
    assert abs(mc_result.draw - an_result.draw) < 0.01
    assert abs(mc_result.over_25 - an_result.over_25) < 0.01
    assert mc_result.mode == "monte_carlo"
    assert mc_result.sigma_mc > 0
    print(f"✅ T7: MC vs 해석적 일치 — Over2.5 차이 {abs(mc_result.over_25 - an_result.over_25):.4f}")

    # ──────────────────────────────────────────────────
    # 테스트 8: MC 결과 — 현재 스코어 포함된 경우
    # ──────────────────────────────────────────────────
    # S_H=1, S_A=0 상태에서 잔여 골을 시뮬레이션
    remaining_h = np.random.poisson(0.5, N)
    remaining_a = np.random.poisson(0.5, N)
    final_with_current = np.column_stack([
        1 + remaining_h,   # 현재 1 + 잔여
        0 + remaining_a,
    ]).astype(np.int32)

    mc_r8 = aggregate_mc_results(final_with_current)
    # 최소 1-0이므로 홈 승 확률이 높아야 함
    assert mc_r8.home_win > 0.5
    print(f"✅ T8: MC 현재 스코어 반영 — 1-0 출발, 홈승={mc_r8.home_win:.3f}")

    # ──────────────────────────────────────────────────
    # 테스트 9: sigma_mc 계산 검증
    # ──────────────────────────────────────────────────
    p = mc_result.over_25
    expected_sigma = math.sqrt(p * (1 - p) / N)
    assert abs(mc_result.sigma_mc - expected_sigma) < 1e-10
    # N=100,000이면 σ ≈ 0.001~0.002
    assert mc_result.sigma_mc < 0.005
    print(f"✅ T9: σ_MC = {mc_result.sigma_mc:.6f} (N={N})")

    # ──────────────────────────────────────────────────
    # 테스트 10: price() 디스패치 — 해석적
    # ──────────────────────────────────────────────────
    r10 = price(mu_H=1.0, mu_A=1.0, S_H=0, S_A=0, X=0, delta_S=0)
    assert r10.mode == "analytical"
    print("✅ T10: price() X=0, ΔS=0 → 해석적")

    # ──────────────────────────────────────────────────
    # 테스트 11: price() 디스패치 — MC
    # ──────────────────────────────────────────────────
    r11 = price(
        mu_H=1.0, mu_A=1.0, S_H=1, S_A=0,
        X=1, delta_S=1,
        mc_final_scores=final_scores,
    )
    assert r11.mode == "monte_carlo"
    print("✅ T11: price() X=1 → MC")

    # ──────────────────────────────────────────────────
    # 테스트 12: price() MC 모드에 결과 없으면 ValueError
    # ──────────────────────────────────────────────────
    try:
        price(mu_H=1.0, mu_A=1.0, S_H=0, S_A=0, X=1, delta_S=0)
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        print("✅ T12: MC 모드 + None → ValueError")

    # ──────────────────────────────────────────────────
    # 테스트 13: PricingResult 클램핑 (수치 오차 방지)
    # ──────────────────────────────────────────────────
    r13 = PricingResult(home_win=1.001, over_25=-0.001)
    assert r13.home_win == 1.0
    assert r13.over_25 == 0.0
    print("✅ T13: PricingResult 클램핑 — 1.001→1.0, -0.001→0.0")

    # ──────────────────────────────────────────────────
    # 테스트 14: 극단적 μ — 한쪽이 매우 강할 때
    # ──────────────────────────────────────────────────
    r14 = analytical_pricing(mu_H=3.0, mu_A=0.3, S_H=0, S_A=0)
    assert r14.home_win > 0.80
    assert r14.over_25 > 0.60
    assert abs(r14.home_win + r14.draw + r14.away_win - 1.0) < 0.001  # 꼬리 절단 허용
    print(f"✅ T14: 극단 μ(3.0 vs 0.3) — 홈승={r14.home_win:.3f}, Over2.5={r14.over_25:.3f}")

    # ──────────────────────────────────────────────────
    print()
    print("═" * 50)
    print("  ALL 14 TESTS PASSED ✅")
    print("═" * 50)


if __name__ == "__main__":
    test_all()
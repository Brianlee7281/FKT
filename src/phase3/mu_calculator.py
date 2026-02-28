"""
Phase 3 Step 3.2: 잔여 기대 득점 계산 (mu_calculator.py)

현재 시간 t부터 경기 종료 T까지 남은 시간 동안의
홈팀/어웨이팀 기대 득점 μ_H(t,T), μ_A(t,T)를 계산한다.

핵심 공식:
  μ_H(t, T) = Σ_ℓ Σ_j P̄_{X(t),j}^(ℓ) · exp(a_H + b_{iℓ} + γ_j + δ_H(ΔS)) · Δτ_ℓ

여기서:
  - P̄_{X,j}^(ℓ): 소구간 ℓ 동안 상태 j에 있을 평균 확률 (P_grid 조회)
  - b_{iℓ}: 소구간 ℓ의 기저함수 계수
  - γ_j: 마르코프 상태 j의 패널티 (0=없음, 1=홈퇴장, 2=원정퇴장, 3=양쪽)
  - δ_H(ΔS): 현재 스코어차에 의한 전술 효과

이 함수는 매초 호출되므로 성능이 중요하다.
P_grid 사전 계산 덕에 조회만 하므로 ~0.01ms.

사용법:
  from src.phase3.mu_calculator import compute_remaining_mu, build_gamma_array, build_delta_array

  gamma = build_gamma_array(gamma_1, gamma_2)
  delta_H_5 = build_delta_array(delta_H_4)

  mu_H, mu_A = compute_remaining_mu(
      t=30.0, T=95.0, X=0, delta_S=0,
      a_H=-0.3, a_A=-0.5,
      b=b_array, gamma=gamma,
      delta_H=delta_H_5, delta_A=delta_A_5,
      P_grid=P_grid, basis_bounds=basis_bounds,
  )
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 파라미터 변환 헬퍼
# ═════════════════════════════════════════════════════════

def build_gamma_array(gamma_1: float, gamma_2: float) -> np.ndarray:
    """
    Phase 1의 gamma_1, gamma_2를 마르코프 상태별 배열로 변환.

    Args:
        gamma_1: 홈팀 퇴장 패널티 (음수: 홈 약화)
        gamma_2: 어웨이팀 퇴장 패널티 (양수: 홈 강화)

    Returns:
        shape (4,) — [γ₀, γ₁, γ₂, γ₃]
        γ₀ = 0       (11v11)
        γ₁ = gamma_1  (홈 퇴장 → 홈 약화)
        γ₂ = gamma_2  (어웨이 퇴장 → 홈 강화)
        γ₃ = γ₁ + γ₂  (양쪽 퇴장)
    """
    return np.array([0.0, gamma_1, gamma_2, gamma_1 + gamma_2])


def build_delta_array(delta_4: List[float]) -> np.ndarray:
    """
    Phase 1의 delta[4]를 5-element 배열로 변환 (ΔS=0에 0 삽입).

    Phase 1 출력 (4개): [δ(ΔS≤-2), δ(ΔS=-1), δ(ΔS=+1), δ(ΔS≥+2)]
    변환 후 (5개):      [δ(ΔS≤-2), δ(ΔS=-1), 0.0, δ(ΔS=+1), δ(ΔS≥+2)]
                         index:     0          1     2         3          4

    인덱스 매핑: delta_index = clamp(ΔS + 2, 0, 4)
      ΔS ≤ -2 → 0
      ΔS = -1 → 1
      ΔS =  0 → 2  (δ=0)
      ΔS = +1 → 3
      ΔS ≥ +2 → 4

    Args:
        delta_4: Phase 1 NLL 파라미터 [4]

    Returns:
        shape (5,) — 0이 중간에 삽입된 배열
    """
    return np.array([
        delta_4[0],   # ΔS ≤ -2
        delta_4[1],   # ΔS = -1
        0.0,          # ΔS = 0
        delta_4[2],   # ΔS = +1
        delta_4[3],   # ΔS ≥ +2
    ])


def delta_index(delta_S: int) -> int:
    """ΔS를 delta 배열 인덱스(0~4)로 변환."""
    return max(0, min(4, delta_S + 2))


# ═════════════════════════════════════════════════════════
# 기저함수 유틸
# ═════════════════════════════════════════════════════════

def get_basis_idx(t: float, basis_bounds: np.ndarray) -> int:
    """
    실효 시간 t에 해당하는 기저함수 인덱스(0~5)를 반환.

    basis_bounds는 7개 경계점: [0, 15, 30, fh_end, fh_end+15, fh_end+30, T_m]
    """
    n_bins = len(basis_bounds) - 1  # 보통 6
    for i in range(n_bins):
        if basis_bounds[i] <= t < basis_bounds[i + 1]:
            return i
    # t >= T_m이면 마지막 빈
    return n_bins - 1


def build_basis_bounds(
    first_half_end: float,
    T_m: float,
    bin_size: float = 15.0,
) -> np.ndarray:
    """
    Phase 2 초기화에서 기저함수 경계 배열을 생성하는 헬퍼.

    Args:
        first_half_end: 전반 종료 실효 시간 (예: 47.0)
        T_m: 경기 종료 실효 시간 (예: 95.0)
        bin_size: 기저함수 구간 크기 (기본 15분)

    Returns:
        shape (7,) — [0, 15, 30, fh_end, fh_end+15, fh_end+30, T_m]
    """
    second_half_start = first_half_end
    return np.array([
        0.0,
        bin_size,
        2 * bin_size,
        first_half_end,
        second_half_start + bin_size,
        second_half_start + 2 * bin_size,
        T_m,
    ])


# ═════════════════════════════════════════════════════════
# 핵심 함수: 잔여 기대 득점 계산
# ═════════════════════════════════════════════════════════

def compute_remaining_mu(
    t: float,
    T: float,
    X: int,
    delta_S: int,
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    P_grid: Dict[int, np.ndarray],
    basis_bounds: np.ndarray,
) -> Tuple[float, float]:
    """
    현재 시간 t부터 경기 종료 T까지의 잔여 기대 득점을 계산한다.

    [t, T] 구간을 기저함수 경계에서 잘라 L개의 소구간으로 나누고,
    각 소구간에서 마르코프 상태 전이 확률(P_grid)을 고려하여
    가중 적분을 수행한다.

    Args:
        t:            현재 실효 플레이 시간 (분). 0 ~ T.
        T:            경기 종료 예정 시간 (분). 보통 ~95.
        X:            현재 마르코프 상태. 0=11v11, 1=홈퇴장, 2=원정퇴장, 3=양쪽.
        delta_S:      현재 스코어차 (S_H - S_A).
        a_H, a_A:     Phase 2에서 역산한 기본 득점 강도.
        b:            shape (6,) — 기저함수 계수.
        gamma:        shape (4,) — 마르코프 상태별 패널티 [0, γ₁, γ₂, γ₁+γ₂].
        delta_H:      shape (5,) — 홈 스코어차 효과 (0이 index 2에 삽입됨).
        delta_A:      shape (5,) — 어웨이 스코어차 효과.
        P_grid:       dict[int, ndarray(4,4)] — 행렬 지수함수 사전 계산.
                      P_grid[dt][i,j] = (exp(Q·dt))_{i,j}
        basis_bounds: shape (7,) — 기저함수 경계점.

    Returns:
        (μ_H, μ_A): 잔여 기대 득점 튜플.

    Notes:
        - 매초(1/60분) 호출됨. P_grid 조회 기반이므로 ~0.01ms.
        - δ(ΔS)는 현재 스코어차로 고정. 미래 ΔS 변화는 MC에서 처리.
        - t >= T이면 (0.0, 0.0) 반환.
    """
    if t >= T:
        return 0.0, 0.0

    # δ 인덱스 (현재 스코어차 → 0~4)
    di = delta_index(delta_S)

    # ── [t, T] 구간을 기저함수 경계에서 분할 ───────────
    # 커팅 포인트: t, (basis_bounds 중 t<τ<T인 것들), T
    cut_points = [t]
    for bound in basis_bounds:
        if t < bound < T:
            cut_points.append(float(bound))
    cut_points.append(T)

    # ── 소구간별 적분 ─────────────────────────────────
    mu_H = 0.0
    mu_A = 0.0
    elapsed_from_t = 0.0  # t부터 누적 경과 시간 (P_grid 조회용)

    for seg_idx in range(len(cut_points) - 1):
        seg_start = cut_points[seg_idx]
        seg_end = cut_points[seg_idx + 1]
        seg_duration = seg_end - seg_start

        if seg_duration <= 0:
            continue

        # 이 소구간의 기저함수 인덱스
        bi = get_basis_idx(seg_start, basis_bounds)

        # P̄: 소구간 중점까지의 누적 경과 시간으로 P_grid 조회
        # t부터의 경과 = (seg_start - t) + seg_duration/2
        dt_mid = elapsed_from_t + seg_duration / 2
        dt_key = max(0, min(100, round(dt_mid)))

        # P_grid[dt][X, j]: 상태 X에서 출발하여 dt분 후 상태 j에 있을 확률
        P_row = P_grid[dt_key][X, :]   # shape (4,)

        # 4개 상태에 대해 가중 강도 합산
        for j in range(4):
            # λ_H(j) = exp(a_H + b[bi] + γ[j] + δ_H[di])
            lambda_H = math.exp(a_H + b[bi] + gamma[j] + delta_H[di])
            lambda_A = math.exp(a_A + b[bi] + gamma[j] + delta_A[di])

            weight = P_row[j] * seg_duration
            mu_H += weight * lambda_H
            mu_A += weight * lambda_A

        elapsed_from_t += seg_duration

    return mu_H, mu_A


# ═════════════════════════════════════════════════════════
# 검증 유틸: 킥오프 시 μ vs Phase 2 μ̂ 교차 검증
# ═════════════════════════════════════════════════════════

def verify_kickoff_mu(
    mu_H_expected: float,
    mu_A_expected: float,
    a_H: float,
    a_A: float,
    b: np.ndarray,
    gamma: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    P_grid: Dict[int, np.ndarray],
    basis_bounds: np.ndarray,
    T_exp: float,
    tolerance: float = 0.01,
) -> Tuple[bool, float, float]:
    """
    킥오프 시점(t=0, X=0, ΔS=0)에서 compute_remaining_mu의 결과가
    Phase 2의 XGBoost 예측값(μ̂)과 일치하는지 검증한다.

    Phase 2 검증의 핵심: 역산 공식 a = ln(μ̂) - ln(C_time)이
    정확하다면, t=0에서 이 함수의 출력이 μ̂와 같아야 한다.

    Args:
        mu_H_expected, mu_A_expected: Phase 2 XGBoost 예측값
        tolerance: 허용 오차 (기본 1%)

    Returns:
        (pass, mu_H_actual, mu_A_actual)
    """
    mu_H, mu_A = compute_remaining_mu(
        t=0.0, T=T_exp, X=0, delta_S=0,
        a_H=a_H, a_A=a_A,
        b=b, gamma=gamma,
        delta_H=delta_H, delta_A=delta_A,
        P_grid=P_grid, basis_bounds=basis_bounds,
    )

    err_H = abs(mu_H - mu_H_expected)
    err_A = abs(mu_A - mu_A_expected)

    passed = (err_H < tolerance) and (err_A < tolerance)

    if not passed:
        logger.warning(
            f"킥오프 μ 불일치! "
            f"expected=({mu_H_expected:.4f}, {mu_A_expected:.4f}), "
            f"actual=({mu_H:.4f}, {mu_A:.4f}), "
            f"err=({err_H:.4f}, {err_A:.4f})"
        )
    else:
        logger.info(
            f"킥오프 μ 검증 통과: "
            f"({mu_H:.4f}, {mu_A:.4f}), err=({err_H:.6f}, {err_A:.6f})"
        )

    return passed, mu_H, mu_A
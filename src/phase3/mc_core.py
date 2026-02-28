"""
Phase 3 Step 3.4b: Monte Carlo 시뮬레이션 코어 (mc_core.py)

X≠0 또는 ΔS≠0일 때 독립성 가정이 깨지므로,
Thinning 알고리즘 기반 Monte Carlo로 최종 스코어 분포를 생성한다.

Thinning 알고리즘:
  1. 현재 상태(X, ΔS)에서 골/퇴장의 총 강도 λ_total 계산
  2. 지수 분포에서 다음 이벤트 시각 샘플링: dt ~ Exp(λ_total)
  3. 기저함수 경계 확인 (경계를 넘으면 강도 재계산)
  4. 이벤트 종류 결정: 홈골 / 어웨이골 / 퇴장
  5. 상태 업데이트 후 반복

구현 전략:
  - Numba @njit: 50~100x 속도. 프로덕션용. (numba 설치 필요)
  - Pure Python fallback: Numba 없을 때 테스트/개발용.
  - 동일한 인터페이스: mc_simulate_remaining(...)

사용법:
  from src.phase3.mc_core import mc_simulate_remaining, warmup

  # 워밍업 (Numba JIT 컴파일 트리거)
  warmup()

  # 시뮬레이션
  final_scores = mc_simulate_remaining(
      t_now=30.0, T_end=95.0,
      S_H=1, S_A=0, state=0, score_diff=1,
      a_H=-4.2, a_A=-4.4,
      b=b, gamma=gamma,
      delta_H=delta_H, delta_A=delta_A,
      Q_diag=Q_diag, Q_off=Q_off,
      basis_bounds=basis_bounds,
      N=50000, seed=42,
  )
  # final_scores.shape == (50000, 2)
"""

from __future__ import annotations

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Numba 가용 여부 확인
# ─────────────────────────────────────────────────────────

try:
    from numba import njit
    HAS_NUMBA = True
    logger.info("mc_core: Numba 사용 가능 → JIT 모드")
except ImportError:
    HAS_NUMBA = False
    logger.warning("mc_core: Numba 미설치 → Pure Python fallback (느림)")


# ═════════════════════════════════════════════════════════
# Pure Python 구현 (Numba 없을 때 fallback)
# ═════════════════════════════════════════════════════════

def _mc_simulate_python(
    t_now: float,
    T_end: float,
    S_H: int, S_A: int,
    state: int,
    score_diff: int,
    a_H: float, a_A: float,
    b: np.ndarray,
    gamma: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    Q_diag: np.ndarray,
    Q_off: np.ndarray,
    basis_bounds: np.ndarray,
    N: int,
    seed: int,
) -> np.ndarray:
    """
    Pure Python MC 시뮬레이션.

    Numba 버전과 동일한 로직. 테스트/개발용.
    N=50,000일 때 ~50ms (Numba의 ~100배 느림).
    """
    rng = np.random.RandomState(seed)
    results = np.empty((N, 2), dtype=np.int32)

    for sim in range(N):
        s = t_now
        sh, sa = S_H, S_A
        st = state
        sd = score_diff

        while s < T_end:
            # 현재 기저함수 인덱스
            bi = 0
            for k in range(6):
                if basis_bounds[k] <= s < basis_bounds[k + 1]:
                    bi = k
                    break

            # δ 인덱스: ΔS → {0:≤-2, 1:-1, 2:0, 3:+1, 4:≥+2}
            di = max(0, min(4, sd + 2))

            # 강도 계산
            lam_H = np.exp(a_H + b[bi] + gamma[st] + delta_H[di])
            lam_A = np.exp(a_A + b[bi] + gamma[st] + delta_A[di])
            lam_red = -Q_diag[st]
            lam_total = lam_H + lam_A + lam_red

            if lam_total <= 0:
                break

            # 다음 이벤트까지 대기 시간
            dt = -np.log(rng.random()) / lam_total
            s_next = s + dt

            # 다음 기저함수 경계 찾기
            next_bound = T_end
            for k in range(7):
                if basis_bounds[k] > s:
                    next_bound = min(next_bound, basis_bounds[k])
                    break

            # 경계를 넘으면 경계까지만 진행 (강도 재계산)
            if s_next >= next_bound:
                s = next_bound
                continue

            s = s_next

            # 이벤트 종류 결정
            u = rng.random() * lam_total
            if u < lam_H:
                sh += 1
                sd += 1
            elif u < lam_H + lam_A:
                sa += 1
                sd -= 1
            else:
                # 퇴장 → 전이 확률에 따라 새 상태
                cum = 0.0
                r = rng.random()
                for j in range(4):
                    if j == st:
                        continue
                    cum += Q_off[st, j]
                    if r < cum:
                        st = j
                        break

        results[sim, 0] = sh
        results[sim, 1] = sa

    return results


# ═════════════════════════════════════════════════════════
# Numba JIT 구현 (프로덕션)
# ═════════════════════════════════════════════════════════

if HAS_NUMBA:
    @njit(cache=True)
    def _mc_simulate_numba(
        t_now, T_end,
        S_H, S_A, state, score_diff,
        a_H, a_A,
        b, gamma, delta_H, delta_A,
        Q_diag, Q_off, basis_bounds,
        N, seed,
    ):
        """
        Numba JIT 컴파일된 MC 시뮬레이션 코어.

        Pure Python 대비 50~100× 속도 향상.
        첫 호출 시 ~2초 JIT 컴파일 소요 → warmup()으로 사전 처리.
        """
        np.random.seed(seed)
        results = np.empty((N, 2), dtype=np.int32)

        for sim in range(N):
            s = t_now
            sh, sa = S_H, S_A
            st = state
            sd = score_diff

            while s < T_end:
                # 기저함수 인덱스
                bi = 0
                for k in range(6):
                    if basis_bounds[k] <= s < basis_bounds[k + 1]:
                        bi = k
                        break

                di = max(0, min(4, sd + 2))

                lam_H = np.exp(a_H + b[bi] + gamma[st] + delta_H[di])
                lam_A = np.exp(a_A + b[bi] + gamma[st] + delta_A[di])
                lam_red = -Q_diag[st]
                lam_total = lam_H + lam_A + lam_red

                if lam_total <= 0:
                    break

                dt = -np.log(np.random.random()) / lam_total
                s_next = s + dt

                next_bound = T_end
                for k in range(7):
                    if basis_bounds[k] > s:
                        next_bound = min(next_bound, basis_bounds[k])
                        break

                if s_next >= next_bound:
                    s = next_bound
                    continue

                s = s_next

                u = np.random.random() * lam_total
                if u < lam_H:
                    sh += 1
                    sd += 1
                elif u < lam_H + lam_A:
                    sa += 1
                    sd -= 1
                else:
                    cum = 0.0
                    r = np.random.random()
                    for j in range(4):
                        if j == st:
                            continue
                        cum += Q_off[st, j]
                        if r < cum:
                            st = j
                            break

            results[sim, 0] = sh
            results[sim, 1] = sa

        return results


# ═════════════════════════════════════════════════════════
# 공개 인터페이스
# ═════════════════════════════════════════════════════════

def mc_simulate_remaining(
    t_now: float,
    T_end: float,
    S_H: int, S_A: int,
    state: int,
    score_diff: int,
    a_H: float, a_A: float,
    b: np.ndarray,
    gamma: np.ndarray,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    Q_diag: np.ndarray,
    Q_off: np.ndarray,
    basis_bounds: np.ndarray,
    N: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """
    MC 시뮬레이션으로 남은 경기를 N번 시뮬레이션한다.

    Numba가 설치되어 있으면 JIT 버전, 없으면 Pure Python 사용.

    Args:
        t_now:       현재 실효 플레이 시간 (분)
        T_end:       경기 종료 예정 시간 (분)
        S_H, S_A:    현재 스코어
        state:       현재 마르코프 상태 (0~3)
        score_diff:  현재 ΔS = S_H - S_A
        a_H, a_A:    기본 득점 강도 (Phase 2 역산)
        b:           shape (6,) — 기저함수 계수
        gamma:       shape (4,) — [0, γ₁, γ₂, γ₁+γ₂]
        delta_H:     shape (5,) — 홈 스코어차 효과
        delta_A:     shape (5,) — 어웨이 스코어차 효과
        Q_diag:      shape (4,) — Q의 대각 성분 (음수). -Q_diag[st] = 총 퇴장률.
        Q_off:       shape (4,4) — 비대각 전이 확률 (행 합 ≈ 1, 자기 자신 = 0).
        basis_bounds: shape (7,) — 기저함수 경계 시점
        N:           시뮬레이션 횟수 (기본 50,000)
        seed:        난수 시드

    Returns:
        shape (N, 2) int32 — 각 시뮬레이션의 (S_H_final, S_A_final)
    """
    if HAS_NUMBA:
        return _mc_simulate_numba(
            t_now, T_end, S_H, S_A, state, score_diff,
            a_H, a_A, b, gamma, delta_H, delta_A,
            Q_diag, Q_off, basis_bounds, N, seed,
        )
    else:
        return _mc_simulate_python(
            t_now, T_end, S_H, S_A, state, score_diff,
            a_H, a_A, b, gamma, delta_H, delta_A,
            Q_diag, Q_off, basis_bounds, N, seed,
        )


def build_Q_diag_and_off(Q: np.ndarray):
    """
    Q matrix에서 mc_core가 필요한 Q_diag, Q_off를 추출한다.

    Args:
        Q: shape (4,4) — 마르코프 생성 행렬

    Returns:
        Q_diag: shape (4,) — 대각 성분 (음수)
        Q_off:  shape (4,4) — 정규화된 비대각 전이 확률
                Q_off[i,j] = Q[i,j] / (-Q[i,i]) for i≠j, Q_off[i,i] = 0
    """
    Q_diag = np.diag(Q).copy()

    Q_off = np.zeros_like(Q)
    for i in range(4):
        rate = -Q_diag[i]
        if rate > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = Q[i, j] / rate
        # rate == 0: 흡수 상태 (state 3) → 전이 없음

    return Q_diag, Q_off


def warmup() -> float:
    """
    Numba JIT 컴파일을 사전 트리거한다.

    Phase 2 초기화 시 호출하면 Phase 3 진입 시 즉시 사용 가능.

    Returns:
        컴파일 소요 시간 (초). Numba 없으면 0.0.
    """
    t0 = time.time()

    _ = mc_simulate_remaining(
        t_now=0.0, T_end=1.0,
        S_H=0, S_A=0, state=0, score_diff=0,
        a_H=0.0, a_A=0.0,
        b=np.zeros(6), gamma=np.zeros(4),
        delta_H=np.zeros(5), delta_A=np.zeros(5),
        Q_diag=np.array([-0.01, -0.01, -0.01, 0.0]),
        Q_off=np.zeros((4, 4)),
        basis_bounds=np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 95.0]),
        N=10, seed=0,
    )

    elapsed = time.time() - t0
    logger.info(f"mc_core warmup 완료: {elapsed:.2f}s ({'Numba' if HAS_NUMBA else 'Python'})")
    return elapsed
"""
Phase 3 Step 3.4: 프라이싱 — True Probability 산출 (pricer.py)

잔여 기대 득점 μ_H, μ_A를 바탕으로 Kalshi 호가와 비교할 수 있는
진짜 확률(P_true)을 산출한다.

하이브리드 프라이싱:
  - 해석적 (X=0, ΔS=0): 독립 푸아송 가정 유효. ~0.01ms.
  - Monte Carlo (그 외): δ(ΔS)로 독립성 깨짐. Numba MC 코어 사용. ~0.5ms.

지원 시장:
  - Over/Under N.5 (N=0,1,2,3,4,5)
  - Match Odds (Home Win / Draw / Away Win)
  - Correct Score (디버깅/로깅용)

사용법:
  from src.phase3.pricer import (
      analytical_pricing, aggregate_mc_results, PricingResult, price
  )

  # 해석적
  result = analytical_pricing(mu_H=0.8, mu_A=0.5, S_H=1, S_A=0)

  # MC 결과 집계
  result = aggregate_mc_results(final_scores, S_H=1, S_A=0)

  # 하이브리드 디스패치
  result = price(mu_H, mu_A, S_H, S_A, X, delta_S)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 상수
MAX_GOALS_ANALYTICAL = 10      # 해석적 계산 시 최대 골 수
DEFAULT_MC_SIMULATIONS = 50_000


# ═════════════════════════════════════════════════════════
# 결과 데이터 클래스
# ═════════════════════════════════════════════════════════

@dataclass
class PricingResult:
    """
    프라이싱 결과.

    Attributes:
        home_win:   홈 승 확률
        draw:       무승부 확률
        away_win:   어웨이 승 확률
        over_25:    Over 2.5 확률
        under_25:   Under 2.5 확률
        over_15:    Over 1.5 확률
        over_35:    Over 3.5 확률
        sigma_mc:   Monte Carlo 표준오차 (해석적이면 0.0)
        mode:       "analytical" 또는 "monte_carlo"
    """
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0
    over_25: float = 0.0
    under_25: float = 0.0
    over_15: float = 0.0
    over_35: float = 0.0
    sigma_mc: float = 0.0
    mode: str = "analytical"

    def __post_init__(self):
        """확률 범위 클램핑 (수치 오차 방지)"""
        for attr in ("home_win", "draw", "away_win",
                      "over_25", "under_25", "over_15", "over_35"):
            val = max(0.0, min(1.0, getattr(self, attr)))
            object.__setattr__(self, attr, val)


# ═════════════════════════════════════════════════════════
# 해석적 프라이싱 (X=0, ΔS=0)
# ═════════════════════════════════════════════════════════

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(X=k) = λ^k · e^(-λ) / k!"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def analytical_pricing(
    mu_H: float,
    mu_A: float,
    S_H: int = 0,
    S_A: int = 0,
) -> PricingResult:
    """
    독립 푸아송 모델로 모든 시장의 확률을 해석적으로 계산한다.

    현재 스코어(S_H, S_A)에 잔여 기대 득점(μ_H, μ_A)을 더한
    최종 스코어 분포를 구한다.

    핵심 공식:
      P(최종 H = h) = P(잔여 H = h - S_H) = Poisson(h - S_H, μ_H)
      P(최종 A = a) = P(잔여 A = a - S_A) = Poisson(a - S_A, μ_A)

    Args:
        mu_H: 잔여 홈 기대 득점
        mu_A: 잔여 어웨이 기대 득점
        S_H:  현재 홈 스코어
        S_A:  현재 어웨이 스코어

    Returns:
        PricingResult (mode="analytical", sigma_mc=0.0)
    """
    max_g = MAX_GOALS_ANALYTICAL

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    over_15 = 0.0
    over_25 = 0.0
    over_35 = 0.0

    # 최종 스코어의 결합 분포
    for h_remaining in range(max_g + 1):
        ph = poisson_pmf(h_remaining, mu_H)
        final_h = S_H + h_remaining

        for a_remaining in range(max_g + 1):
            pa = poisson_pmf(a_remaining, mu_A)
            final_a = S_A + a_remaining
            joint = ph * pa

            total_goals = final_h + final_a

            # Match Odds
            if final_h > final_a:
                p_home += joint
            elif final_h == final_a:
                p_draw += joint
            else:
                p_away += joint

            # Over/Under
            if total_goals >= 2:
                over_15 += joint
            if total_goals >= 3:
                over_25 += joint
            if total_goals >= 4:
                over_35 += joint

    return PricingResult(
        home_win=p_home,
        draw=p_draw,
        away_win=p_away,
        over_25=over_25,
        under_25=1.0 - over_25,
        over_15=over_15,
        over_35=over_35,
        sigma_mc=0.0,
        mode="analytical",
    )


# ═════════════════════════════════════════════════════════
# Monte Carlo 결과 집계
# ═════════════════════════════════════════════════════════

def aggregate_mc_results(
    final_scores: np.ndarray,
    S_H: int = 0,
    S_A: int = 0,
) -> PricingResult:
    """
    MC 시뮬레이션 결과(final_scores)에서 시장별 확률을 집계한다.

    Args:
        final_scores: shape (N, 2) — 각 시뮬레이션의 최종 (S_H, S_A).
                      mc_core.mc_simulate_remaining()의 반환값.
        S_H, S_A:     현재 스코어 (MC가 현재 스코어를 포함해서 반환한다면 0).
                      MC가 잔여 골만 반환하면 현재 스코어를 더해야 함.

    Returns:
        PricingResult (mode="monte_carlo", sigma_mc 포함)
    """
    N = final_scores.shape[0]
    if N == 0:
        return PricingResult(mode="monte_carlo")

    # final_scores는 이미 현재 스코어가 포함된 최종 스코어로 가정
    h = final_scores[:, 0]
    a = final_scores[:, 1]
    total = h + a

    # Match Odds
    home_wins = np.sum(h > a)
    draws = np.sum(h == a)
    away_wins = np.sum(h < a)

    p_home = home_wins / N
    p_draw = draws / N
    p_away = away_wins / N

    # Over/Under
    p_over_15 = np.sum(total >= 2) / N
    p_over_25 = np.sum(total >= 3) / N
    p_over_35 = np.sum(total >= 4) / N

    # Monte Carlo 표준오차 (Over 2.5 기준 — 가장 많이 거래되는 시장)
    # σ_MC = sqrt(p*(1-p)/N)
    sigma_mc = math.sqrt(p_over_25 * (1 - p_over_25) / N) if N > 0 else 0.0

    return PricingResult(
        home_win=p_home,
        draw=p_draw,
        away_win=p_away,
        over_25=p_over_25,
        under_25=1.0 - p_over_25,
        over_15=p_over_15,
        over_35=p_over_35,
        sigma_mc=sigma_mc,
        mode="monte_carlo",
    )


# ═════════════════════════════════════════════════════════
# 하이브리드 디스패치
# ═════════════════════════════════════════════════════════

def price(
    mu_H: float,
    mu_A: float,
    S_H: int,
    S_A: int,
    X: int,
    delta_S: int,
    mc_final_scores: Optional[np.ndarray] = None,
) -> PricingResult:
    """
    하이브리드 프라이싱 디스패치.

    X=0 AND ΔS=0이면 해석적, 그 외 MC.
    MC 모드에서는 mc_final_scores가 필요하다
    (engine.py에서 executor를 통해 미리 계산해서 전달).

    Args:
        mu_H, mu_A: 잔여 기대 득점
        S_H, S_A:   현재 스코어
        X:          마르코프 상태
        delta_S:    현재 스코어차
        mc_final_scores: MC 시뮬레이션 결과 (MC 모드 시 필수)

    Returns:
        PricingResult

    Raises:
        ValueError: MC 모드인데 mc_final_scores가 None일 때
    """
    use_analytical = (X == 0 and delta_S == 0)

    if use_analytical:
        return analytical_pricing(mu_H, mu_A, S_H, S_A)
    else:
        if mc_final_scores is None:
            raise ValueError(
                f"MC 모드 (X={X}, ΔS={delta_S})인데 mc_final_scores가 None입니다. "
                "engine.py에서 mc_core를 먼저 실행해야 합니다."
            )
        return aggregate_mc_results(mc_final_scores, S_H, S_A)
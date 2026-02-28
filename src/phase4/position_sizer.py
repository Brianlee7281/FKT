"""
Phase 4 Step 4.3: 포지션 사이징 (position_sizer.py)

Fee-Adjusted Kelly Criterion으로 최적 투자 비중을 계산하고,
3-Layer 리스크 한도로 클램핑.

핵심 공식:
  1. f* = EV_adj / ((1-fee_pct) × P_ask × (1-P_ask))  [근사]
     → 단순화: f* = EV_adj / (P_mkt × (1-P_mkt))
  2. f_invest = K_frac × f*                             [Fractional Kelly]
  3. contracts = floor(f_invest × bankroll / P_mkt)     [계약 수 변환]
  4. 3-Layer 클램핑:
     Layer 1: 단일 주문 ≤ bankroll × 3%
     Layer 2: 경기별 합산 ≤ bankroll × 5%
     Layer 3: 전체 포트폴리오 ≤ bankroll × 20%

사용법:
  sizer = PositionSizer()
  result = sizer.size(
      ev_adj=0.05, p_kalshi=0.45, bankroll=10000,
      match_exposure=200, total_exposure=1000,
  )
  # result.contracts = 매수할 계약 수
  # result.dollar_amount = 투입 금액 (센트→달러)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════

@dataclass
class SizingResult:
    """포지션 사이징 결과."""
    contracts: int = 0            # 매수할 계약 수
    dollar_amount: float = 0.0    # 투입 금액 (달러)
    f_kelly: float = 0.0          # Full Kelly 비중
    f_invest: float = 0.0         # Fractional Kelly 비중
    clamped_by: str = ""          # 클램핑 사유 ("" / "layer1" / "layer2" / "layer3")
    reason: str = ""              # 0인 경우 사유


# ═══════════════════════════════════════════════════
# Position Sizer
# ═══════════════════════════════════════════════════

class PositionSizer:
    """
    Fee-Adjusted Kelly + 3-Layer 리스크 한도.

    Args:
        kelly_fraction:  Fractional Kelly 계수 (기본 0.25 = Quarter-Kelly)
        order_cap:       Layer 1 — 단일 주문 한도 (bankroll 대비, 기본 3%)
        match_cap:       Layer 2 — 경기별 한도 (bankroll 대비, 기본 5%)
        total_cap:       Layer 3 — 전체 한도 (bankroll 대비, 기본 20%)
        min_contracts:   최소 계약 수 (이하면 진입 안 함, 기본 1)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        order_cap: float = 0.03,
        match_cap: float = 0.05,
        total_cap: float = 0.20,
        min_contracts: int = 1,
    ):
        self.kelly_fraction = kelly_fraction
        self.order_cap = order_cap
        self.match_cap = match_cap
        self.total_cap = total_cap
        self.min_contracts = min_contracts

    # ─── Kelly 계산 ───────────────────────────────

    def kelly_fraction_full(self, ev_adj: float, p_market: float) -> float:
        """
        Full Kelly 비중.

        바이너리 옵션 Kelly:
          f* = EV / (W × L)
        여기서:
          W = 1 - P_mkt  (승리 시 수익률)
          L = P_mkt      (패배 시 손실률)

          f* = EV_adj / (P_mkt × (1 - P_mkt))

        EV_adj에 이미 수수료가 반영되어 있으므로
        별도 수수료 조정 불필요.
        """
        if ev_adj <= 0:
            return 0.0
        denom = p_market * (1.0 - p_market)
        if denom <= 0:
            return 0.0
        return ev_adj / denom

    # ─── 3-Layer 리스크 한도 ──────────────────────

    def apply_risk_limits(
        self,
        amount: float,
        bankroll: float,
        match_exposure: float = 0.0,
        total_exposure: float = 0.0,
    ) -> tuple[float, str]:
        """
        3-Layer 리스크 한도 적용.

        Returns:
            (clamped_amount, clamped_by)
        """
        clamped_by = ""

        # Layer 1: 단일 주문
        max_order = bankroll * self.order_cap
        if amount > max_order:
            amount = max_order
            clamped_by = "layer1"

        # Layer 2: 경기별
        remaining_match = bankroll * self.match_cap - match_exposure
        if remaining_match <= 0:
            return 0.0, "layer2_full"
        if amount > remaining_match:
            amount = remaining_match
            clamped_by = "layer2"

        # Layer 3: 전체 포트폴리오
        remaining_total = bankroll * self.total_cap - total_exposure
        if remaining_total <= 0:
            return 0.0, "layer3_full"
        if amount > remaining_total:
            amount = remaining_total
            clamped_by = "layer3"

        return amount, clamped_by

    # ─── 메인 사이징 ─────────────────────────────

    def size(
        self,
        ev_adj: float,
        p_kalshi: float,
        bankroll: float,
        match_exposure: float = 0.0,
        total_exposure: float = 0.0,
    ) -> SizingResult:
        """
        포지션 사이즈 계산.

        Args:
            ev_adj:          수수료 반영 기댓값 (달러, 0~1 스케일)
            p_kalshi:        시장 가격 (달러, 0~1 스케일)
            bankroll:        현재 잔고 (달러)
            match_exposure:  현재 경기 노출액 (달러)
            total_exposure:  전체 포트폴리오 노출액 (달러)

        Returns:
            SizingResult
        """
        result = SizingResult()

        # ── 기본 검증 ────────────────────────────
        if ev_adj <= 0:
            result.reason = "ev_nonpositive"
            return result
        if bankroll <= 0:
            result.reason = "no_bankroll"
            return result
        if p_kalshi <= 0 or p_kalshi >= 1:
            result.reason = "invalid_price"
            return result

        # ── Kelly 계산 ───────────────────────────
        f_kelly = self.kelly_fraction_full(ev_adj, p_kalshi)
        f_invest = f_kelly * self.kelly_fraction
        result.f_kelly = f_kelly
        result.f_invest = f_invest

        # ── 달러 금액 변환 ───────────────────────
        dollar_amount = f_invest * bankroll

        # ── 3-Layer 리스크 한도 ──────────────────
        dollar_amount, clamped_by = self.apply_risk_limits(
            dollar_amount, bankroll, match_exposure, total_exposure,
        )
        result.clamped_by = clamped_by

        if dollar_amount <= 0:
            result.reason = clamped_by
            return result

        # ── 계약 수 변환 ─────────────────────────
        # 계약당 비용 = p_kalshi (달러)
        # 계약 수 = floor(dollar_amount / p_kalshi)
        contracts = int(dollar_amount / p_kalshi)

        if contracts < self.min_contracts:
            result.reason = f"below_min({contracts}<{self.min_contracts})"
            return result

        result.contracts = contracts
        result.dollar_amount = contracts * p_kalshi
        return result

    # ─── 유틸리티 ────────────────────────────────

    @staticmethod
    def format_result(r: SizingResult) -> str:
        """사람이 읽을 수 있는 포맷."""
        if r.contracts == 0:
            return f"SIZE=0 ({r.reason})"
        clamp = f" [clamped: {r.clamped_by}]" if r.clamped_by else ""
        return (
            f"SIZE={r.contracts} contracts @ ${r.dollar_amount:.2f}  "
            f"f*={r.f_kelly:.4f} f_inv={r.f_invest:.4f}"
            f"{clamp}"
        )

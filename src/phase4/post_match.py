"""
Phase 4 Step 4.6: 경기 종료 정산 & 사후 분석 (post_match.py)

경기 종료 후 모든 포지션을 정산하고,
모델 성능을 분석하여 Phase 1 재학습에 피드백.

핵심 지표:
  1. 경기별 P&L
  2. Brier Score (누적 모델 정확도)
  3. Edge 실현율 (예상 EV vs 실제 수익)
  4. 슬리피지 실적
  5. 쿨다운 효과 분석

모델 건강 대시보드:
  | 지표          | 건강      | 경고       | 위험     |
  |--------------|----------|-----------|---------|
  | Brier Score  | ±0.02    | ±0.05     | 벗어남   |
  | Edge 실현율  | 0.7~1.3  | 0.5~0.7   | <0.5    |
  | 누적 P&L     | 양의 기울기 | 횡보      | 음의 기울기 |
  | Max Drawdown | <10%     | 10~20%    | >20%    |

Phase 1 재학습 트리거:
  - Brier Score 3주 연속 악화
  - Edge 실현율 < 0.5
  - 새 시즌 시작
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 열거형 & 데이터 클래스
# ═══════════════════════════════════════════════════

class HealthStatus(Enum):
    """모델 건강 상태."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DANGER = "DANGER"


class RetrainTrigger(Enum):
    """Phase 1 재학습 트리거."""
    NONE = "NONE"
    BRIER_DEGRADATION = "BRIER_DEGRADATION"
    LOW_EDGE_REALIZATION = "LOW_EDGE_REALIZATION"
    HIGH_DRAWDOWN = "HIGH_DRAWDOWN"
    NEW_SEASON = "NEW_SEASON"


@dataclass
class TradeOutcome:
    """
    단일 거래의 결과 (정산 완료 후).

    진입 시점의 모델 예측과 실제 결과를 기록.
    """
    ticker: str = ""
    match_id: str = ""
    direction: str = ""          # BUY_YES / BUY_NO
    entry_price: float = 0.0     # 진입가 (0~1)
    fill_price: float = 0.0      # 실제 체결가 (0~1)
    signal_price: float = 0.0    # 시그널 시점 best quote (0~1)
    p_true_at_entry: float = 0.0 # 진입 시 P_true
    ev_adj_at_entry: float = 0.0 # 진입 시 EV_adj
    outcome: int = 0             # 1=Yes 승리, 0=No 승리
    pnl: float = 0.0             # 실현 P&L (달러)
    contracts: int = 0
    timestamp: str = ""


@dataclass
class MatchReport:
    """경기별 분석 리포트."""
    match_id: str = ""
    total_pnl: float = 0.0
    trade_count: int = 0
    brier_score: float = 0.0
    edge_realization: float = 0.0
    avg_slippage: float = 0.0
    outcomes: List[TradeOutcome] = field(default_factory=list)


@dataclass
class HealthDashboard:
    """모델 건강 대시보드."""
    cumulative_brier: float = 0.0
    brier_status: HealthStatus = HealthStatus.HEALTHY
    edge_realization: float = 0.0
    edge_status: HealthStatus = HealthStatus.HEALTHY
    cumulative_pnl: float = 0.0
    pnl_status: HealthStatus = HealthStatus.HEALTHY
    max_drawdown_pct: float = 0.0
    drawdown_status: HealthStatus = HealthStatus.HEALTHY
    total_trades: int = 0
    total_matches: int = 0
    retrain_triggers: List[RetrainTrigger] = field(default_factory=list)
    kelly_adjustment: str = ""   # "maintain" / "increase" / "decrease"


# ═══════════════════════════════════════════════════
# Post-Match Analyzer
# ═══════════════════════════════════════════════════

class PostMatchAnalyzer:
    """
    경기 종료 사후 분석기.

    모든 거래 결과를 축적하고,
    모델 건강 지표를 계산하여 Phase 1 재학습 판단.

    Args:
        baseline_brier:  Phase 1.5 검증 Brier Score (기준선)
        brier_warn:      Brier 경고 임계값 (기준 대비 차이)
        brier_danger:    Brier 위험 임계값
        edge_warn:       Edge 실현율 경고 하한
        edge_danger:     Edge 실현율 위험 하한
        drawdown_warn:   Drawdown 경고 (%)
        drawdown_danger: Drawdown 위험 (%)
        retrain_weeks:   Brier 악화 연속 주 수 → 재학습
    """

    def __init__(
        self,
        baseline_brier: float = 0.20,
        brier_warn: float = 0.05,
        brier_danger: float = 0.10,
        edge_warn: float = 0.7,
        edge_danger: float = 0.5,
        drawdown_warn: float = 10.0,
        drawdown_danger: float = 20.0,
        retrain_weeks: int = 3,
    ):
        self.baseline_brier = baseline_brier
        self.brier_warn = brier_warn
        self.brier_danger = brier_danger
        self.edge_warn = edge_warn
        self.edge_danger = edge_danger
        self.drawdown_warn = drawdown_warn
        self.drawdown_danger = drawdown_danger
        self.retrain_weeks = retrain_weeks

        # 누적 데이터
        self.all_outcomes: List[TradeOutcome] = []
        self.match_reports: List[MatchReport] = []
        self.weekly_brier: List[float] = []  # 주간 Brier 추이
        self.peak_bankroll: float = 0.0
        self.current_bankroll: float = 0.0

    # ─── 핵심 지표 계산 ──────────────────────────

    def brier_score(self, outcomes: List[TradeOutcome]) -> float:
        """
        Brier Score 계산.

        BS = (1/N) × Σ(P_true - O)²
        O = 1 (Yes 승리) or 0 (No 승리)

        낮을수록 좋음. 완벽 = 0, 무작위 = 0.25
        """
        if not outcomes:
            return 0.0
        total = sum(
            (o.p_true_at_entry - o.outcome) ** 2
            for o in outcomes
        )
        return total / len(outcomes)

    def edge_realization_rate(self, outcomes: List[TradeOutcome]) -> float:
        """
        Edge 실현율.

        실현율 = 실제 평균 수익률 / 예상 평균 EV_adj

        1.0 = 모델 정확, <0.5 = 모델 과신
        """
        if not outcomes:
            return 0.0

        avg_expected_ev = sum(o.ev_adj_at_entry for o in outcomes) / len(outcomes)
        if avg_expected_ev <= 0:
            return 0.0

        # 실제 계약당 수익률
        actual_returns = []
        for o in outcomes:
            if o.contracts > 0:
                per_contract_pnl = o.pnl / o.contracts
                actual_returns.append(per_contract_pnl)

        if not actual_returns:
            return 0.0

        avg_actual = sum(actual_returns) / len(actual_returns)
        return avg_actual / avg_expected_ev

    def avg_slippage(self, outcomes: List[TradeOutcome]) -> float:
        """
        평균 슬리피지 (센트).

        Slippage = Fill Price - Signal Price (시그널 시점 best quote)
        양수 = 불리한 체결
        """
        slippages = []
        for o in outcomes:
            if o.signal_price > 0 and o.fill_price > 0:
                slip = (o.fill_price - o.signal_price) * 100  # 센트
                slippages.append(slip)
        if not slippages:
            return 0.0
        return sum(slippages) / len(slippages)

    def match_pnl(self, outcomes: List[TradeOutcome]) -> float:
        """경기별 총 P&L."""
        return sum(o.pnl for o in outcomes)

    # ─── 경기별 리포트 ───────────────────────────

    def analyze_match(
        self,
        match_id: str,
        outcomes: List[TradeOutcome],
    ) -> MatchReport:
        """
        단일 경기 사후 분석.

        Args:
            match_id: 경기 ID
            outcomes: 해당 경기의 거래 결과 목록

        Returns:
            MatchReport
        """
        report = MatchReport(
            match_id=match_id,
            total_pnl=self.match_pnl(outcomes),
            trade_count=len(outcomes),
            brier_score=self.brier_score(outcomes),
            edge_realization=self.edge_realization_rate(outcomes),
            avg_slippage=self.avg_slippage(outcomes),
            outcomes=outcomes,
        )

        # 누적 데이터에 추가
        self.all_outcomes.extend(outcomes)
        self.match_reports.append(report)

        return report

    # ─── 모델 건강 대시보드 ──────────────────────

    def health_dashboard(
        self,
        current_bankroll: float = 0.0,
        peak_bankroll: float = 0.0,
    ) -> HealthDashboard:
        """
        전체 모델 건강 대시보드.

        100+ 거래 축적 후 의미 있는 판단 가능.
        """
        self.current_bankroll = current_bankroll
        self.peak_bankroll = max(peak_bankroll, self.peak_bankroll)

        dash = HealthDashboard(
            total_trades=len(self.all_outcomes),
            total_matches=len(self.match_reports),
        )

        # ── Brier Score ──────────────────────────
        dash.cumulative_brier = self.brier_score(self.all_outcomes)
        brier_diff = dash.cumulative_brier - self.baseline_brier
        if abs(brier_diff) <= self.brier_warn:
            dash.brier_status = HealthStatus.HEALTHY
        elif abs(brier_diff) <= self.brier_danger:
            dash.brier_status = HealthStatus.WARNING
        else:
            dash.brier_status = HealthStatus.DANGER

        # ── Edge 실현율 ──────────────────────────
        dash.edge_realization = self.edge_realization_rate(self.all_outcomes)
        if dash.edge_realization >= self.edge_warn:
            dash.edge_status = HealthStatus.HEALTHY
        elif dash.edge_realization >= self.edge_danger:
            dash.edge_status = HealthStatus.WARNING
        else:
            dash.edge_status = HealthStatus.DANGER

        # ── 누적 P&L ────────────────────────────
        dash.cumulative_pnl = sum(o.pnl for o in self.all_outcomes)
        if dash.cumulative_pnl > 0:
            dash.pnl_status = HealthStatus.HEALTHY
        elif dash.cumulative_pnl > -abs(current_bankroll * 0.05) if current_bankroll > 0 else True:
            dash.pnl_status = HealthStatus.WARNING
        else:
            dash.pnl_status = HealthStatus.DANGER

        # ── Max Drawdown ─────────────────────────
        if self.peak_bankroll > 0 and current_bankroll > 0:
            dash.max_drawdown_pct = (1.0 - current_bankroll / self.peak_bankroll) * 100
        if dash.max_drawdown_pct < self.drawdown_warn:
            dash.drawdown_status = HealthStatus.HEALTHY
        elif dash.max_drawdown_pct < self.drawdown_danger:
            dash.drawdown_status = HealthStatus.WARNING
        else:
            dash.drawdown_status = HealthStatus.DANGER

        # ── 재학습 트리거 ────────────────────────
        dash.retrain_triggers = self._check_retrain_triggers(dash)

        # ── Kelly 조정 판단 ──────────────────────
        dash.kelly_adjustment = self._kelly_adjustment(dash)

        return dash

    # ─── Phase 1 재학습 트리거 ────────────────────

    def _check_retrain_triggers(self, dash: HealthDashboard) -> List[RetrainTrigger]:
        """재학습 트리거 체크."""
        triggers = []

        # Brier Score 연속 악화
        if len(self.weekly_brier) >= self.retrain_weeks:
            recent = self.weekly_brier[-self.retrain_weeks:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                triggers.append(RetrainTrigger.BRIER_DEGRADATION)

        # Edge 실현율 위험
        if (dash.total_trades >= 50 and
                dash.edge_realization < self.edge_danger):
            triggers.append(RetrainTrigger.LOW_EDGE_REALIZATION)

        # 과도한 Drawdown
        if dash.max_drawdown_pct > self.drawdown_danger:
            triggers.append(RetrainTrigger.HIGH_DRAWDOWN)

        return triggers

    def _kelly_adjustment(self, dash: HealthDashboard) -> str:
        """
        K_frac 조정 판단.

        | Edge 실현율 | 조치                |
        |------------|-------------------|
        | ≥ 0.8      | 상향 (0.25→0.50)  |
        | 0.5~0.8    | 유지              |
        | < 0.5      | 하향 or 중단       |
        """
        if dash.total_trades < 100:
            return "maintain (insufficient data)"

        if dash.edge_realization >= 0.8:
            return "increase (0.25→0.50)"
        elif dash.edge_realization >= 0.5:
            return "maintain"
        else:
            return "decrease (consider pause)"

    # ─── 주간 업데이트 ───────────────────────────

    def record_weekly_brier(self):
        """주간 Brier Score 기록 (매주 호출)."""
        if self.all_outcomes:
            bs = self.brier_score(self.all_outcomes[-50:])  # 최근 50거래
            self.weekly_brier.append(bs)

    # ─── 유틸리티 ────────────────────────────────

    @staticmethod
    def format_match_report(r: MatchReport) -> str:
        """경기 리포트 포맷."""
        return (
            f"Match [{r.match_id}]  "
            f"P&L=${r.total_pnl:+.2f}  "
            f"trades={r.trade_count}  "
            f"Brier={r.brier_score:.4f}  "
            f"EdgeReal={r.edge_realization:.2f}  "
            f"Slip={r.avg_slippage:+.2f}¢"
        )

    @staticmethod
    def format_dashboard(d: HealthDashboard) -> str:
        """대시보드 포맷."""
        lines = [
            "═" * 55,
            "  MODEL HEALTH DASHBOARD",
            "═" * 55,
            f"  Trades: {d.total_trades}  Matches: {d.total_matches}",
            f"  Cumulative P&L: ${d.cumulative_pnl:+.2f}  [{d.pnl_status.value}]",
            f"  Brier Score:    {d.cumulative_brier:.4f}  [{d.brier_status.value}]",
            f"  Edge Realized:  {d.edge_realization:.2f}   [{d.edge_status.value}]",
            f"  Max Drawdown:   {d.max_drawdown_pct:.1f}%  [{d.drawdown_status.value}]",
            f"  Kelly Adj:      {d.kelly_adjustment}",
        ]
        if d.retrain_triggers:
            triggers_str = ", ".join(t.value for t in d.retrain_triggers)
            lines.append(f"  ⚠️ RETRAIN: {triggers_str}")
        lines.append("═" * 55)
        return "\n".join(lines)

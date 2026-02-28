"""
test_post_match.py — PostMatchAnalyzer 단위 테스트.

Brier Score, Edge 실현율, 슬리피지, 건강 대시보드, 재학습 트리거.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4.post_match import (
    PostMatchAnalyzer, TradeOutcome, MatchReport,
    HealthDashboard, HealthStatus, RetrainTrigger,
)


def make_outcome(p_true=0.60, outcome=1, pnl=0.15, ev=0.10,
                 contracts=100, entry=0.45, fill=0.46, signal=0.45,
                 ticker="T", match_id="M"):
    return TradeOutcome(
        ticker=ticker, match_id=match_id, direction="BUY_YES",
        entry_price=entry, fill_price=fill, signal_price=signal,
        p_true_at_entry=p_true, ev_adj_at_entry=ev,
        outcome=outcome, pnl=pnl, contracts=contracts,
    )


def test_brier_perfect():
    """T1: Brier Score — 완벽한 예측."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(p_true=1.0, outcome=1),
        make_outcome(p_true=0.0, outcome=0),
    ]
    bs = pa.brier_score(outcomes)
    assert bs == 0.0
    print(f"✅ T1: Perfect Brier = {bs:.4f}")


def test_brier_random():
    """T2: Brier Score — 무작위 (항상 0.5)."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(p_true=0.50, outcome=1),
        make_outcome(p_true=0.50, outcome=0),
    ]
    bs = pa.brier_score(outcomes)
    assert abs(bs - 0.25) < 0.001
    print(f"✅ T2: Random Brier = {bs:.4f}")


def test_brier_typical():
    """T3: Brier Score — 현실적 예측."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(p_true=0.70, outcome=1),  # (0.7-1)²=0.09
        make_outcome(p_true=0.30, outcome=0),  # (0.3-0)²=0.09
        make_outcome(p_true=0.80, outcome=0),  # (0.8-0)²=0.64
    ]
    bs = pa.brier_score(outcomes)
    expected = (0.09 + 0.09 + 0.64) / 3  # 0.2733
    assert abs(bs - expected) < 0.001
    print(f"✅ T3: Typical Brier = {bs:.4f}")


def test_brier_empty():
    """T4: Brier Score — 빈 리스트."""
    pa = PostMatchAnalyzer()
    assert pa.brier_score([]) == 0.0
    print("✅ T4: Empty Brier = 0")


def test_edge_realization():
    """T5: Edge 실현율 — 모델 정확."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(ev=0.10, pnl=10.0, contracts=100),  # per_contract=0.10
        make_outcome(ev=0.08, pnl=8.0, contracts=100),   # per_contract=0.08
    ]
    er = pa.edge_realization_rate(outcomes)
    # avg_ev = 0.09, avg_actual = 0.09 → 실현율 = 1.0
    assert abs(er - 1.0) < 0.01
    print(f"✅ T5: Edge realization = {er:.2f} (perfect)")


def test_edge_realization_overconfident():
    """T6: Edge 실현율 — 과신 모델."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(ev=0.10, pnl=3.0, contracts=100),   # actual=0.03
        make_outcome(ev=0.10, pnl=2.0, contracts=100),   # actual=0.02
    ]
    er = pa.edge_realization_rate(outcomes)
    # avg_ev=0.10, avg_actual=0.025 → 0.25
    assert er < 0.5
    print(f"✅ T6: Edge realization = {er:.2f} (overconfident)")


def test_slippage():
    """T7: 슬리피지 계산."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(fill=0.46, signal=0.45),  # +1¢
        make_outcome(fill=0.47, signal=0.45),  # +2¢
        make_outcome(fill=0.45, signal=0.45),  # 0¢
    ]
    slip = pa.avg_slippage(outcomes)
    assert abs(slip - 1.0) < 0.01  # 평균 1¢
    print(f"✅ T7: Avg slippage = {slip:.2f}¢")


def test_match_report():
    """T8: 경기 리포트 생성."""
    pa = PostMatchAnalyzer()
    outcomes = [
        make_outcome(p_true=0.70, outcome=1, pnl=15.0, ev=0.10,
                     contracts=100, match_id="EPL001"),
        make_outcome(p_true=0.30, outcome=0, pnl=-10.0, ev=0.08,
                     contracts=50, match_id="EPL001"),
    ]
    report = pa.analyze_match("EPL001", outcomes)
    assert report.match_id == "EPL001"
    assert report.trade_count == 2
    assert abs(report.total_pnl - 5.0) < 0.01
    assert report.brier_score > 0
    assert len(pa.all_outcomes) == 2
    print(f"✅ T8: Match report P&L=${report.total_pnl:+.2f}")


def test_dashboard_healthy():
    """T9: 건강한 대시보드."""
    pa = PostMatchAnalyzer(baseline_brier=0.20)
    # 좋은 결과 50개 추가
    for i in range(50):
        pa.all_outcomes.append(
            make_outcome(p_true=0.65, outcome=1, pnl=5.0, ev=0.05, contracts=100)
        )
    pa.match_reports = [MatchReport(match_id=f"M{i}") for i in range(10)]

    dash = pa.health_dashboard(current_bankroll=11000, peak_bankroll=11000)
    assert dash.pnl_status == HealthStatus.HEALTHY
    assert dash.cumulative_pnl > 0
    assert dash.max_drawdown_pct == 0.0
    print(f"✅ T9: Healthy dashboard (P&L=${dash.cumulative_pnl:.0f})")


def test_dashboard_danger():
    """T10: 위험 대시보드 — 높은 drawdown."""
    pa = PostMatchAnalyzer()
    for i in range(20):
        pa.all_outcomes.append(
            make_outcome(p_true=0.50, outcome=0, pnl=-10.0, ev=0.05, contracts=100)
        )
    pa.match_reports = [MatchReport() for _ in range(5)]

    dash = pa.health_dashboard(current_bankroll=7000, peak_bankroll=10000)
    assert abs(dash.max_drawdown_pct - 30.0) < 0.01
    assert dash.drawdown_status == HealthStatus.DANGER
    print(f"✅ T10: Danger dashboard (DD={dash.max_drawdown_pct:.0f}%)")


def test_retrain_brier_degradation():
    """T11: Brier 3주 연속 악화 → 재학습 트리거."""
    pa = PostMatchAnalyzer(retrain_weeks=3)
    pa.weekly_brier = [0.18, 0.20, 0.22, 0.25]  # 연속 악화
    # 더미 데이터
    pa.all_outcomes = [make_outcome() for _ in range(10)]
    pa.match_reports = [MatchReport() for _ in range(3)]

    dash = pa.health_dashboard(current_bankroll=10000, peak_bankroll=10000)
    assert RetrainTrigger.BRIER_DEGRADATION in dash.retrain_triggers
    print("✅ T11: Brier degradation → RETRAIN trigger")


def test_retrain_low_edge():
    """T12: Edge 실현율 < 0.5 (50+ trades) → 재학습 트리거."""
    pa = PostMatchAnalyzer()
    for i in range(60):
        pa.all_outcomes.append(
            make_outcome(ev=0.10, pnl=2.0, contracts=100)  # actual=0.02, rate=0.2
        )
    pa.match_reports = [MatchReport() for _ in range(15)]

    dash = pa.health_dashboard(current_bankroll=10000, peak_bankroll=10000)
    assert RetrainTrigger.LOW_EDGE_REALIZATION in dash.retrain_triggers
    print(f"✅ T12: Low edge realization → RETRAIN (rate={dash.edge_realization:.2f})")


def test_kelly_adjustment():
    """T13: Kelly 조정 판단."""
    pa = PostMatchAnalyzer()
    # 100+ trades, 좋은 edge 실현
    for i in range(110):
        pa.all_outcomes.append(
            make_outcome(ev=0.05, pnl=5.0, contracts=100)
        )
    pa.match_reports = [MatchReport() for _ in range(30)]

    dash = pa.health_dashboard(current_bankroll=12000, peak_bankroll=12000)
    assert "increase" in dash.kelly_adjustment
    print(f"✅ T13: Kelly adj = {dash.kelly_adjustment}")


def test_format_dashboard():
    """T14: 대시보드 포맷 출력."""
    pa = PostMatchAnalyzer()
    for i in range(20):
        pa.all_outcomes.append(make_outcome(pnl=3.0, ev=0.05, contracts=100))
    pa.match_reports = [MatchReport() for _ in range(5)]

    dash = pa.health_dashboard(current_bankroll=10500, peak_bankroll=10500)
    output = PostMatchAnalyzer.format_dashboard(dash)
    assert "MODEL HEALTH DASHBOARD" in output
    assert "HEALTHY" in output or "WARNING" in output
    print(f"✅ T14: Dashboard formatted\n{output}")


if __name__ == "__main__":
    test_brier_perfect()
    test_brier_random()
    test_brier_typical()
    test_brier_empty()
    test_edge_realization()
    test_edge_realization_overconfident()
    test_slippage()
    test_match_report()
    test_dashboard_healthy()
    test_dashboard_danger()
    test_retrain_brier_degradation()
    test_retrain_low_edge()
    test_kelly_adjustment()
    test_format_dashboard()

    print(f"\n{'='*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'='*50}")

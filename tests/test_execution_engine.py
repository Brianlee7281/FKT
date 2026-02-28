"""
test_execution_engine.py — ExecutionEngine 단위 테스트.

포트폴리오 상태, 진입/청산/정산, 거래 로그, 리스크 대시보드.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4.execution_engine import (
    ExecutionEngine, PortfolioState, TradeRecord,
    OrderType, FillStatus,
)
from src.phase4.edge_detector import Direction, EdgeSignal
from src.phase4.position_sizer import SizingResult
from src.phase4.exit_manager import ExitAction, ExitDecision, OpenPosition


def make_signal(ticker="TEST-HW", p_true=0.60, p_kalshi=0.45,
                ev=0.13, direction=Direction.BUY_YES):
    return EdgeSignal(
        direction=direction, ev_adj=ev,
        p_true=p_true, p_true_cons=p_true,
        p_kalshi=p_kalshi, sigma_mc=0.0,
        fee_per_contract=0.017, ticker=ticker,
        market_type="home_win",
    )


def make_sizing(contracts=100, dollar=45.0, f_kelly=0.20, f_invest=0.05):
    return SizingResult(
        contracts=contracts, dollar_amount=dollar,
        f_kelly=f_kelly, f_invest=f_invest,
    )


def test_initial_state():
    """T1: 초기 포트폴리오 상태."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    assert eng.bankroll == 10000
    assert len(eng.positions) == 0
    assert eng.get_total_exposure() == 0
    assert eng.get_drawdown() == 0
    print("✅ T1: Initial state OK")


def test_paper_entry():
    """T2: Paper 진입 — 포지션 생성 + 잔고 차감."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal(p_kalshi=0.45)
    sizing = make_sizing(contracts=100)

    rec = eng.process_entry(sig, sizing, match_id="MATCH1")

    assert rec.fill_status == "PAPER"
    assert rec.quantity_filled == 100
    assert "TEST-HW" in eng.positions
    assert eng.positions["TEST-HW"].contracts == 100
    assert eng.bankroll < 10000  # 비용 차감
    print(f"✅ T2: Paper entry → {rec.quantity_filled} contracts, bankroll=${eng.bankroll:.2f}")


def test_paper_entry_cost():
    """T3: 진입 비용 정확성 (cost + fee)."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal(p_kalshi=0.50)  # 50¢
    sizing = make_sizing(contracts=100)

    eng.process_entry(sig, sizing, match_id="M1")

    # cost = 100 × $0.50 = $50
    # fee = 0.07 × 0.50 × 0.50 × 100 = $1.75
    # total = $51.75
    expected = 10000 - 50 - 1.75
    assert abs(eng.bankroll - expected) < 0.01
    print(f"✅ T3: Entry cost correct (bankroll=${eng.bankroll:.2f})")


def test_sizing_zero():
    """T4: Sizing 0 → 주문 안 함."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal()
    sizing = SizingResult(contracts=0, reason="ev_nonpositive")

    rec = eng.process_entry(sig, sizing)
    assert rec.fill_status == "UNFILLED"
    assert eng.bankroll == 10000
    print("✅ T4: Sizing=0 → no order")


def test_paper_exit():
    """T5: Paper 청산 — 포지션 제거 + P&L."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    # 진입
    sig = make_signal(p_kalshi=0.40)
    sizing = make_sizing(contracts=50)
    eng.process_entry(sig, sizing, match_id="M1")

    bankroll_after_entry = eng.bankroll

    # 청산
    pos = eng.positions["TEST-HW"]
    decision = ExitDecision(
        ticker="TEST-HW", action=ExitAction.CLOSE,
        trigger="edge_erosion", exit_price_cents=55,
    )
    rec = eng.process_exit(decision, pos)

    assert rec.fill_status == "PAPER"
    assert "TEST-HW" not in eng.positions  # 포지션 제거
    assert eng.bankroll > bankroll_after_entry  # 이익 실현
    assert rec.pnl > 0
    print(f"✅ T5: Paper exit → pnl=${rec.pnl:.2f}, bankroll=${eng.bankroll:.2f}")


def test_exit_hold():
    """T6: HOLD 판정 → 주문 안 함."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    pos = OpenPosition(ticker="TEST", direction=Direction.BUY_YES,
                       contracts=50, entry_price=0.40)
    decision = ExitDecision(ticker="TEST", action=ExitAction.HOLD)

    rec = eng.process_exit(decision, pos)
    assert rec.fill_status == "UNFILLED"
    print("✅ T6: HOLD → no exit order")


def test_match_exposure_tracking():
    """T7: 경기별 노출 추적."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)

    sig1 = make_signal(ticker="M1-HW", p_kalshi=0.40)
    sizing1 = make_sizing(contracts=50)
    eng.process_entry(sig1, sizing1, match_id="MATCH_A")

    sig2 = make_signal(ticker="M1-AW", p_kalshi=0.30)
    sizing2 = make_sizing(contracts=30)
    eng.process_entry(sig2, sizing2, match_id="MATCH_A")

    # MATCH_A 노출 = 50×0.40 + 30×0.30 = 20 + 9 = 29
    assert eng.get_match_exposure("MATCH_A") > 0
    assert eng.get_total_exposure() > 0
    print(f"✅ T7: Match exposure = ${eng.get_match_exposure('MATCH_A'):.2f}")


def test_settlement_yes_win():
    """T8: 정산 — Yes 승리."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal(ticker="SETTLE-HW", p_kalshi=0.40)
    sizing = make_sizing(contracts=100)
    eng.process_entry(sig, sizing, match_id="SETTLE_M")

    bankroll_before_settle = eng.bankroll

    records = eng.settle_match("SETTLE_M", {"SETTLE-HW": True})
    assert len(records) == 1
    assert records[0].pnl > 0  # 40¢에 사서 $1 수령
    assert "SETTLE-HW" not in eng.positions
    assert eng.bankroll > bankroll_before_settle
    print(f"✅ T8: Settlement Yes win → pnl=${records[0].pnl:.2f}")


def test_settlement_yes_lose():
    """T9: 정산 — Yes 패배."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal(ticker="LOSE-HW", p_kalshi=0.60)
    sizing = make_sizing(contracts=50)
    eng.process_entry(sig, sizing, match_id="LOSE_M")

    records = eng.settle_match("LOSE_M", {"LOSE-HW": False})
    assert len(records) == 1
    assert records[0].pnl < 0  # 60¢에 사서 $0 수령
    assert "LOSE-HW" not in eng.positions
    print(f"✅ T9: Settlement Yes lose → pnl=${records[0].pnl:.2f}")


def test_drawdown():
    """T10: 드로다운 계산."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    # 손실 시뮬레이션
    eng.portfolio.bankroll = 8000
    dd = eng.get_drawdown()
    assert abs(dd - 20.0) < 0.1  # 20%
    print(f"✅ T10: Drawdown = {dd:.1f}%")


def test_risk_dashboard():
    """T11: 리스크 대시보드."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal(p_kalshi=0.45)
    sizing = make_sizing(contracts=50)
    eng.process_entry(sig, sizing, match_id="M1")

    dash = eng.risk_dashboard()
    assert dash["mode"] == "PAPER"
    assert dash["open_positions"] == 1
    assert dash["total_trades"] == 1
    assert dash["bankroll"] < 10000
    print(f"✅ T11: Dashboard = {dash}")


def test_trade_log_fields():
    """T12: 거래 로그 필드 완전성."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)
    sig = make_signal(p_true=0.65, p_kalshi=0.45, ev=0.13)
    sizing = make_sizing(contracts=80, f_kelly=0.20, f_invest=0.05)

    rec = eng.process_entry(sig, sizing, match_id="LOG_M")

    assert rec.p_true == 0.65
    assert rec.p_kalshi == 0.45
    assert rec.ev_adj == 0.13
    assert rec.f_kelly == 0.20
    assert rec.f_invest == 0.05
    assert rec.match_id == "LOG_M"
    assert rec.timestamp != ""
    print("✅ T12: Trade log fields complete")


def test_position_averaging():
    """T13: 같은 ticker 추가 진입 → 평균 진입가."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)

    # 첫 진입: 50 contracts @ 40¢
    sig1 = make_signal(ticker="AVG-HW", p_kalshi=0.40)
    sizing1 = make_sizing(contracts=50)
    eng.process_entry(sig1, sizing1, match_id="AVG_M")

    # 추가 진입: 50 contracts @ 50¢
    sig2 = make_signal(ticker="AVG-HW", p_kalshi=0.50)
    sizing2 = make_sizing(contracts=50)
    eng.process_entry(sig2, sizing2, match_id="AVG_M")

    pos = eng.positions["AVG-HW"]
    assert pos.contracts == 100
    # avg = (50×0.40 + 50×0.50) / 100 = 0.45
    assert abs(pos.entry_price - 0.45) < 0.01
    print(f"✅ T13: Position averaging → {pos.contracts}@{pos.entry_price:.2f}")


def test_multiple_matches():
    """T14: 복수 경기 동시 관리."""
    eng = ExecutionEngine(initial_bankroll=10000, paper=True)

    # Match A
    sig_a = make_signal(ticker="MA-HW", p_kalshi=0.45)
    eng.process_entry(sig_a, make_sizing(contracts=30), match_id="MATCH_A")

    # Match B
    sig_b = make_signal(ticker="MB-HW", p_kalshi=0.50)
    eng.process_entry(sig_b, make_sizing(contracts=40), match_id="MATCH_B")

    assert len(eng.positions) == 2
    assert eng.get_match_exposure("MATCH_A") > 0
    assert eng.get_match_exposure("MATCH_B") > 0

    # Match A 정산
    eng.settle_match("MATCH_A", {"MA-HW": True})
    assert "MA-HW" not in eng.positions
    assert "MB-HW" in eng.positions  # B는 유지
    print(f"✅ T14: Multi-match → A settled, B alive")


if __name__ == "__main__":
    test_initial_state()
    test_paper_entry()
    test_paper_entry_cost()
    test_sizing_zero()
    test_paper_exit()
    test_exit_hold()
    test_match_exposure_tracking()
    test_settlement_yes_win()
    test_settlement_yes_lose()
    test_drawdown()
    test_risk_dashboard()
    test_trade_log_fields()
    test_position_averaging()
    test_multiple_matches()

    print(f"\n{'='*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'='*50}")

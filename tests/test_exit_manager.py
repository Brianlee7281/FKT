"""
test_exit_manager.py — ExitManager 단위 테스트.

3가지 청산 트리거 + 필터 + 일괄 평가 검증.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4.exit_manager import (
    ExitManager, ExitAction, ExitDecision, OpenPosition,
)
from src.phase4.edge_detector import Direction, EnginePhase


def test_edge_erosion_close():
    """T1: Edge 소멸 → CLOSE."""
    mgr = ExitManager(exit_threshold=0.005)  # 0.5¢
    pos = OpenPosition(
        ticker="TEST-HW", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.45,
    )
    # P_cons=0.46, ask=45¢ → EV = 0.46 - 0.45 - fee(0.45) = 0.01 - 0.01733 < 0
    d = mgr.evaluate_position(
        pos, p_true=0.46, sigma_mc=0.0,
        yes_bid_cents=43, yes_ask_cents=45, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.CLOSE
    assert "edge_erosion" in d.trigger
    print(f"✅ T1: Edge erosion → CLOSE ({d.trigger})")


def test_edge_erosion_hold():
    """T2: 충분한 Edge → HOLD."""
    mgr = ExitManager(exit_threshold=0.005)
    pos = OpenPosition(
        ticker="TEST-HW", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.45,
    )
    # P_cons=0.60, ask=45¢ → EV = 0.60 - 0.45 - 0.0173 = 0.1327 >> 0.5¢
    d = mgr.evaluate_position(
        pos, p_true=0.60, sigma_mc=0.0,
        yes_bid_cents=43, yes_ask_cents=45, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.HOLD
    print("✅ T2: Strong edge → HOLD")


def test_edge_reversal_buy_yes():
    """T3: BUY_YES 역전 → CLOSE."""
    mgr = ExitManager(entry_threshold=0.02)
    pos = OpenPosition(
        ticker="TEST-HW", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.55,
    )
    # P_cons=0.40, bid=50¢ → 0.40 < 0.50 - 0.02 = 0.48 → 역전!
    d = mgr.evaluate_position(
        pos, p_true=0.40, sigma_mc=0.0,
        yes_bid_cents=50, yes_ask_cents=55, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.CLOSE
    assert "edge_reversal" in d.trigger
    print(f"✅ T3: BUY_YES reversal → CLOSE")


def test_edge_reversal_buy_no():
    """T4: BUY_NO 역전 → CLOSE."""
    mgr = ExitManager(entry_threshold=0.02)
    pos = OpenPosition(
        ticker="TEST-HW", direction=Direction.BUY_NO,
        contracts=100, entry_price=0.60,
    )
    # BUY_NO 역전: P_cons > yes_ask + θ
    # yes_bid=30 → yes_ask = 70¢ = 0.70
    # P_cons=0.80 > 0.70 + 0.02 = 0.72 → 역전!
    d = mgr.evaluate_position(
        pos, p_true=0.80, sigma_mc=0.0,
        yes_bid_cents=30, yes_ask_cents=75, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.CLOSE
    assert "edge_reversal" in d.trigger
    print(f"✅ T4: BUY_NO reversal → CLOSE")


def test_no_reversal():
    """T5: 역전 미달 → HOLD (erosion도 안 걸림)."""
    mgr = ExitManager(entry_threshold=0.02, exit_threshold=0.005)
    pos = OpenPosition(
        ticker="TEST-HW", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.45,
    )
    # P_cons=0.55, bid=50¢ → 0.55 > 0.50 - 0.02 = 0.48 → 역전 아님
    # EV = 0.55 - 0.50 - fee(0.50) = 0.05 - 0.0175 = 0.0325 > 0.5¢
    d = mgr.evaluate_position(
        pos, p_true=0.55, sigma_mc=0.0,
        yes_bid_cents=50, yes_ask_cents=50, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.HOLD
    print("✅ T5: No reversal, healthy edge → HOLD")


def test_expiry_exit():
    """T6: 만기 평가 — 청산이 유리 (직접 메서드 테스트)."""
    mgr = ExitManager(expiry_minutes=3, match_duration=90)
    pos = OpenPosition(
        ticker="TEST2", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.80,
    )
    # P_cons=0.30, bid=35¢, minute=88
    # E[Hold] = 0.30×0.20 - 0.70×0.80 = 0.06 - 0.56 = -0.50
    # E[Exit] = (0.35-0.80) = -0.45 (no fee, loss)
    # E[Exit]=-0.45 > E[Hold]=-0.50 → 청산 유리!
    result = mgr.check_expiry_evaluation(pos, 0.30, 35, 88)
    assert result is not None
    assert "expiry_eval" in result
    print(f"✅ T6: Expiry evaluation → CLOSE (direct method)")


def test_expiry_hold():
    """T7: 만기 평가 — 보유가 유리."""
    mgr = ExitManager(expiry_minutes=3, match_duration=90)
    pos = OpenPosition(
        ticker="TEST-HW", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.40,
    )
    # P_cons=0.70, bid=60¢, minute=88
    # E[Hold] = 0.70×0.60 - 0.30×0.40 = 0.42 - 0.12 = 0.30
    # E[Exit] = (0.60-0.40) - fee = 0.20 - 0.07*0.6*0.4 = 0.20 - 0.0168 = 0.1832
    # E[Hold]=0.30 > E[Exit]=0.18 → HOLD
    d = mgr.evaluate_position(
        pos, p_true=0.70, sigma_mc=0.0,
        yes_bid_cents=60, yes_ask_cents=65,
        minute=88, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.HOLD
    print("✅ T7: Expiry hold (E[Hold] > E[Exit])")


def test_not_expiry_yet():
    """T8: 만기 평가 시점 전 → 트리거 안 됨."""
    mgr = ExitManager(expiry_minutes=3, match_duration=90)
    # minute=60, 종료 30분 전 → 만기 평가 안 함
    result = mgr.check_expiry_evaluation(
        OpenPosition(entry_price=0.80, direction=Direction.BUY_YES),
        p_true_cons=0.30, yes_bid_cents=35, minute=60,
    )
    assert result is None
    print("✅ T8: minute=60, not expiry time yet")


def test_halftime_no_exit():
    """T9: 하프타임 → HOLD (거래 불가)."""
    mgr = ExitManager()
    pos = OpenPosition(
        ticker="TEST", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.50,
    )
    d = mgr.evaluate_position(
        pos, p_true=0.20, sigma_mc=0.0,
        yes_bid_cents=50, yes_ask_cents=55,
        engine_phase=EnginePhase.HALFTIME, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.HOLD
    assert "halftime" in d.trigger
    print("✅ T9: HALFTIME → HOLD (no trading)")


def test_finished_hold_to_settlement():
    """T10: 경기 종료 → 정산까지 보유."""
    mgr = ExitManager()
    pos = OpenPosition(
        ticker="TEST", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.50,
    )
    d = mgr.evaluate_position(
        pos, p_true=0.20, sigma_mc=0.0,
        yes_bid_cents=10, yes_ask_cents=15,
        engine_phase=EnginePhase.FINISHED, z=0.0,
        current_tick=100,
    )
    assert d.action == ExitAction.HOLD
    assert "settlement" in d.trigger
    print("✅ T10: FINISHED → hold to settlement")


def test_pnl_estimate_profit():
    """T11: P&L 추정 — 이익."""
    mgr = ExitManager()
    pos = OpenPosition(
        ticker="TEST", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.40,
    )
    # exit at 60¢ → profit = 100 × (0.60-0.40) = $20
    # fee = 0.07 × 0.60 × 0.40 × 100 = $1.68
    # net = 20 - 1.68 = $18.32
    pnl = mgr._estimate_pnl(pos, 60)
    assert abs(pnl - 18.32) < 0.01
    print(f"✅ T11: P&L profit = ${pnl:.2f}")


def test_pnl_estimate_loss():
    """T12: P&L 추정 — 손실 (수수료 없음)."""
    mgr = ExitManager()
    pos = OpenPosition(
        ticker="TEST", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.50,
    )
    # exit at 30¢ → loss = 100 × (0.30-0.50) = -$20
    # no fee on loss
    pnl = mgr._estimate_pnl(pos, 30)
    assert abs(pnl - (-20.0)) < 0.01
    print(f"✅ T12: P&L loss = ${pnl:.2f}")


def test_conservative_p_applied():
    """T13: σ_MC > 0 → P_cons 적용."""
    mgr = ExitManager(exit_threshold=0.005, entry_threshold=0.02)
    pos = OpenPosition(
        ticker="TEST", direction=Direction.BUY_YES,
        contracts=100, entry_price=0.45,
    )
    # P_true=0.50, σ=0.02, z=1.645 → P_cons = 0.50 - 0.0329 = 0.4671
    # ask=45¢, EV = 0.4671 - 0.45 - fee(0.45) ≈ 0.017 - 0.0173 < 0.005
    d = mgr.evaluate_position(
        pos, p_true=0.50, sigma_mc=0.02,
        yes_bid_cents=43, yes_ask_cents=45,
        z=1.645,
        current_tick=100,
    )
    assert d.action == ExitAction.CLOSE
    assert "edge_erosion" in d.trigger
    print("✅ T13: σ_MC applied → P_cons lowers → CLOSE")


def test_evaluate_all():
    """T14: 일괄 평가."""
    mgr = ExitManager(exit_threshold=0.005, entry_threshold=0.02)
    positions = [
        OpenPosition(ticker="A", direction=Direction.BUY_YES,
                     contracts=50, entry_price=0.45),
        OpenPosition(ticker="B", direction=Direction.BUY_YES,
                     contracts=50, entry_price=0.30),
    ]
    market_data = {
        "A": {"p_true": 0.46, "sigma_mc": 0.0,
               "yes_bid_cents": 43, "yes_ask_cents": 45},
        "B": {"p_true": 0.70, "sigma_mc": 0.0,
               "yes_bid_cents": 60, "yes_ask_cents": 65},
    }
    decisions = mgr.evaluate_all(positions, market_data, z=0.0, current_tick=100)
    assert len(decisions) == 2
    # A: edge eroded (EV ≈ 0.46-0.45-fee < 0.5¢)
    assert decisions[0].action == ExitAction.CLOSE
    # B: strong edge (EV ≈ 0.70-0.65-fee >> 0.5¢)
    assert decisions[1].action == ExitAction.HOLD
    print(f"✅ T14: evaluate_all → A=CLOSE, B=HOLD")


if __name__ == "__main__":
    test_edge_erosion_close()
    test_edge_erosion_hold()
    test_edge_reversal_buy_yes()
    test_edge_reversal_buy_no()
    test_no_reversal()
    test_expiry_exit()
    test_expiry_hold()
    test_not_expiry_yet()
    test_halftime_no_exit()
    test_finished_hold_to_settlement()
    test_pnl_estimate_profit()
    test_pnl_estimate_loss()
    test_conservative_p_applied()
    test_evaluate_all()

    print(f"\n{'='*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'='*50}")
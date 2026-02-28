"""
test_edge_detector.py — EdgeDetector 단위 테스트.

수수료 공식, 보수적 확률, EV 계산, 필터 조건, 양방향 탐색 검증.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4.edge_detector import (
    EdgeDetector, EdgeSignal, MarketSnapshot,
    Direction, EnginePhase,
    TAKER_FEE_MULTIPLIER, MAKER_FEE_MULTIPLIER,
)


def test_conservative_p_basic():
    """T1: 보수적 확률 — 기본."""
    ed = EdgeDetector(z=1.645)
    p = ed.conservative_p(0.55, 0.02)
    assert abs(p - 0.5171) < 0.001
    print("✅ T1: P_cons = 0.5171 (z=1.645)")


def test_conservative_p_zero_sigma():
    """T2: σ_MC=0 (해석적 모드) → P_cons = P_true."""
    ed = EdgeDetector(z=1.645)
    p = ed.conservative_p(0.60, 0.0)
    assert abs(p - 0.60) < 0.001
    print("✅ T2: σ=0 → P_cons = P_true")


def test_conservative_p_clamp():
    """T3: 클램핑 — 범위 [0.001, 0.999]."""
    ed = EdgeDetector(z=2.0)
    p_low = ed.conservative_p(0.02, 0.05)
    assert p_low == 0.001
    p_high = ed.conservative_p(0.999, 0.0)
    assert p_high == 0.999
    print("✅ T3: 클램핑 [0.001, 0.999]")


def test_fee_taker():
    """T4: Taker 수수료 계산."""
    ed = EdgeDetector(fee_multiplier=TAKER_FEE_MULTIPLIER)
    fee_50 = ed.fee_per_contract(0.50)
    assert abs(fee_50 - 0.0175) < 0.0001
    fee_20 = ed.fee_per_contract(0.20)
    assert abs(fee_20 - 0.0112) < 0.0001
    fee_90 = ed.fee_per_contract(0.90)
    assert abs(fee_90 - 0.0063) < 0.0001
    print("✅ T4: Taker fee (50¢=1.75¢, 20¢=1.12¢, 90¢=0.63¢)")


def test_fee_maker():
    """T5: Maker 수수료 계산."""
    ed = EdgeDetector(fee_multiplier=MAKER_FEE_MULTIPLIER)
    fee = ed.fee_per_contract(0.50)
    assert abs(fee - 0.004375) < 0.0001
    print("✅ T5: Maker fee (50¢=0.44¢)")


def test_ev_buy_yes():
    """T6: Yes 매수 EV — 모델이 시장보다 높게 평가."""
    ed = EdgeDetector(z=0.0, fee_multiplier=TAKER_FEE_MULTIPLIER)
    p_cons = ed.conservative_p(0.60, 0.0)
    ev = ed.ev_buy_yes(p_cons, 0.45)
    expected = 0.60 - 0.45 - 0.07 * 0.45 * 0.55
    assert abs(ev - expected) < 0.001
    print(f"✅ T6: EV Buy Yes = {ev*100:.2f}¢ (P=60%, ask=45¢)")


def test_ev_buy_no():
    """T7: No 매수 EV — 모델이 시장보다 낮게 평가."""
    ed = EdgeDetector(z=0.0, fee_multiplier=TAKER_FEE_MULTIPLIER)
    p_cons = ed.conservative_p(0.30, 0.0)
    ev = ed.ev_buy_no(p_cons, 0.60)
    expected = 0.70 - 0.60 - 0.07 * 0.60 * 0.40
    assert abs(ev - expected) < 0.001
    print(f"✅ T7: EV Buy No = {ev*100:.2f}¢ (P=30%, no_ask=60¢)")


def test_ev_negative():
    """T8: 엣지 없음 — EV < 0."""
    ed = EdgeDetector(z=0.0, fee_multiplier=TAKER_FEE_MULTIPLIER)
    ev = ed.ev_buy_yes(0.45, 0.45)
    assert ev < 0
    print(f"✅ T8: EV 음수 = {ev*100:.2f}¢ (fair price)")


def test_evaluate_buy_yes_signal():
    """T9: BUY_YES 시그널 발행."""
    ed = EdgeDetector(z=0.0, entry_threshold=0.02, min_depth=10)
    snap = MarketSnapshot(
        ticker="KXEPLGAME-TEST-ARS", market_type="home_win",
        p_true=0.70, sigma_mc=0.0,
        yes_ask_cents=55, yes_bid_cents=50,
        yes_depth=100, no_depth=100,
    )
    sig = ed.evaluate_market(snap)
    assert sig.direction == Direction.BUY_YES
    assert sig.ev_adj > 0.10
    print(f"✅ T9: BUY_YES signal (EV={sig.ev_cents:.2f}¢)")


def test_evaluate_buy_no_signal():
    """T10: BUY_NO 시그널 발행."""
    ed = EdgeDetector(z=0.0, entry_threshold=0.02, min_depth=10)
    snap = MarketSnapshot(
        ticker="KXEPLGAME-TEST-ARS", market_type="home_win",
        p_true=0.25, sigma_mc=0.0,
        yes_ask_cents=40, yes_bid_cents=35,
        yes_depth=100, no_depth=100,
    )
    sig = ed.evaluate_market(snap)
    assert sig.direction == Direction.BUY_NO
    assert sig.ev_adj > 0.05
    print(f"✅ T10: BUY_NO signal (EV={sig.ev_cents:.2f}¢)")


def test_filter_cooldown():
    """T11: 쿨다운 → HOLD."""
    ed = EdgeDetector(z=0.0, entry_threshold=0.01, min_depth=5)
    snap = MarketSnapshot(
        ticker="TEST", p_true=0.90, yes_ask_cents=50,
        yes_bid_cents=45, yes_depth=100, no_depth=100,
    )
    sig = ed.evaluate_market(snap, cooldown=True)
    assert sig.direction == Direction.HOLD
    assert sig.reason == "cooldown"
    print("✅ T11: cooldown → HOLD")


def test_filter_engine_phase():
    """T12: 하프타임/종료 → HOLD."""
    ed = EdgeDetector(z=0.0, entry_threshold=0.01, min_depth=5)
    snap = MarketSnapshot(
        ticker="TEST", p_true=0.90, yes_ask_cents=50,
        yes_bid_cents=45, yes_depth=100, no_depth=100,
    )
    for phase in [EnginePhase.HALFTIME, EnginePhase.FINISHED]:
        sig = ed.evaluate_market(snap, engine_phase=phase)
        assert sig.direction == Direction.HOLD
        assert "engine_phase" in sig.reason
    print("✅ T12: HALFTIME/FINISHED → HOLD")


def test_filter_liquidity():
    """T13: 유동성 부족 → HOLD."""
    ed = EdgeDetector(z=0.0, entry_threshold=0.01, min_depth=20)
    snap = MarketSnapshot(
        ticker="TEST", p_true=0.90, yes_ask_cents=50,
        yes_bid_cents=45, yes_depth=5, no_depth=100,
    )
    sig = ed.evaluate_market(snap)
    assert sig.direction == Direction.HOLD
    assert "low_liquidity" in sig.reason
    print("✅ T13: low_liquidity → HOLD")


def test_scan_markets_multi():
    """T14: 복수 마켓 스캔 — EV 순 정렬."""
    ed = EdgeDetector(z=0.0, entry_threshold=0.02, min_depth=10)
    snapshots = [
        MarketSnapshot(
            ticker="MATCH-HW", market_type="home_win",
            p_true=0.60, yes_ask_cents=55, yes_bid_cents=50,
            yes_depth=100, no_depth=100,
        ),
        MarketSnapshot(
            ticker="MATCH-AW", market_type="away_win",
            p_true=0.80, yes_ask_cents=55, yes_bid_cents=50,
            yes_depth=100, no_depth=100,
        ),
        MarketSnapshot(
            ticker="MATCH-DR", market_type="draw",
            p_true=0.30, yes_ask_cents=30, yes_bid_cents=25,
            yes_depth=100, no_depth=100,
        ),
    ]
    signals = ed.scan_markets(snapshots)
    assert len(signals) >= 1
    assert signals[0].ticker == "MATCH-AW"
    assert signals[0].ev_adj > signals[-1].ev_adj if len(signals) > 1 else True
    print(f"✅ T14: scan_markets → {len(signals)} signals, top={signals[0].ticker}")


if __name__ == "__main__":
    test_conservative_p_basic()
    test_conservative_p_zero_sigma()
    test_conservative_p_clamp()
    test_fee_taker()
    test_fee_maker()
    test_ev_buy_yes()
    test_ev_buy_no()
    test_ev_negative()
    test_evaluate_buy_yes_signal()
    test_evaluate_buy_no_signal()
    test_filter_cooldown()
    test_filter_engine_phase()
    test_filter_liquidity()
    test_scan_markets_multi()

    print(f"\n{'='*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'='*50}")

"""
test_position_sizer.py — PositionSizer 단위 테스트.

Kelly 공식, Fractional Kelly, 3-Layer 리스크, 계약 수 변환, 엣지 케이스.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4.position_sizer import PositionSizer, SizingResult


def test_kelly_basic():
    """T1: Full Kelly — 기본."""
    ps = PositionSizer()
    # EV=0.05 (5¢), P_mkt=0.45
    # f* = 0.05 / (0.45 × 0.55) = 0.05 / 0.2475 = 0.2020
    f = ps.kelly_fraction_full(0.05, 0.45)
    assert abs(f - 0.2020) < 0.001
    print(f"✅ T1: Full Kelly = {f:.4f}")


def test_kelly_50_50():
    """T2: Kelly at 50¢ — P*(1-P) 최대."""
    ps = PositionSizer()
    # EV=0.03, P=0.50 → f* = 0.03 / 0.25 = 0.12
    f = ps.kelly_fraction_full(0.03, 0.50)
    assert abs(f - 0.12) < 0.001
    print(f"✅ T2: Kelly at 50¢ = {f:.4f}")


def test_kelly_extreme_price():
    """T3: Kelly at 극단 가격 — 비중 커짐."""
    ps = PositionSizer()
    # EV=0.03, P=0.10 → f* = 0.03 / (0.10 × 0.90) = 0.03/0.09 = 0.3333
    f = ps.kelly_fraction_full(0.03, 0.10)
    assert abs(f - 0.3333) < 0.001
    print(f"✅ T3: Kelly at 10¢ = {f:.4f} (높은 비중)")


def test_kelly_zero_ev():
    """T4: EV ≤ 0 → Kelly = 0."""
    ps = PositionSizer()
    assert ps.kelly_fraction_full(0.0, 0.50) == 0.0
    assert ps.kelly_fraction_full(-0.01, 0.50) == 0.0
    print("✅ T4: EV ≤ 0 → f* = 0")


def test_fractional_kelly():
    """T5: Quarter-Kelly 적용."""
    ps = PositionSizer(kelly_fraction=0.25)
    # f* = 0.2020, f_invest = 0.25 × 0.2020 = 0.0505
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000)
    assert abs(r.f_kelly - 0.2020) < 0.001
    assert abs(r.f_invest - 0.0505) < 0.001
    print(f"✅ T5: Quarter-Kelly f_invest = {r.f_invest:.4f}")


def test_contracts_calculation():
    """T6: 계약 수 변환."""
    ps = PositionSizer(kelly_fraction=0.25)
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000)
    # dollar = 0.0505 × 10000 = 505
    # but Layer 1: max_order = 0.03 × 10000 = 300 → clamped
    # contracts = floor(300 / 0.45) = 666
    assert r.contracts == 666
    assert r.clamped_by == "layer1"
    print(f"✅ T6: {r.contracts} contracts (Layer 1 clamped)")


def test_layer1_clamp():
    """T7: Layer 1 — 단일 주문 3% 한도."""
    ps = PositionSizer(kelly_fraction=1.0, order_cap=0.03)
    # f*=0.2020, f_invest=0.2020, dollar=2020 → L1=300 → clamp
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000)
    assert r.dollar_amount <= 10000 * 0.03 + 0.45  # 계약 단위 반올림 허용
    assert r.clamped_by == "layer1"
    print(f"✅ T7: Layer 1 clamp → ${r.dollar_amount:.2f}")


def test_layer2_clamp():
    """T8: Layer 2 — 경기별 5% 한도."""
    ps = PositionSizer(kelly_fraction=0.25, order_cap=0.10, match_cap=0.05)
    # order_cap 넉넉하게 설정, match_exposure=400 → remaining=100
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000,
                match_exposure=400)
    # remaining_match = 500 - 400 = 100
    # contracts = floor(100 / 0.45) = 222
    assert r.contracts == 222
    assert r.clamped_by == "layer2"
    print(f"✅ T8: Layer 2 clamp → {r.contracts} contracts")


def test_layer2_full():
    """T9: Layer 2 꽉 참 → 0 계약."""
    ps = PositionSizer(kelly_fraction=0.25, match_cap=0.05)
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000,
                match_exposure=500)  # 5% 한도 도달
    assert r.contracts == 0
    assert "layer2" in r.reason
    print("✅ T9: Layer 2 full → 0 contracts")


def test_layer3_clamp():
    """T10: Layer 3 — 전체 20% 한도."""
    ps = PositionSizer(kelly_fraction=0.25, order_cap=0.10,
                       match_cap=0.50, total_cap=0.20)
    # total_exposure=1900 → remaining=100
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000,
                total_exposure=1900)
    assert r.contracts == 222  # floor(100/0.45)
    assert r.clamped_by == "layer3"
    print(f"✅ T10: Layer 3 clamp → {r.contracts} contracts")


def test_layer3_full():
    """T11: Layer 3 꽉 참 → 0 계약."""
    ps = PositionSizer(total_cap=0.20)
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000,
                total_exposure=2000)
    assert r.contracts == 0
    assert "layer3" in r.reason
    print("✅ T11: Layer 3 full → 0 contracts")


def test_no_bankroll():
    """T12: 잔고 0 → 0 계약."""
    ps = PositionSizer()
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=0)
    assert r.contracts == 0
    assert r.reason == "no_bankroll"
    print("✅ T12: bankroll=0 → 0 contracts")


def test_half_kelly():
    """T13: Half-Kelly (K_frac=0.50)."""
    ps = PositionSizer(kelly_fraction=0.50)
    r = ps.size(ev_adj=0.05, p_kalshi=0.45, bankroll=10000)
    assert abs(r.f_invest - 0.1010) < 0.001
    print(f"✅ T13: Half-Kelly f_invest = {r.f_invest:.4f}")


def test_small_edge_min_contracts():
    """T14: 아주 작은 엣지 → 최소 계약 미달."""
    ps = PositionSizer(kelly_fraction=0.25, min_contracts=5)
    # 아주 작은 EV → 계약 수 < 5
    r = ps.size(ev_adj=0.001, p_kalshi=0.50, bankroll=100)
    # f* = 0.001/0.25=0.004, f_inv=0.001, dollar=0.10
    # contracts = floor(0.10/0.50) = 0
    assert r.contracts == 0
    print(f"✅ T14: tiny edge → below min_contracts ({r.reason})")


if __name__ == "__main__":
    test_kelly_basic()
    test_kelly_50_50()
    test_kelly_extreme_price()
    test_kelly_zero_ev()
    test_fractional_kelly()
    test_contracts_calculation()
    test_layer1_clamp()
    test_layer2_clamp()
    test_layer2_full()
    test_layer3_clamp()
    test_layer3_full()
    test_no_bankroll()
    test_half_kelly()
    test_small_edge_min_contracts()

    print(f"\n{'='*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'='*50}")

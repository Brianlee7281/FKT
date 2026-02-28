"""
test_kalshi_client.py — KalshiClient 단위 테스트.

실제 API 호출 없이 로직만 테스트.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4.kalshi_client import (
    KalshiClient, OrderBook, OrderBookLevel, OrderResponse, Position,
)


def test_orderbook_best_prices():
    """T1: 호가창 best bid/ask 계산."""
    ob = OrderBook(
        yes_bids=[
            OrderBookLevel(price_cents=42, quantity=100),
            OrderBookLevel(price_cents=40, quantity=200),
        ],
        no_bids=[
            OrderBookLevel(price_cents=56, quantity=150),
            OrderBookLevel(price_cents=54, quantity=300),
        ],
    )
    assert ob.best_yes_bid == 42
    assert ob.best_no_bid == 56
    assert ob.best_yes_ask == 44   # 100 - 56
    assert ob.best_no_ask == 58    # 100 - 42
    assert ob.spread == 2          # 44 - 42
    print("✅ T1: best bid/ask/spread 계산")


def test_orderbook_empty():
    """T2: 빈 호가창."""
    ob = OrderBook()
    assert ob.best_yes_bid is None
    assert ob.best_yes_ask is None
    assert ob.spread is None
    assert ob.yes_total_depth == 0
    assert ob.no_total_depth == 0
    print("✅ T2: 빈 호가창 None 처리")


def test_orderbook_depth():
    """T3: 총 물량 계산."""
    ob = OrderBook(
        yes_bids=[
            OrderBookLevel(price_cents=42, quantity=100),
            OrderBookLevel(price_cents=40, quantity=200),
        ],
        no_bids=[
            OrderBookLevel(price_cents=56, quantity=150),
        ],
    )
    assert ob.yes_total_depth == 300
    assert ob.no_total_depth == 150
    print("✅ T3: 총 물량(depth) 계산")


def test_vwap_yes_buy():
    """T4: Yes 매수 VWAP — No bids 소비."""
    ob = OrderBook(
        no_bids=[
            OrderBookLevel(price_cents=56, quantity=50),   # ask=44
            OrderBookLevel(price_cents=55, quantity=100),   # ask=45
            OrderBookLevel(price_cents=54, quantity=200),   # ask=46
        ],
    )
    # 80계약 매수: 50@44 + 30@45 = 2200+1350 = 3550 / 80 = 44.375
    vwap = ob.vwap_yes_buy(80)
    assert vwap is not None
    assert abs(vwap - 44.375) < 0.001
    print("✅ T4: VWAP Yes 매수 — 44.375¢")


def test_vwap_yes_buy_exact():
    """T5: Yes 매수 VWAP — 정확히 1레벨."""
    ob = OrderBook(
        no_bids=[
            OrderBookLevel(price_cents=56, quantity=100),  # ask=44
        ],
    )
    vwap = ob.vwap_yes_buy(50)
    assert vwap == 44.0
    print("✅ T5: VWAP Yes 매수 정확히 1레벨 — 44¢")


def test_vwap_yes_buy_insufficient():
    """T6: 물량 부족 시 None."""
    ob = OrderBook(
        no_bids=[
            OrderBookLevel(price_cents=56, quantity=10),
        ],
    )
    vwap = ob.vwap_yes_buy(50)
    assert vwap is None
    print("✅ T6: VWAP 물량 부족 → None")


def test_vwap_yes_sell():
    """T7: Yes 매도 VWAP — Yes bids 소비."""
    ob = OrderBook(
        yes_bids=[
            OrderBookLevel(price_cents=42, quantity=60),
            OrderBookLevel(price_cents=40, quantity=100),
        ],
    )
    # 80계약 매도: 60@42 + 20@40 = 2520+800 = 3320 / 80 = 41.5
    vwap = ob.vwap_yes_sell(80)
    assert vwap is not None
    assert abs(vwap - 41.5) < 0.001
    print("✅ T7: VWAP Yes 매도 — 41.5¢")


def test_liquidity_ok():
    """T8: 유동성 필터."""
    ob_good = OrderBook(
        yes_bids=[OrderBookLevel(price_cents=42, quantity=100)],
        no_bids=[OrderBookLevel(price_cents=56, quantity=100)],
    )
    assert ob_good.liquidity_ok(min_depth=20) is True

    ob_thin = OrderBook(
        yes_bids=[OrderBookLevel(price_cents=42, quantity=5)],
        no_bids=[OrderBookLevel(price_cents=56, quantity=100)],
    )
    assert ob_thin.liquidity_ok(min_depth=20) is False
    print("✅ T8: 유동성 필터")


def test_sign_method():
    """T9: RSA-PSS 서명 — 키 없이 구조만 확인."""
    client = KalshiClient(api_key="test-key", private_key_path="nonexistent.pem")
    # _sign은 키 로드 후에만 동작하므로, auth_headers 구조만 확인
    # 실제 서명은 통합 테스트에서
    assert client.api_key == "test-key"
    assert client.base_url == "https://api.elections.kalshi.com"
    print("✅ T9: 클라이언트 초기화 (프로덕션 URL)")


def test_demo_mode():
    """T10: 데모 모드 URL."""
    client = KalshiClient(api_key="test", private_key_path="test.pem", demo=True)
    assert client.base_url == "https://demo-api.kalshi.co"
    print("✅ T10: 데모 모드 URL")


def test_orderresponse_dataclass():
    """T11: OrderResponse 데이터 클래스."""
    resp = OrderResponse(
        order_id="abc123",
        ticker="MATCH-HW",
        side="yes",
        action="buy",
        price_cents=45,
        count=10,
        status="resting",
    )
    assert resp.order_id == "abc123"
    assert resp.price_cents == 45
    print("✅ T11: OrderResponse 데이터 클래스")


def test_position_dataclass():
    """T12: Position 데이터 클래스."""
    pos = Position(
        ticker="MATCH-HW",
        yes_count=50.0,
        yes_avg_price=44.0,
    )
    assert pos.ticker == "MATCH-HW"
    assert pos.yes_count == 50.0
    print("✅ T12: Position 데이터 클래스")


def test_vwap_edge_cases():
    """T13: VWAP 0 수량, 빈 호가."""
    ob = OrderBook()
    assert ob.vwap_yes_buy(10) is None
    assert ob.vwap_yes_sell(10) is None

    ob2 = OrderBook(
        no_bids=[OrderBookLevel(price_cents=56, quantity=100)],
    )
    # 0 수량 → None
    assert ob2.vwap_yes_buy(0) is None
    print("✅ T13: VWAP 엣지 케이스")


def test_one_sided_book():
    """T14: 한쪽만 있는 호가창."""
    ob = OrderBook(
        yes_bids=[OrderBookLevel(price_cents=42, quantity=100)],
        no_bids=[],
    )
    assert ob.best_yes_bid == 42
    assert ob.best_yes_ask is None  # No bid 없어서 ask 계산 불가
    assert ob.best_no_ask == 58     # 100 - 42
    assert ob.spread is None
    print("✅ T14: 한쪽만 있는 호가창")


if __name__ == "__main__":
    test_orderbook_best_prices()
    test_orderbook_empty()
    test_orderbook_depth()
    test_vwap_yes_buy()
    test_vwap_yes_buy_exact()
    test_vwap_yes_buy_insufficient()
    test_vwap_yes_sell()
    test_liquidity_ok()
    test_sign_method()
    test_demo_mode()
    test_orderresponse_dataclass()
    test_position_dataclass()
    test_vwap_edge_cases()
    test_one_sided_book()

    print(f"\n{'═'*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'═'*50}")
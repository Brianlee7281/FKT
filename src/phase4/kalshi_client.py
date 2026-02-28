"""
Phase 4 Step 4.1: Kalshi API 클라이언트 (kalshi_client.py)

RSA-PSS 서명 인증, 호가창 조회, 주문 제출/취소, 잔고/포지션 조회.

Kalshi 호가창 특성 (바이너리 마켓):
  - Yes bid / No bid만 반환 (ask 없음)
  - Yes ask = 100 - (Best No bid)  [센트 단위]
  - No ask  = 100 - (Best Yes bid)

사용법:
  client = KalshiClient(api_key="...", private_key_path="./kalshi.pem")
  # 또는 .env에서 자동 로드:
  client = KalshiClient()

  balance = await client.get_balance()
  ob = await client.get_orderbook("TICKER-HERE")
  order = await client.place_order(ticker="...", side="yes", action="buy", count=10, yes_price=45)
"""

from __future__ import annotations

import base64
import datetime
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════

KALSHI_PROD_URL = "https://api.elections.kalshi.com"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co"
API_PATH_PREFIX = "/trade-api/v2"


# ═══════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════

@dataclass
class OrderBookLevel:
    """호가 한 단계: 가격(센트) + 수량(계약)."""
    price_cents: int
    quantity: float  # fp이면 소수점 가능


@dataclass
class OrderBook:
    """
    Kalshi 호가창. 바이너리 마켓이므로:
      yes_bids: Yes 매수 호가 (높은 가격순)
      no_bids:  No 매수 호가 (높은 가격순)

    파생 값:
      best_yes_bid: Yes 최고 매수가 (센트)
      best_yes_ask: Yes 최저 매도가 = 100 - best_no_bid (센트)
      spread: best_yes_ask - best_yes_bid
    """
    yes_bids: List[OrderBookLevel] = field(default_factory=list)
    no_bids: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_yes_bid(self) -> Optional[int]:
        return self.yes_bids[0].price_cents if self.yes_bids else None

    @property
    def best_no_bid(self) -> Optional[int]:
        return self.no_bids[0].price_cents if self.no_bids else None

    @property
    def best_yes_ask(self) -> Optional[int]:
        """Yes ask = 100 - best No bid."""
        if self.best_no_bid is not None:
            return 100 - self.best_no_bid
        return None

    @property
    def best_no_ask(self) -> Optional[int]:
        """No ask = 100 - best Yes bid."""
        if self.best_yes_bid is not None:
            return 100 - self.best_yes_bid
        return None

    @property
    def spread(self) -> Optional[int]:
        """Bid-Ask 스프레드 (센트)."""
        if self.best_yes_bid is not None and self.best_yes_ask is not None:
            return self.best_yes_ask - self.best_yes_bid
        return None

    @property
    def yes_total_depth(self) -> float:
        """Yes bid 총 물량."""
        return sum(lvl.quantity for lvl in self.yes_bids)

    @property
    def no_total_depth(self) -> float:
        """No bid 총 물량."""
        return sum(lvl.quantity for lvl in self.no_bids)

    def vwap_yes_buy(self, quantity: float) -> Optional[float]:
        """
        Yes 매수 시 VWAP 실효 가격 (센트).
        No bids를 역순으로 소비하여 계산.
        (Yes를 사려면 No bid의 반대편을 가져가는 것)
        """
        if not self.no_bids or quantity <= 0:
            return None
        filled = 0.0
        cost = 0.0
        for lvl in self.no_bids:  # 높은 가격(=낮은 ask)부터
            ask_price = 100 - lvl.price_cents
            take = min(lvl.quantity, quantity - filled)
            cost += ask_price * take
            filled += take
            if filled >= quantity:
                break
        if filled < quantity:
            return None  # 물량 부족
        return cost / filled

    def vwap_yes_sell(self, quantity: float) -> Optional[float]:
        """
        Yes 매도 시 VWAP 실효 가격 (센트).
        Yes bids를 소비하여 계산.
        """
        if not self.yes_bids or quantity <= 0:
            return None
        filled = 0.0
        cost = 0.0
        for lvl in self.yes_bids:  # 높은 가격부터
            take = min(lvl.quantity, quantity - filled)
            cost += lvl.price_cents * take
            filled += take
            if filled >= quantity:
                break
        if filled < quantity:
            return None  # 물량 부족
        return cost / filled

    def liquidity_ok(self, min_depth: float = 20.0) -> bool:
        """유동성 필터: 양쪽 depth가 최소 기준 이상인지."""
        return self.yes_total_depth >= min_depth and self.no_total_depth >= min_depth


@dataclass
class OrderResponse:
    """주문 제출/취소 응답."""
    order_id: str = ""
    ticker: str = ""
    side: str = ""       # "yes" / "no"
    action: str = ""     # "buy" / "sell"
    price_cents: int = 0
    count: int = 0
    status: str = ""     # "resting", "filled", "canceled" 등
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """보유 포지션."""
    ticker: str = ""
    yes_count: float = 0.0
    no_count: float = 0.0
    yes_avg_price: float = 0.0
    no_avg_price: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════
# Kalshi 클라이언트
# ═══════════════════════════════════════════════════

class KalshiClient:
    """
    Kalshi REST API 클라이언트.

    인증: RSA-PSS 서명 (매 요청마다)
    호가: Yes/No bid만 반환 → ask는 100 - bid으로 계산

    Args:
        api_key:          Kalshi API 키 ID
        private_key_path: RSA private key (.pem) 파일 경로
        demo:             True이면 데모 환경 사용
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        demo: bool = False,
    ):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        self.private_key_path = private_key_path or os.getenv(
            "KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem"
        )
        self.base_url = KALSHI_DEMO_URL if demo else KALSHI_PROD_URL
        self._private_key = None
        self._client = None  # httpx.AsyncClient when connected

    # ─── 초기화 / 정리 ────────────────────────────

    async def connect(self) -> None:
        """HTTP 클라이언트 초기화 + 키 로드."""
        if not HAS_HTTPX:
            raise ImportError("httpx 필요: pip install httpx")
        self._load_private_key()
        self._client = httpx.AsyncClient(timeout=30.0)

        # 연결 검증: 잔고 조회
        try:
            balance = await self.get_balance()
            logger.info(f"KalshiClient: 연결 성공 (잔고: ${balance / 100:.2f})")
        except Exception as e:
            logger.warning(f"KalshiClient: 연결 검증 실패 — {type(e).__name__}: {e}")

    async def disconnect(self) -> None:
        """HTTP 클라이언트 정리."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    # ─── RSA-PSS 서명 ─────────────────────────────

    def _load_private_key(self) -> None:
        """PEM 파일에서 RSA 키 로드."""
        with open(self.private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        """
        RSA-PSS 서명 생성.

        message = timestamp + METHOD + path (쿼리 파라미터 제외)
        """
        path_without_query = path.split("?")[0]
        message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """인증 헤더 생성."""
        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        signature = self._sign(timestamp, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    # ─── HTTP 요청 ─────────────────────────────────

    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """인증된 GET 요청."""
        full_path = f"{API_PATH_PREFIX}{path}"
        headers = self._auth_headers("GET", full_path)
        url = f"{self.base_url}{full_path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: Dict) -> Dict:
        """인증된 POST 요청."""
        full_path = f"{API_PATH_PREFIX}{path}"
        headers = self._auth_headers("POST", full_path)
        url = f"{self.base_url}{full_path}"
        resp = await self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()

    async def _delete(self, path: str) -> Dict:
        """인증된 DELETE 요청."""
        full_path = f"{API_PATH_PREFIX}{path}"
        headers = self._auth_headers("DELETE", full_path)
        url = f"{self.base_url}{full_path}"
        resp = await self._client.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ─── 계좌 ─────────────────────────────────────

    async def get_balance(self) -> int:
        """계좌 잔고 조회 (센트 단위)."""
        data = await self._get("/portfolio/balance")
        return data.get("balance", 0)

    async def get_positions(self) -> List[Position]:
        """현재 보유 포지션 목록."""
        data = await self._get("/portfolio/positions")
        positions = []
        for p in data.get("market_positions", []):
            positions.append(Position(
                ticker=p.get("ticker", ""),
                yes_count=float(p.get("position", 0)),
                no_count=float(p.get("total_traded", 0)),
                raw=p,
            ))
        return positions

    # ─── 마켓 & 호가창 ─────────────────────────────

    async def get_market(self, ticker: str) -> Dict:
        """특정 마켓 정보 조회."""
        data = await self._get(f"/markets/{ticker}")
        return data.get("market", {})

    async def search_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
    ) -> List[Dict]:
        """마켓 검색."""
        params = {"status": status, "limit": str(limit)}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = await self._get("/markets", params=params)
        return data.get("markets", [])

    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """
        호가창 조회.

        Kalshi는 bid만 반환:
          yes[]: [[price, qty], ...] — 낮은→높은 가격 순
          no[]:  [[price, qty], ...]

        Yes 매수가(ask) = 100 - Best No bid
        """
        data = await self._get(f"/markets/{ticker}/orderbook", {"depth": str(depth)})
        ob_data = data.get("orderbook", {})

        # Yes bids (높은 가격순으로 정렬)
        yes_raw = ob_data.get("yes", [])
        yes_bids = [
            OrderBookLevel(price_cents=int(lvl[0]), quantity=float(lvl[1]))
            for lvl in yes_raw if len(lvl) >= 2
        ]
        yes_bids.sort(key=lambda x: x.price_cents, reverse=True)

        # No bids (높은 가격순으로 정렬)
        no_raw = ob_data.get("no", [])
        no_bids = [
            OrderBookLevel(price_cents=int(lvl[0]), quantity=float(lvl[1]))
            for lvl in no_raw if len(lvl) >= 2
        ]
        no_bids.sort(key=lambda x: x.price_cents, reverse=True)

        return OrderBook(yes_bids=yes_bids, no_bids=no_bids)

    async def get_orderbook_snapshot(self, ticker: str) -> Dict[str, Optional[int]]:
        """
        호가창 요약: best bid/ask + 스프레드.

        Returns:
            {
                "yes_bid": 42,
                "yes_ask": 44,
                "spread": 2,
                "yes_depth": 150.0,
                "no_depth": 200.0,
            }
        """
        ob = await self.get_orderbook(ticker)
        return {
            "yes_bid": ob.best_yes_bid,
            "yes_ask": ob.best_yes_ask,
            "spread": ob.spread,
            "yes_depth": ob.yes_total_depth,
            "no_depth": ob.no_total_depth,
        }

    # ─── 주문 ─────────────────────────────────────

    async def place_order(
        self,
        ticker: str,
        side: str,         # "yes" / "no"
        action: str,       # "buy" / "sell"
        count: int = 1,
        yes_price: Optional[int] = None,   # 센트
        no_price: Optional[int] = None,    # 센트
        client_order_id: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> OrderResponse:
        """
        주문 제출.

        Args:
            ticker:   마켓 티커
            side:     "yes" 또는 "no"
            action:   "buy" 또는 "sell"
            count:    계약 수
            yes_price: Yes 가격 (센트, 1~99)
            no_price:  No 가격 (센트, 1~99)
            client_order_id: 클라이언트 고유 주문 ID
            expiration_ts: 만료 타임스탬프 (밀리초)

        Returns:
            OrderResponse
        """
        body: Dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        if expiration_ts:
            body["expiration_ts"] = expiration_ts

        data = await self._post("/portfolio/orders", body)
        order = data.get("order", {})

        return OrderResponse(
            order_id=order.get("order_id", ""),
            ticker=order.get("ticker", ticker),
            side=order.get("side", side),
            action=order.get("action", action),
            price_cents=order.get("yes_price", 0) or order.get("no_price", 0),
            count=order.get("remaining_count", count),
            status=order.get("status", ""),
            raw=order,
        )

    async def cancel_order(self, order_id: str) -> Dict:
        """주문 취소."""
        return await self._delete(f"/portfolio/orders/{order_id}")

    async def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """주문 목록 조회."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        data = await self._get("/portfolio/orders", params if params else None)
        return data.get("orders", [])

    # ─── 편의 메서드 ──────────────────────────────

    async def buy_yes(
        self, ticker: str, count: int, price_cents: int, client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """Yes 매수 (Limit)."""
        return await self.place_order(
            ticker=ticker, side="yes", action="buy",
            count=count, yes_price=price_cents,
            client_order_id=client_order_id,
        )

    async def buy_no(
        self, ticker: str, count: int, price_cents: int, client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """No 매수 (Limit)."""
        return await self.place_order(
            ticker=ticker, side="no", action="buy",
            count=count, no_price=price_cents,
            client_order_id=client_order_id,
        )

    async def sell_yes(
        self, ticker: str, count: int, price_cents: int, client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """Yes 매도 (Limit)."""
        return await self.place_order(
            ticker=ticker, side="yes", action="sell",
            count=count, yes_price=price_cents,
            client_order_id=client_order_id,
        )

    async def sell_no(
        self, ticker: str, count: int, price_cents: int, client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """No 매도 (Limit)."""
        return await self.place_order(
            ticker=ticker, side="no", action="sell",
            count=count, no_price=price_cents,
            client_order_id=client_order_id,
        )

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """특정 마켓(또는 전체)의 미체결 주문 모두 취소. 취소 건수 반환."""
        orders = await self.get_orders(ticker=ticker, status="resting")
        count = 0
        for order in orders:
            try:
                await self.cancel_order(order["order_id"])
                count += 1
            except Exception as e:
                logger.warning(f"주문 취소 실패 {order.get('order_id')}: {e}")
        return count
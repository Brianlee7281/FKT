"""
test_orchestrator.py — 통합 오케스트레이터 테스트.

TickerMapper, TickRouter, MatchSession의 통합 동작 검증.
Phase 3 TickSnapshot을 모킹하여 전체 파이프라인 테스트.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional

# Phase 4 모듈
from src.phase4.edge_detector import EdgeDetector, Direction, EnginePhase
from src.phase4.position_sizer import PositionSizer
from src.phase4.exit_manager import ExitManager
from src.phase4.execution_engine import ExecutionEngine
from src.phase4.post_match import PostMatchAnalyzer

# 오케스트레이터 모듈
from src.orchestrator.ticker_mapper import TickerMapper, LEAGUE_SERIES
from src.orchestrator.tick_router import TickRouter, OrderbookSnapshot, TickResult
from src.orchestrator.match_session import MatchSession, SessionState


# ═══════════════════════════════════════════════════
# Mock Phase 3 TickSnapshot & PricingResult
# ═══════════════════════════════════════════════════

@dataclass
class MockPricingResult:
    home_win: float = 0.50
    draw: float = 0.25
    away_win: float = 0.25
    sigma_mc: float = 0.0
    mode: str = "analytical"


@dataclass
class MockTickSnapshot:
    tick: int = 1
    minute: float = 30.0
    S_H: int = 0
    S_A: int = 0
    engine_phase: str = "FIRST_HALF"
    orders_allowed: bool = True
    pricing: Optional[MockPricingResult] = None


# ═══════════════════════════════════════════════════
# Helper: 표준 컴포넌트 생성
# ═══════════════════════════════════════════════════

def make_components(bankroll=10000):
    """Phase 4 컴포넌트 세트 생성."""
    engine = ExecutionEngine(initial_bankroll=bankroll, paper=True)
    detector = EdgeDetector(entry_threshold=0.02)
    sizer = PositionSizer(kelly_fraction=0.25)
    exit_mgr = ExitManager()
    return engine, detector, sizer, exit_mgr


def make_router(bankroll=10000):
    """TickRouter + 컴포넌트 생성."""
    engine, detector, sizer, exit_mgr = make_components(bankroll)
    router = TickRouter(engine, detector, sizer, exit_mgr)
    return router


SAMPLE_TICKERS = {
    "home_win": "KXEPLGAME-26MAR15ARSEVE-ARS",
    "away_win": "KXEPLGAME-26MAR15ARSEVE-EVE",
    "draw":     "KXEPLGAME-26MAR15ARSEVE-DRW",
}


def make_orderbooks(home_ask=45, home_bid=42,
                    away_ask=30, away_bid=27,
                    draw_ask=28, draw_bid=25):
    """3개 시장 호가창 생성."""
    return {
        "home_win": OrderbookSnapshot(
            ticker=SAMPLE_TICKERS["home_win"],
            yes_ask_cents=home_ask, yes_bid_cents=home_bid,
            yes_depth=100, no_depth=100,
        ),
        "away_win": OrderbookSnapshot(
            ticker=SAMPLE_TICKERS["away_win"],
            yes_ask_cents=away_ask, yes_bid_cents=away_bid,
            yes_depth=100, no_depth=100,
        ),
        "draw": OrderbookSnapshot(
            ticker=SAMPLE_TICKERS["draw"],
            yes_ask_cents=draw_ask, yes_bid_cents=draw_bid,
            yes_depth=100, no_depth=100,
        ),
    }


# ═══════════════════════════════════════════════════
# TickerMapper 테스트
# ═══════════════════════════════════════════════════

def test_mapper_register():
    """T1: 수동 매핑 등록."""
    mapper = TickerMapper()
    m = mapper.register_match(
        match_id="GS_001", home_team="Arsenal", away_team="Everton",
        kalshi_tickers=SAMPLE_TICKERS, league="epl",
    )
    assert mapper.has_match("GS_001")
    assert m.home_team == "Arsenal"
    tickers = mapper.get_kalshi_tickers("GS_001")
    assert "home_win" in tickers
    print("✅ T1: Mapper register OK")


def test_mapper_config_load():
    """T2: JSON config 일괄 로드."""
    mapper = TickerMapper()
    config = [
        {"match_id": "GS_A", "home_team": "Real Madrid", "away_team": "Barcelona",
         "kalshi_tickers": {"home_win": "KX-RM", "away_win": "KX-BAR", "draw": "KX-DRW"}},
        {"match_id": "GS_B", "home_team": "Milan", "away_team": "Inter",
         "kalshi_tickers": {"home_win": "KX-MIL"}},
    ]
    count = mapper.load_from_config(config)
    assert count == 2
    assert mapper.has_match("GS_A")
    assert mapper.has_match("GS_B")
    print(f"✅ T2: Config load → {count} matches")


def test_mapper_league_series():
    """T3: 리그 시리즈 매핑 확인."""
    assert LEAGUE_SERIES["epl"] == "KXEPLGAME"
    assert LEAGUE_SERIES["ucl"] == "KXUCLGAME"
    assert LEAGUE_SERIES["liga_mx"] == "KXLIGAMXGAME"
    print(f"✅ T3: {len(LEAGUE_SERIES)} league series mapped")


# ═══════════════════════════════════════════════════
# TickRouter 테스트
# ═══════════════════════════════════════════════════

def test_router_no_edge():
    """T4: Edge 없음 → 진입 안 함."""
    router = make_router()
    snap = MockTickSnapshot(
        pricing=MockPricingResult(home_win=0.45, away_win=0.25, draw=0.30),
    )
    # home_win=0.45, ask=45¢ → EV ≈ 0 (모델과 시장 일치)
    result = router.on_tick(snap, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(result.entries) == 0
    assert result.signals_evaluated > 0
    print(f"✅ T4: No edge → 0 entries (evaluated {result.signals_evaluated})")


def test_router_entry_signal():
    """T5: 확실한 Edge → 진입."""
    router = make_router()
    snap = MockTickSnapshot(
        pricing=MockPricingResult(home_win=0.70),  # 모델 70% vs 시장 45%
    )
    result = router.on_tick(snap, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(result.entries) > 0
    entry_ticker = result.entries[0].ticker
    assert "ARS" in entry_ticker  # 홈승 마켓
    print(f"✅ T5: Strong edge → entry on {entry_ticker}")


def test_router_halftime_no_trade():
    """T6: 하프타임 → 진입 안 함."""
    router = make_router()
    snap = MockTickSnapshot(
        engine_phase="HALFTIME", orders_allowed=False,
        pricing=MockPricingResult(home_win=0.80),
    )
    result = router.on_tick(snap, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(result.entries) == 0
    print("✅ T6: Halftime → 0 entries")


def test_router_exit_trigger():
    """T7: 포지션 보유 중 Edge 소멸 → 청산."""
    router = make_router()

    # 먼저 진입
    snap1 = MockTickSnapshot(
        pricing=MockPricingResult(home_win=0.70),
    )
    router.on_tick(snap1, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(router.engine.positions) > 0

    # min_hold_ticks 건너뛰기 (테스트용)
    router._tick_count = 200

    # 다음 틱: 모델 확률 급락 → edge 소멸/역전
    snap2 = MockTickSnapshot(
        tick=2, minute=31,
        pricing=MockPricingResult(home_win=0.30),  # 급락!
    )
    result = router.on_tick(snap2, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(result.exits) > 0
    print(f"✅ T7: Edge reversal → exit ({result.exits[0].notes})")


def test_router_no_duplicate_entry():
    """T8: 같은 티커 중복 진입 방지."""
    router = make_router()
    snap = MockTickSnapshot(
        pricing=MockPricingResult(home_win=0.70),
    )
    # 첫 진입
    r1 = router.on_tick(snap, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(r1.entries) > 0

    # 두 번째 틱 — 같은 조건이지만 이미 포지션 있음
    r2 = router.on_tick(snap, make_orderbooks(), SAMPLE_TICKERS, "M1")
    assert len(r2.entries) == 0
    print("✅ T8: No duplicate entry")


# ═══════════════════════════════════════════════════
# MatchSession 테스트
# ═══════════════════════════════════════════════════

def test_session_lifecycle():
    """T9: 매치 세션 라이프사이클."""
    router = make_router()
    session = MatchSession(
        match_id="GS_001",
        kalshi_tickers=SAMPLE_TICKERS,
        tick_router=router,
    )
    # 호가창 설정
    for mt, ob in make_orderbooks().items():
        session.update_orderbook(mt, ob)

    assert session.state == SessionState.WAITING

    # 전반전 틱
    snap = MockTickSnapshot(pricing=MockPricingResult(home_win=0.50))
    session.on_tick(snap)
    assert session.state == SessionState.LIVE

    # 경기 종료
    snap_fin = MockTickSnapshot(engine_phase="FINISHED", orders_allowed=False,
                                 pricing=MockPricingResult(home_win=0.90))
    session.on_tick(snap_fin)
    assert session.state == SessionState.FINISHED

    print(f"✅ T9: Session lifecycle: WAITING→LIVE→FINISHED")


def test_session_entry_via_tick():
    """T10: 세션 통해 진입."""
    router = make_router()
    session = MatchSession("GS_002", SAMPLE_TICKERS, router)
    for mt, ob in make_orderbooks().items():
        session.update_orderbook(mt, ob)

    snap = MockTickSnapshot(
        pricing=MockPricingResult(home_win=0.70),
    )
    result = session.on_tick(snap)
    assert result is not None
    assert session.stats.total_entries > 0
    print(f"✅ T10: Session entry → {session.stats.total_entries} entries")


def test_session_settlement():
    """T11: 세션 정산."""
    router = make_router()
    session = MatchSession("GS_003", SAMPLE_TICKERS, router)
    for mt, ob in make_orderbooks().items():
        session.update_orderbook(mt, ob)

    # 진입
    snap = MockTickSnapshot(pricing=MockPricingResult(home_win=0.70))
    session.on_tick(snap)

    # 경기 종료
    snap_fin = MockTickSnapshot(engine_phase="FINISHED", orders_allowed=False,
                                 pricing=MockPricingResult())
    session.on_tick(snap_fin)

    # 정산 (홈팀 승리)
    outcomes = {SAMPLE_TICKERS["home_win"]: True,
                SAMPLE_TICKERS["away_win"]: False,
                SAMPLE_TICKERS["draw"]: False}
    analyzer = PostMatchAnalyzer()
    report = session.finalize(outcomes, analyzer)

    assert session.state == SessionState.SETTLED
    assert report is not None
    print(f"✅ T11: Settlement → P&L=${report.total_pnl:+.2f}")


def test_session_summary():
    """T12: 세션 요약."""
    router = make_router()
    session = MatchSession("GS_004", SAMPLE_TICKERS, router)
    for mt, ob in make_orderbooks().items():
        session.update_orderbook(mt, ob)

    for i in range(5):
        snap = MockTickSnapshot(tick=i, minute=30+i,
                                 pricing=MockPricingResult(home_win=0.50))
        session.on_tick(snap)

    s = session.summary()
    assert s["ticks"] == 5
    assert s["state"] == SessionState.LIVE
    print(f"✅ T12: Summary → {s}")


def test_full_pipeline_multi_tick():
    """T13: 전체 파이프라인 — 멀티 틱 시뮬레이션."""
    router = make_router(bankroll=5000)
    session = MatchSession("GS_FULL", SAMPLE_TICKERS, router)
    for mt, ob in make_orderbooks().items():
        session.update_orderbook(mt, ob)

    # 전반전: 점진적 edge 발생
    for i in range(10):
        p_home = 0.50 + i * 0.02  # 점점 상승
        snap = MockTickSnapshot(
            tick=i, minute=10 + i,
            pricing=MockPricingResult(home_win=p_home),
        )
        session.on_tick(snap)

    # 진입이 발생했는지 확인
    entries = session.stats.total_entries
    assert entries > 0

    # 하프타임
    snap_ht = MockTickSnapshot(engine_phase="HALFTIME", orders_allowed=False,
                                pricing=MockPricingResult(home_win=0.65))
    session.on_tick(snap_ht)

    # 후반전: edge 유지
    for i in range(5):
        snap = MockTickSnapshot(
            tick=20+i, minute=50+i, engine_phase="SECOND_HALF",
            pricing=MockPricingResult(home_win=0.65),
        )
        session.on_tick(snap)

    # 경기 종료
    snap_fin = MockTickSnapshot(engine_phase="FINISHED", orders_allowed=False,
                                 pricing=MockPricingResult(home_win=0.90))
    session.on_tick(snap_fin)

    # 정산
    outcomes = {SAMPLE_TICKERS["home_win"]: True,
                SAMPLE_TICKERS["away_win"]: False,
                SAMPLE_TICKERS["draw"]: False}
    analyzer = PostMatchAnalyzer()
    report = session.finalize(outcomes, analyzer)

    s = session.summary()
    print(f"✅ T13: Full pipeline → {s['ticks']} ticks, "
          f"{s['entries']} entries, P&L=${report.total_pnl:+.2f}")


def test_dashboard_after_session():
    """T14: 세션 후 건강 대시보드."""
    router = make_router(bankroll=5000)
    session = MatchSession("GS_DASH", SAMPLE_TICKERS, router)
    for mt, ob in make_orderbooks().items():
        session.update_orderbook(mt, ob)

    # 거래 발생시킴
    snap = MockTickSnapshot(pricing=MockPricingResult(home_win=0.70))
    session.on_tick(snap)

    # 정산
    snap_fin = MockTickSnapshot(engine_phase="FINISHED", orders_allowed=False,
                                 pricing=MockPricingResult())
    session.on_tick(snap_fin)
    outcomes = {SAMPLE_TICKERS["home_win"]: True,
                SAMPLE_TICKERS["away_win"]: False,
                SAMPLE_TICKERS["draw"]: False}
    analyzer = PostMatchAnalyzer()
    session.finalize(outcomes, analyzer)

    # 대시보드
    dash = analyzer.health_dashboard(
        current_bankroll=router.engine.bankroll,
        peak_bankroll=router.engine.portfolio.peak_bankroll,
    )
    output = PostMatchAnalyzer.format_dashboard(dash)
    assert "MODEL HEALTH" in output
    print(f"✅ T14: Dashboard after session\n{output}")


if __name__ == "__main__":
    test_mapper_register()
    test_mapper_config_load()
    test_mapper_league_series()
    test_router_no_edge()
    test_router_entry_signal()
    test_router_halftime_no_trade()
    test_router_exit_trigger()
    test_router_no_duplicate_entry()
    test_session_lifecycle()
    test_session_entry_via_tick()
    test_session_settlement()
    test_session_summary()
    test_full_pipeline_multi_tick()
    test_dashboard_after_session()

    print(f"\n{'='*50}")
    print(f"  ALL 14 TESTS PASSED ✅")
    print(f"{'='*50}")
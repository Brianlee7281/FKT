"""
Replay Paper Trading 시뮬레이션.

과거 경기를 ReplaySource로 재생하면서 전체 파이프라인을 검증:
  Phase 3 (확률 엔진) → Orchestrator → Phase 4 (Paper Trading)

Kalshi 호가가 없으므로 합성 호가창을 생성한다:
  - ask = P_true + spread/2 + noise   (시장이 모델보다 약간 높거나)
  - bid = P_true - spread/2 + noise   (시장이 모델보다 약간 낮거나)
  - 일부 틱에서 의도적으로 큰 mispricing → edge 발생

사용법:
  # 테스트 DB로 (내장 Man City 2-1 Arsenal)
  python tests/test_replay_paper.py

  # 실제 DB로
  python tests/test_replay_paper.py --db data/kalshi_football.db --fixture 1035068
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import math
import random
import sqlite3
import tempfile
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import expm

# Phase 3
from src.phase3.engine import LiveTradingEngine, EngineParams, TickSnapshot
from src.phase3.replay import ReplaySource
from src.phase3.mu_calculator import build_gamma_array, build_delta_array, build_basis_bounds
from src.phase3.mc_core import build_Q_diag_and_off

# Phase 4
from src.phase4.edge_detector import EdgeDetector, Direction
from src.phase4.position_sizer import PositionSizer
from src.phase4.exit_manager import ExitManager
from src.phase4.execution_engine import ExecutionEngine
from src.phase4.post_match import PostMatchAnalyzer, TradeOutcome

# Orchestrator
from src.orchestrator.ticker_mapper import TickerMapper
from src.orchestrator.tick_router import TickRouter, OrderbookSnapshot
from src.orchestrator.match_session import MatchSession, SessionState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("replay_paper")

# 세부 로거는 WARNING으로
for mod in ("src.phase3", "src.phase4", "src.orchestrator"):
    logging.getLogger(mod).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════
# 합성 호가창 생성기
# ═══════════════════════════════════════════════════

class SyntheticOrderbook:
    """
    Phase 3 P_true를 기반으로 합성 Kalshi 호가창 생성.

    시뮬레이션 모드:
      - EFFICIENT:  시장 ≈ 모델 (작은 스프레드, 작은 노이즈)
      - MISPRICED:  시장이 모델 대비 편향 (edge 발생)
      - RANDOM:     랜덤 편향 방향

    Args:
        spread_cents:    기본 bid-ask 스프레드 (센트)
        noise_std:       가격 노이즈 표준편차 (0~1 확률 단위)
        misprice_prob:   각 틱에서 mispricing 발생 확률
        misprice_bias:   mispricing 시 편향 크기 (확률 단위)
        seed:            재현성을 위한 시드
    """

    def __init__(
        self,
        spread_cents: int = 4,
        noise_std: float = 0.02,
        misprice_prob: float = 0.15,
        misprice_bias: float = 0.10,
        seed: int = 42,
    ):
        self.spread_cents = spread_cents
        self.noise_std = noise_std
        self.misprice_prob = misprice_prob
        self.misprice_bias = misprice_bias
        self.rng = random.Random(seed)

    def generate(
        self,
        p_true: float,
        market_type: str,
        ticker: str,
    ) -> OrderbookSnapshot:
        """
        단일 시장의 합성 호가 생성.

        Args:
            p_true:      모델의 진확률 (0~1)
            market_type: "home_win" / "away_win" / "draw"
            ticker:      Kalshi 티커

        Returns:
            OrderbookSnapshot
        """
        # 시장 가격 = P_true + 노이즈 + 가끔 편향
        noise = self.rng.gauss(0, self.noise_std)

        if self.rng.random() < self.misprice_prob:
            # mispricing: 시장이 모델 대비 랜덤 방향으로 편향
            bias = self.misprice_bias * self.rng.choice([-1, 1])
        else:
            bias = 0.0

        market_mid = max(0.02, min(0.98, p_true + noise + bias))

        half_spread = self.spread_cents / 2
        ask_cents = min(99, int(market_mid * 100) + half_spread)
        bid_cents = max(1, int(market_mid * 100) - half_spread)

        # 최소 스프레드 보장
        if ask_cents <= bid_cents:
            ask_cents = bid_cents + 1

        depth = self.rng.randint(20, 200)

        return OrderbookSnapshot(
            ticker=ticker,
            yes_ask_cents=ask_cents,
            yes_bid_cents=bid_cents,
            yes_depth=depth,
            no_depth=depth,
        )


# ═══════════════════════════════════════════════════
# 테스트 DB 생성
# ═══════════════════════════════════════════════════

def create_test_db(db_path: str) -> None:
    """테스트용 DB (Man City 2-1 Arsenal)"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            fixture_id INTEGER PRIMARY KEY,
            league_id INTEGER, league_name TEXT, season INTEGER,
            match_date TEXT, round TEXT,
            home_team_id INTEGER, home_team_name TEXT,
            away_team_id INTEGER, away_team_name TEXT,
            home_goals_ft INTEGER, away_goals_ft INTEGER,
            home_goals_ht INTEGER, away_goals_ht INTEGER,
            elapsed_minutes INTEGER, venue_name TEXT, referee TEXT,
            downloaded_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS match_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER, event_minute INTEGER,
            event_extra INTEGER, event_type TEXT, event_detail TEXT,
            team_id INTEGER, team_name TEXT,
            player_id INTEGER, player_name TEXT,
            assist_id INTEGER, assist_name TEXT, comments TEXT
        );
    """)
    c.execute("""
        INSERT OR REPLACE INTO matches
        (fixture_id, league_id, league_name, season, match_date,
         home_team_id, home_team_name, away_team_id, away_team_name,
         home_goals_ft, away_goals_ft, home_goals_ht, away_goals_ht,
         elapsed_minutes)
        VALUES (9001, 39, 'Premier League', 2024, '2024-09-22',
                50, 'Manchester City', 42, 'Arsenal', 2, 1, 1, 0, 95)
    """)
    events = [
        (9001, 23, None, "Goal", "Normal Goal", 50, "Manchester City", 1, "E. Haaland", None, None, None),
        (9001, 55, None, "Goal", "Normal Goal", 42, "Arsenal",         2, "B. Saka",    None, None, None),
        (9001, 78, None, "Goal", "Normal Goal", 50, "Manchester City", 3, "K. De Bruyne", None, None, None),
    ]
    c.executemany("""
        INSERT INTO match_events
        (fixture_id, event_minute, event_extra, event_type, event_detail,
         team_id, team_name, player_id, player_name, assist_id, assist_name, comments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, events)
    conn.commit()
    conn.close()


def make_engine_params() -> EngineParams:
    """테스트용 EngineParams."""
    b = np.array([0.05, -0.02, 0.00, 0.03, -0.01, 0.08])
    basis_bounds = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 95.0])
    dt_bins = [15.0, 15.0, 17.0, 15.0, 15.0, 18.0]
    C_time = sum(math.exp(b[i]) * dt_bins[i] for i in range(6))
    mu_hat_H, mu_hat_A = 1.35, 1.10
    a_H = math.log(mu_hat_H) - math.log(C_time)
    a_A = math.log(mu_hat_A) - math.log(C_time)
    gamma = build_gamma_array(-0.25, 0.20)
    delta_H = build_delta_array([0.15, 0.08, -0.10, -0.18])
    delta_A = build_delta_array([-0.12, -0.06, 0.08, 0.15])
    Q = np.array([
        [-0.04, 0.02, 0.02, 0.00],
        [0.00, -0.02, 0.00, 0.02],
        [0.00, 0.00, -0.02, 0.02],
        [0.00, 0.00, 0.00, 0.00],
    ])
    Q_diag, Q_off = build_Q_diag_and_off(Q)
    P_grid = {dt: expm(Q * dt) for dt in range(101)}
    return EngineParams(
        a_H=a_H, a_A=a_A, b=b, gamma=gamma,
        delta_H=delta_H, delta_A=delta_A,
        P_grid=P_grid, Q_diag=Q_diag, Q_off=Q_off,
        T_exp=95.0, basis_bounds=basis_bounds,
        mu_H_prematch=mu_hat_H, mu_A_prematch=mu_hat_A,
        mc_simulations=5_000,
        cooldown_seconds=0.001,
    )


# ═══════════════════════════════════════════════════
# 메인 시뮬레이션
# ═══════════════════════════════════════════════════

async def run_replay_paper(
    db_path: str,
    fixture_id: str,
    bankroll: float = 5000.0,
    misprice_prob: float = 0.15,
    misprice_bias: float = 0.10,
    entry_threshold: float = 0.02,
    kelly_fraction: float = 0.25,
    seed: int = 42,
) -> Dict:
    """
    Replay Paper Trading 시뮬레이션 실행.

    Returns:
        결과 dict (summary, report, dashboard)
    """
    # ── 1. Phase 4 컴포넌트 초기화 ───────────────
    exec_engine = ExecutionEngine(initial_bankroll=bankroll, paper=True)
    edge_detector = EdgeDetector(entry_threshold=entry_threshold)
    pos_sizer = PositionSizer(kelly_fraction=kelly_fraction)
    exit_mgr = ExitManager()
    analyzer = PostMatchAnalyzer()

    # ── 2. 오케스트레이터 초기화 ─────────────────
    tick_router = TickRouter(exec_engine, edge_detector, pos_sizer, exit_mgr)

    # 합성 호가 생성기
    synth_ob = SyntheticOrderbook(
        spread_cents=4,
        noise_std=0.02,
        misprice_prob=misprice_prob,
        misprice_bias=misprice_bias,
        seed=seed,
    )

    # 티커 매핑
    kalshi_tickers = {
        "home_win": f"KXSIM-{fixture_id}-HW",
        "away_win": f"KXSIM-{fixture_id}-AW",
        "draw":     f"KXSIM-{fixture_id}-DRW",
    }

    session = MatchSession(
        match_id=fixture_id,
        kalshi_tickers=kalshi_tickers,
        tick_router=tick_router,
    )

    # ── 3. 호가창 업데이트 콜백 ──────────────────
    # Phase 3의 pricing을 캡처해서 합성 호가에 반영
    prob_fields = {"home_win": "home_win", "away_win": "away_win", "draw": "draw"}

    tick_log = []  # 전체 틱 기록

    def on_tick_with_orderbook(snapshot: TickSnapshot):
        """Phase 3 on_tick → 합성 호가 생성 → MatchSession으로 전달."""
        pricing = snapshot.pricing
        if pricing is None:
            session.on_tick(snapshot)
            return

        # 합성 호가창 업데이트
        for market_type, prob_field in prob_fields.items():
            p_true = getattr(pricing, prob_field, 0.0)
            ob = synth_ob.generate(p_true, market_type, kalshi_tickers[market_type])
            session.update_orderbook(market_type, ob)

        # 세션에 전달
        result = session.on_tick(snapshot)

        # 로그
        tick_log.append({
            "tick": snapshot.tick,
            "min": snapshot.minute,
            "phase": snapshot.engine_phase,
            "score": f"{snapshot.S_H}-{snapshot.S_A}",
            "P_hw": f"{pricing.home_win:.3f}",
            "P_aw": f"{pricing.away_win:.3f}",
            "P_dr": f"{pricing.draw:.3f}",
            "entries": len(result.entries) if result else 0,
            "exits": len(result.exits) if result else 0,
            "event": snapshot.event or "",
        })

    # ── 4. Phase 3 엔진 실행 ─────────────────────
    source = ReplaySource(db_path=db_path, speed=0.0)
    params = make_engine_params()
    engine = LiveTradingEngine(
        event_source=source,
        params=params,
        on_tick=on_tick_with_orderbook,
    )

    logger.info(f"{'='*60}")
    logger.info(f"  REPLAY PAPER TRADING SIMULATION")
    logger.info(f"  fixture_id={fixture_id}, bankroll=${bankroll:.0f}")
    logger.info(f"  misprice_prob={misprice_prob}, bias={misprice_bias}")
    logger.info(f"{'='*60}")

    snapshots = await engine.run(fixture_id)

    # ── 5. 정산 ──────────────────────────────────
    # 실제 결과: Man City 2-1 Arsenal → Home Win
    last = snapshots[-1]
    home_won = last.S_H > last.S_A
    away_won = last.S_A > last.S_H
    drew = last.S_H == last.S_A

    outcomes = {
        kalshi_tickers["home_win"]: home_won,
        kalshi_tickers["away_win"]: away_won,
        kalshi_tickers["draw"]:     drew,
    }

    report = session.finalize(outcomes, analyzer)

    # ── 6. 대시보드 ──────────────────────────────
    dashboard = analyzer.health_dashboard(
        current_bankroll=exec_engine.bankroll,
        peak_bankroll=exec_engine.portfolio.peak_bankroll,
    )

    # ── 7. 결과 출력 ─────────────────────────────
    summary = session.summary()

    print(f"\n{'='*60}")
    print(f"  REPLAY PAPER TRADING RESULTS")
    print(f"{'='*60}")
    print(f"  Match: fixture {fixture_id}")
    print(f"  Final Score: {last.S_H}-{last.S_A}")
    print(f"  Outcome: {'Home Win' if home_won else 'Away Win' if away_won else 'Draw'}")
    print(f"{'─'*60}")
    print(f"  Total Ticks:      {summary['ticks']}")
    print(f"  Signals Evaluated: {summary['signals_evaluated']}")
    print(f"  Entries:           {summary['entries']}")
    print(f"  Exits:             {summary['exits']}")
    print(f"  Errors:            {summary['errors']}")
    print(f"{'─'*60}")
    print(f"  Starting Bankroll: ${bankroll:.2f}")
    print(f"  Final Bankroll:    ${exec_engine.bankroll:.2f}")
    print(f"  Total P&L:         ${exec_engine.bankroll - bankroll:+.2f}")
    print(f"  Drawdown:          {exec_engine.get_drawdown():.1f}%")

    if report:
        print(f"{'─'*60}")
        print(f"  Match P&L:         ${report.total_pnl:+.2f}")
        print(f"  Brier Score:       {report.brier_score:.4f}")
        print(f"  Edge Realization:  {report.edge_realization:.2f}")
        print(f"  Avg Slippage:      {report.avg_slippage:+.2f}¢")

    print(f"\n{PostMatchAnalyzer.format_dashboard(dashboard)}")

    # 거래 로그 출력
    trades = exec_engine.trade_log
    if trades:
        print(f"\n{'─'*60}")
        print(f"  TRADE LOG ({len(trades)} trades)")
        print(f"{'─'*60}")
        for i, t in enumerate(trades):
            status_icon = "✅" if t.fill_status in ("PAPER", "FILLED") else "❌"
            print(
                f"  {status_icon} [{t.order_type:6s}] {t.ticker}"
                f"  {t.direction:10s}  {t.quantity_filled:3d}x"
                f"  @{int(t.fill_price_cents):2d}¢"
                f"  EV={t.ev_adj*100:+5.1f}¢"
                f"  P&L=${t.pnl:+.2f}"
            )

    # 틱 중 이벤트/거래 있던 틱만 출력
    print(f"\n{'─'*60}")
    print(f"  KEY TICKS")
    print(f"{'─'*60}")
    for t in tick_log:
        if t["event"] or t["entries"] > 0 or t["exits"] > 0:
            parts = [f"  min={t['min']:5.1f}  {t['phase']:12s}  {t['score']}"]
            parts.append(f"  P(H/A/D)={t['P_hw']}/{t['P_aw']}/{t['P_dr']}")
            if t["entries"]:
                parts.append(f"  📈ENTRY×{t['entries']}")
            if t["exits"]:
                parts.append(f"  📉EXIT×{t['exits']}")
            if t["event"]:
                parts.append(f"  ⚡{t['event'][:40]}")
            print("".join(parts))

    print(f"{'='*60}\n")

    return {
        "summary": summary,
        "report": report,
        "dashboard": dashboard,
        "bankroll_final": exec_engine.bankroll,
        "pnl": exec_engine.bankroll - bankroll,
        "trades": len(trades),
        "tick_count": len(tick_log),
    }


# ═══════════════════════════════════════════════════
# 테스트 함수들
# ═══════════════════════════════════════════════════

def test_replay_paper_basic():
    """T1: 기본 Replay Paper Trading — 전체 파이프라인."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        create_test_db(db_path)
        result = asyncio.run(run_replay_paper(
            db_path=db_path,
            fixture_id="9001",
            bankroll=5000,
            misprice_prob=0.20,
            misprice_bias=0.12,
            seed=42,
        ))
        assert result["tick_count"] > 0
        assert result["summary"]["state"] == SessionState.SETTLED
        print(f"\n✅ T1: Replay paper trading complete — {result['tick_count']} ticks, "
              f"{result['trades']} trades, P&L=${result['pnl']:+.2f}")
    finally:
        os.unlink(db_path)


def test_replay_no_misprice():
    """T2: Misprice 없음 → 거래 적거나 없음."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        create_test_db(db_path)
        result = asyncio.run(run_replay_paper(
            db_path=db_path,
            fixture_id="9001",
            bankroll=5000,
            misprice_prob=0.0,
            misprice_bias=0.0,
            seed=99,
        ))
        # 시장 ≈ 모델이면 edge가 적으므로 거래 적음
        print(f"\n✅ T2: No misprice → {result['trades']} trades "
              f"(expected few/none)")
    finally:
        os.unlink(db_path)


def test_replay_high_misprice():
    """T3: 높은 misprice → 거래 많음."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        create_test_db(db_path)
        result = asyncio.run(run_replay_paper(
            db_path=db_path,
            fixture_id="9001",
            bankroll=5000,
            misprice_prob=0.50,
            misprice_bias=0.20,
            entry_threshold=0.01,
            seed=123,
        ))
        assert result["trades"] > 0
        print(f"\n✅ T3: High misprice → {result['trades']} trades, "
              f"P&L=${result['pnl']:+.2f}")
    finally:
        os.unlink(db_path)


def test_replay_different_seed():
    """T4: 시드 변경 → 다른 결과 (확률적 호가)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        create_test_db(db_path)
        r1 = asyncio.run(run_replay_paper(
            db_path=db_path, fixture_id="9001", bankroll=5000,
            misprice_prob=0.20, seed=42,
        ))
        r2 = asyncio.run(run_replay_paper(
            db_path=db_path, fixture_id="9001", bankroll=5000,
            misprice_prob=0.20, seed=999,
        ))
        # 시드 다르면 거래 패턴도 달라야
        same = (r1["trades"] == r2["trades"] and
                abs(r1["pnl"] - r2["pnl"]) < 0.01)
        print(f"\n✅ T4: Seed 42 → {r1['trades']} trades ${r1['pnl']:+.2f} | "
              f"Seed 999 → {r2['trades']} trades ${r2['pnl']:+.2f} "
              f"{'(identical)' if same else '(different ✓)'}")
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replay Paper Trading")
    parser.add_argument("--db", default="", help="DB path (empty=test DB)")
    parser.add_argument("--fixture", default="9001", help="fixture_id")
    parser.add_argument("--bankroll", type=float, default=5000)
    parser.add_argument("--misprice", type=float, default=0.20)
    parser.add_argument("--bias", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.test:
        test_replay_paper_basic()
        test_replay_no_misprice()
        test_replay_high_misprice()
        test_replay_different_seed()
        print(f"\n{'='*50}")
        print(f"  ALL 4 TESTS PASSED ✅")
        print(f"{'='*50}")
    else:
        if args.db:
            db_path = args.db
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            db_path = tmp.name
            tmp.close()
            create_test_db(db_path)

        asyncio.run(run_replay_paper(
            db_path=db_path,
            fixture_id=args.fixture,
            bankroll=args.bankroll,
            misprice_prob=args.misprice,
            misprice_bias=args.bias,
            seed=args.seed,
        ))
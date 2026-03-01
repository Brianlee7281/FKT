#!/usr/bin/env python3
"""
Paper Trading Runner — 라이브 페이퍼 트레이딩 실행.

모든 컴포넌트를 하나로 연결:
  GoalserveSource (3초 폴링)
      ↓ 이벤트
  Phase 3 LiveTradingEngine (확률 계산)
      ↓ TickSnapshot
  MatchSession → TickRouter
      ↓ 호가 참조
  KalshiOrderbookPoller (3초 폴링)
      ↓ 시그널
  Phase 4 (Paper 모드 — 주문 로깅만)

사용법:
  # 1) Kalshi 이벤트 자동 탐색 + Goalserve 라이브 매칭
  python tools/paper_runner.py \\
      --goalserve-match 6495401 \\
      --kalshi-event KXMLSGAME-26FEB28VANTOR \\
      --home "Vancouver" --away "Toronto"

  # 2) 수동 티커 지정
  python tools/paper_runner.py \\
      --goalserve-match 6495401 \\
      --tickers home_win=KXMLSGAME-26FEB28VANTOR-VAN \\
                away_win=KXMLSGAME-26FEB28VANTOR-TOR \\
                draw=KXMLSGAME-26FEB28VANTOR-TIE

  # 3) 라이브 경기 목록 + 자동 매칭
  python tools/paper_runner.py --discover mls

환경변수:
  GOALSERVE_API_KEY=<key>
  KALSHI_API_KEY=<key>
  KALSHI_PRIVATE_KEY_PATH=<path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.linalg import expm

# Phase 3
from src.phase3.goalserve import GoalserveSource
from src.phase3.engine import LiveTradingEngine, EngineParams, TickSnapshot
from src.phase3.mc_core import build_Q_diag_and_off
from src.phase3.mu_calculator import build_gamma_array, build_delta_array, build_basis_bounds

# Phase 4
from src.phase4.edge_detector import EdgeDetector
from src.phase4.position_sizer import PositionSizer
from src.phase4.exit_manager import ExitManager
from src.phase4.execution_engine import ExecutionEngine
from src.phase4.kalshi_client import KalshiClient

# Orchestrator
from src.orchestrator.tick_router import TickRouter, OrderbookSnapshot
from src.orchestrator.match_session import MatchSession
from src.orchestrator.orderbook_poller import KalshiOrderbookPoller
from src.orchestrator.ticker_mapper import TickerMapper, GOALSERVE_LEAGUE_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_runner")


def load_dotenv():
    """프로젝트 루트의 .env 파일에서 환경변수 로드."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip('"').strip("'")
                    if key and val and key not in os.environ:
                        os.environ[key] = val


# 시작 시 .env 로드
load_dotenv()


# ═══════════════════════════════════════════════════
# 기본 EngineParams (Phase 1 캘리브레이션 없이 사용)
# ═══════════════════════════════════════════════════

def make_default_engine_params(
    mu_H_prematch: float = 1.3,
    mu_A_prematch: float = 1.0,
) -> EngineParams:
    """
    Phase 1 모델 파일 없이 합리적인 기본 파라미터 생성.

    이 파라미터는 MLS 평균에 근거한 추정치이며,
    실제 캘리브레이션 후 대체된다.

    Args:
        mu_H_prematch: 홈팀 프리매치 기대 득점
        mu_A_prematch: 원정팀 프리매치 기대 득점
    """
    # NLL 기저함수 계수 (6개) — 평균적 시간 분포
    b = np.array([0.12, 0.10, 0.09, 0.11, 0.13, 0.15])

    # 마르코프 상태 효과 (11v11=0, 10v11=-1, 11v10=+1, 10v10=+0.5에 해당)
    gamma_1 = -0.15   # 홈팀 퇴장 효과
    gamma_2 = 0.12    # 원정팀 퇴장 효과
    gamma = build_gamma_array(gamma_1, gamma_2)

    # 득점차 효과 — [ΔS≤-2, ΔS=-1, ΔS=+1, ΔS≥+2]
    delta_H_raw = [-0.08, -0.04, 0.04, 0.06]  # 홈팀이 뒤지면 더 공격적
    delta_A_raw = [0.06, 0.04, -0.04, -0.08]   # 원정팀 대칭
    delta_H = build_delta_array(delta_H_raw)
    delta_A = build_delta_array(delta_A_raw)

    # Q 행렬 (마르코프 전이율)
    # 상태: 0=11v11, 1=10v11, 2=11v10, 3=10v10
    # 퇴장 발생률 ≈ 0.04/90분
    Q = np.array([
        [-0.0009, 0.0004, 0.0004, 0.0001],
        [0.0, -0.0005, 0.0001, 0.0004],
        [0.0, 0.0001, -0.0005, 0.0004],
        [0.0, 0.0, 0.0, 0.0],
    ])

    # P_grid: dt → exp(Q·dt)
    P_grid = {}
    for dt in range(1, 100):
        P_grid[dt] = expm(Q * dt)

    Q_diag, Q_off = build_Q_diag_and_off(Q)

    T_exp = 95.0
    basis_bounds = build_basis_bounds(first_half_end=47.0, T_m=T_exp)

    # a_H, a_A — 기저강도 역산
    # 공식: μ_H = Σ_ℓ Δτ_ℓ · exp(a_H + b_ℓ)  (at t=0, X=0, ΔS=0)
    #       = exp(a_H) · C_time
    #       → a_H = ln(μ_H / C_time)
    seg_durations = np.diff(basis_bounds)  # [15, 15, 17, 15, 15, 18]
    C_time = sum(d * np.exp(b[i]) for i, d in enumerate(seg_durations))
    a_H = np.log(mu_H_prematch / C_time)
    a_A = np.log(mu_A_prematch / C_time)

    return EngineParams(
        a_H=a_H,
        a_A=a_A,
        b=b,
        gamma=gamma,
        delta_H=delta_H,
        delta_A=delta_A,
        P_grid=P_grid,
        Q_diag=Q_diag,
        Q_off=Q_off,
        T_exp=T_exp,
        basis_bounds=basis_bounds,
        mu_H_prematch=mu_H_prematch,
        mu_A_prematch=mu_A_prematch,
        mc_simulations=20_000,    # 빠른 반복을 위해 줄임
        cooldown_seconds=15.0,
    )


# ═══════════════════════════════════════════════════
# Paper Trading 세션 로거
# ═══════════════════════════════════════════════════

class PaperLogger:
    """Paper Trading 결과를 로깅하고 CSV로 저장."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tick_log = []
        self.trade_log = []
        self.event_log = []

    def on_tick(self, snapshot: TickSnapshot, tick_result=None):
        """매 틱 기록."""
        entry = {
            "ts": datetime.now().isoformat(),
            "tick": snapshot.tick,
            "minute": snapshot.minute,
            "score": f"{snapshot.S_H}-{snapshot.S_A}",
            "phase": snapshot.engine_phase,
            "mu_H": round(snapshot.mu_H, 4),
            "mu_A": round(snapshot.mu_A, 4),
        }
        if snapshot.pricing:
            entry.update({
                "p_home": round(snapshot.pricing.home_win, 4),
                "p_draw": round(snapshot.pricing.draw, 4),
                "p_away": round(snapshot.pricing.away_win, 4),
            })
        if tick_result:
            entry["signals"] = tick_result.signals_evaluated
            entry["entries"] = len(tick_result.entries)
            entry["exits"] = len(tick_result.exits)

        self.tick_log.append(entry)

        # 이벤트
        if snapshot.event:
            self.event_log.append({
                "ts": datetime.now().isoformat(),
                "minute": snapshot.minute,
                "event": snapshot.event,
            })

    def on_trade(self, trade_type: str, record):
        """거래 기록."""
        self.trade_log.append({
            "ts": datetime.now().isoformat(),
            "type": trade_type,
            "ticker": getattr(record, "ticker", ""),
            "direction": getattr(record, "direction", ""),
            "contracts": getattr(record, "contracts", 0),
            "price": getattr(record, "fill_price_cents", 0),
            "pnl": getattr(record, "pnl", 0.0),
        })

    def save(self, match_info: str = ""):
        """세션 결과를 JSON으로 저장."""
        fname = f"{self.log_dir}/paper_{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "match_info": match_info,
            "ticks": len(self.tick_log),
            "trades": len(self.trade_log),
            "events": len(self.event_log),
            "tick_log": self.tick_log,
            "trade_log": self.trade_log,
            "event_log": self.event_log,
        }
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Paper log saved: {fname}")
        return fname

    def print_summary(self):
        """세션 요약 출력."""
        print(f"\n{'='*60}")
        print(f"  PAPER TRADING 세션 요약")
        print(f"{'='*60}")
        print(f"  틱:     {len(self.tick_log)}")
        print(f"  이벤트: {len(self.event_log)}")
        print(f"  거래:   {len(self.trade_log)}")

        if self.event_log:
            print(f"\n  ── 이벤트 ──")
            for ev in self.event_log:
                print(f"    {ev['minute']:.0f}' {ev['event'][:60]}")

        if self.trade_log:
            total_pnl = sum(t.get("pnl", 0) for t in self.trade_log)
            print(f"\n  ── 거래 ──")
            for t in self.trade_log:
                print(f"    {t['type']} {t['ticker']}: {t['contracts']}x @ {t['price']}¢ PnL=${t['pnl']:.2f}")
            print(f"\n  총 PnL: ${total_pnl:.2f}")
        else:
            print(f"\n  (거래 없음)")

        if self.tick_log:
            last = self.tick_log[-1]
            print(f"\n  최종: {last['score']} ({last['phase']})")
            if "p_home" in last:
                print(f"  P_true: H={last['p_home']:.3f} D={last['p_draw']:.3f} A={last['p_away']:.3f}")

        print(f"{'='*60}")


# ═══════════════════════════════════════════════════
# 메인 Paper Trading 실행
# ═══════════════════════════════════════════════════

async def run_paper_session(
    goalserve_match_id: str,
    kalshi_tickers: Dict[str, str],
    home_team: str = "Home",
    away_team: str = "Away",
    mu_H: float = 1.3,
    mu_A: float = 1.0,
    bankroll: float = 1000.0,
    log_dir: str = "logs",
):
    """
    단일 경기 Paper Trading 세션.

    Args:
        goalserve_match_id: Goalserve 경기 ID
        kalshi_tickers:     {"home_win": "...", "away_win": "...", "draw": "..."}
        home_team:          홈팀명
        away_team:          원정팀명
        mu_H:               프리매치 홈 기대 득점
        mu_A:               프리매치 원정 기대 득점
        bankroll:           가상 자본금 ($)
        log_dir:            로그 저장 디렉토리
    """
    paper_log = PaperLogger(log_dir=log_dir)

    print(f"\n{'='*60}")
    print(f"  PAPER TRADING SESSION")
    print(f"  {home_team} vs {away_team}")
    print(f"  Goalserve: {goalserve_match_id}")
    print(f"  Kalshi:    {list(kalshi_tickers.values())[:1]}...")
    print(f"  μ_H={mu_H:.2f}  μ_A={mu_A:.2f}  Bankroll=${bankroll:.0f}")
    print(f"{'='*60}\n")

    # ── 1. Phase 3 엔진 초기화 ──────────────────

    engine_params = make_default_engine_params(mu_H, mu_A)
    goalserve = GoalserveSource(poll_interval=3.0)

    # ── 2. Phase 4 파이프라인 (Paper 모드) ──────

    edge_detector = EdgeDetector()
    sizer = PositionSizer(
        kelly_fraction=0.25,
        order_cap=0.05,
        match_cap=0.15,
        total_cap=0.30,
    )
    exit_mgr = ExitManager()
    exec_engine = ExecutionEngine(
        kalshi_client=None,
        initial_bankroll=bankroll,  # 달러 단위
        paper=True,
    )

    tick_router = TickRouter(
        engine=exec_engine,
        detector=edge_detector,
        sizer=sizer,
        exit_mgr=exit_mgr,
    )

    # ── 3. MatchSession 생성 ────────────────────

    session = MatchSession(
        match_id=goalserve_match_id,
        kalshi_tickers=kalshi_tickers,
        tick_router=tick_router,
    )

    # ── 4. Kalshi 호가 폴러 ─────────────────────

    kalshi_client = None
    ob_poller = None
    ob_task = None

    try:
        kalshi_client = KalshiClient()
        await kalshi_client.connect()
        logger.info("Kalshi 연결 성공")

        ob_poller = KalshiOrderbookPoller(
            kalshi_client=kalshi_client,
            tickers=kalshi_tickers,
            poll_interval=3.0,
        )
        ob_poller.bind_to_session(session)
        ob_task = await ob_poller.start_background()
        logger.info("Kalshi 호가 폴링 시작")

    except Exception as e:
        logger.warning(f"Kalshi 연결 실패: {e} — 합성 호가 사용")

    # ── 5. on_tick 콜백 ─────────────────────────

    tick_count = [0]

    def on_tick_callback(snapshot: TickSnapshot):
        """Phase 3 → Phase 4 → 로깅."""
        tick_count[0] += 1

        # MatchSession 경유 → TickRouter 실행
        tick_result = session.on_tick(snapshot)

        # 로깅
        paper_log.on_tick(snapshot, tick_result)

        # 거래 기록
        if tick_result:
            for entry in tick_result.entries:
                paper_log.on_trade("ENTRY", entry)
            for exit_rec in tick_result.exits:
                paper_log.on_trade("EXIT", exit_rec)

        # 주기적 출력 (30틱마다)
        if tick_count[0] % 30 == 0 or snapshot.event:
            ts = datetime.now().strftime("%H:%M:%S")
            pricing_str = ""
            if snapshot.pricing:
                p = snapshot.pricing
                pricing_str = f"H={p.home_win:.3f} D={p.draw:.3f} A={p.away_win:.3f}"

            ob_str = ""
            if ob_poller:
                ob_str = ob_poller.format_latest()

            print(
                f"  [{ts}] {snapshot.minute:.0f}' "
                f"{snapshot.S_H}-{snapshot.S_A} "
                f"({snapshot.engine_phase}) "
                f"{pricing_str}"
            )
            if ob_str:
                print(f"           OB: {ob_str}")
            if snapshot.event:
                print(f"           EVENT: {snapshot.event[:80]}")

    # ── 6. 엔진 실행 ───────────────────────────

    engine = LiveTradingEngine(
        event_source=goalserve,
        params=engine_params,
        on_tick=on_tick_callback,
    )

    logger.info("엔진 시작 — 경기 종료까지 실행...")

    try:
        await engine.run(goalserve_match_id)
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        # 정리
        if ob_poller:
            await ob_poller.stop_and_wait()
        if kalshi_client:
            await kalshi_client.disconnect()

    # ── 7. 결과 정리 ───────────────────────────

    # 경기 종료 정산
    match_info = f"{home_team} vs {away_team} ({goalserve_match_id})"
    paper_log.print_summary()
    log_file = paper_log.save(match_info)

    # ExecutionEngine 통계
    stats = exec_engine.risk_dashboard()
    print(f"\n  ExecutionEngine: {json.dumps(stats, indent=2, default=str)}")

    if ob_poller:
        print(f"  Orderbook Poller: {ob_poller.format_stats()}")

    print(f"\n  로그: {log_file}")
    return session


# ═══════════════════════════════════════════════════
# 경기 탐색 모드
# ═══════════════════════════════════════════════════

async def discover_matches(league: str = "mls"):
    """Goalserve + Kalshi에서 매칭 가능한 경기 탐색."""
    import httpx

    api_key = os.getenv("GOALSERVE_API_KEY", "")
    if not api_key:
        print("❌ GOALSERVE_API_KEY 미설정")
        print("   .env 파일 또는 export GOALSERVE_API_KEY=<key>")
        return

    print(f"\n{'='*70}")
    print(f"  경기 탐색: {league.upper()}")
    print(f"{'='*70}")

    # Goalserve 라이브 경기
    url = f"http://www.goalserve.com/getfeed/{api_key}/soccernew/live?json=1"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        gs_data = resp.json()

    gs_matches = GoalserveSource.find_all_matches(gs_data)
    print(f"\n  Goalserve: {len(gs_matches)}개 경기")

    # Kalshi 이벤트
    mapper = TickerMapper()
    try:
        kalshi = KalshiClient()
        await kalshi.connect()
        kalshi_events = await mapper.discover_all_for_league(kalshi, league)
        await kalshi.disconnect()
    except Exception as e:
        print(f"  Kalshi 연결 실패: {e}")
        kalshi_events = []

    print(f"  Kalshi:    {len(kalshi_events)}개 이벤트")

    # 매칭 시도
    print(f"\n  ── 매칭 결과 ──")
    from src.orchestrator.ticker_mapper import _fuzzy_team_match

    matched = []
    for gs in gs_matches:
        for ke in kalshi_events:
            if (_fuzzy_team_match(gs["home"].lower(), ke["title"].lower())
                    and _fuzzy_team_match(gs["away"].lower(), ke["title"].lower())):
                matched.append({
                    "gs_id": gs["match_id"],
                    "gs_home": gs["home"],
                    "gs_away": gs["away"],
                    "gs_score": gs["score"],
                    "gs_status": gs["status"],
                    "kalshi_event": ke["event_ticker"],
                    "kalshi_title": ke["title"],
                })

    if matched:
        for m in matched:
            print(f"\n    ✅ {m['gs_home']} vs {m['gs_away']}")
            print(f"       GS:    {m['gs_id']} ({m['gs_score']}, {m['gs_status']})")
            print(f"       Kalshi: {m['kalshi_event']}")
    else:
        print("    매칭된 경기 없음")
        print("\n  Goalserve 경기:")
        for gs in gs_matches[:10]:
            print(f"    [{gs['match_id']}] {gs['home']} vs {gs['away']} ({gs['status']})")
        print("\n  Kalshi 이벤트:")
        for ke in kalshi_events[:10]:
            print(f"    {ke['event_ticker']}: {ke['title']}")

    return matched


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def parse_tickers(ticker_args: list) -> Dict[str, str]:
    """'home_win=TICKER' 형태의 인자를 파싱."""
    tickers = {}
    for arg in ticker_args:
        if "=" in arg:
            key, val = arg.split("=", 1)
            tickers[key] = val
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Runner")

    # 모드
    parser.add_argument("--discover", type=str, metavar="LEAGUE",
                       help="경기 탐색 (mls, epl, ...)")

    # 경기 설정
    parser.add_argument("--goalserve-match", type=str, metavar="ID",
                       help="Goalserve 경기 ID")
    parser.add_argument("--kalshi-event", type=str, metavar="EVENT",
                       help="Kalshi 이벤트 티커")
    parser.add_argument("--tickers", nargs="+", metavar="KEY=VAL",
                       help="수동 티커 지정 (home_win=X away_win=Y draw=Z)")
    parser.add_argument("--home", type=str, default="Home")
    parser.add_argument("--away", type=str, default="Away")

    # 파라미터
    parser.add_argument("--mu-h", type=float, default=1.3,
                       help="홈팀 프리매치 기대 득점")
    parser.add_argument("--mu-a", type=float, default=1.0,
                       help="원정팀 프리매치 기대 득점")
    parser.add_argument("--bankroll", type=float, default=1000,
                       help="가상 자본금 ($)")
    parser.add_argument("--log-dir", type=str, default="logs")

    args = parser.parse_args()

    # 탐색 모드
    if args.discover:
        asyncio.run(discover_matches(args.discover))
        return

    # 실행 모드
    if not args.goalserve_match:
        parser.print_help()
        print("\n예시:")
        print("  python tools/paper_runner.py --discover mls")
        print("  python tools/paper_runner.py \\")
        print("      --goalserve-match 6495401 \\")
        print("      --kalshi-event KXMLSGAME-26FEB28VANTOR \\")
        print("      --home Vancouver --away Toronto")
        return

    # 티커 설정
    if args.tickers:
        kalshi_tickers = parse_tickers(args.tickers)
    elif args.kalshi_event:
        # 이벤트에서 마켓 티커 자동 생성
        # 패턴: EVENT-HOMECODE, EVENT-AWAYCODE, EVENT-TIE
        # 이건 실행 시 Kalshi API로 조회해야 정확
        logger.info(f"Kalshi 이벤트에서 마켓 조회: {args.kalshi_event}")
        kalshi_tickers = {}  # run 안에서 조회
    else:
        print("❌ --tickers 또는 --kalshi-event 필요")
        return

    # Kalshi 이벤트에서 마켓 조회
    if args.kalshi_event and not kalshi_tickers:
        async def fetch_tickers():
            client = KalshiClient()
            await client.connect()
            data = await client._get("/markets", {
                "event_ticker": args.kalshi_event,
                "limit": "10",
            })
            markets = data.get("markets", [])
            await client.disconnect()

            from src.orchestrator.ticker_mapper import _classify_markets
            return _classify_markets(
                markets,
                args.home.lower(),
                args.away.lower(),
            )

        kalshi_tickers = asyncio.run(fetch_tickers())
        if not kalshi_tickers:
            print(f"❌ {args.kalshi_event}에서 마켓 못 찾음")
            return
        print(f"  마켓 발견: {kalshi_tickers}")

    # 실행
    asyncio.run(run_paper_session(
        goalserve_match_id=args.goalserve_match,
        kalshi_tickers=kalshi_tickers,
        home_team=args.home,
        away_team=args.away,
        mu_H=args.mu_h,
        mu_A=args.mu_a,
        bankroll=args.bankroll,
        log_dir=args.log_dir,
    ))


if __name__ == "__main__":
    main()
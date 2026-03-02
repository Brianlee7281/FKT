"""
Phase 3: Live Trading Engine (engine.py)

모든 Phase 3 컴포넌트를 연결하는 메인 오케스트레이션 루프.

매 초(1틱) 사이클:
  ① 시간 전진 (t += 1/60)
  ② 이벤트 확인 (EventSource에서 도착한 이벤트 처리)
  ③ μ 계산 (compute_remaining_mu)
  ④ P_true 산출 (analytical 또는 MC)
  ⑤ Phase 4 전달 (P_true, σ_MC, orders_allowed)

구성 요소:
  - EventSource (ReplaySource / GoalserveSource)
  - EventHandler (골/퇴장/하프타임 처리)
  - StoppageTimeManager (T 실시간 보정)
  - mu_calculator (잔여 기대 득점)
  - pricer (해석적 + MC 프라이싱)
  - mc_core (Numba MC 시뮬레이션)

사용법:
  engine = LiveTradingEngine(
      event_source=ReplaySource(db_path="data/kalshi_football.db"),
      params=phase2_state,   # Phase 2 초기화 결과
  )
  await engine.run("1035068")  # fixture_id
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np

from src.phase3.event_source import EventSource, EventType, NormalizedEvent
from src.phase3.event_handler import EventHandler, GameState
from src.phase3.stoppage import StoppageTimeManager
from src.phase3.mu_calculator import (
    compute_remaining_mu,
    build_gamma_array,
    build_delta_array,
    build_basis_bounds,
)
from src.phase3.pricer import (
    analytical_pricing,
    aggregate_mc_results,
    price,
    PricingResult,
)
from src.phase3.mc_core import (
    mc_simulate_remaining,
    build_Q_diag_and_off,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 틱 스냅샷 (매 틱 결과 기록)
# ═════════════════════════════════════════════════════════

@dataclass
class TickSnapshot:
    """매 틱마다 생성되는 엔진 상태 스냅샷."""
    tick: int
    minute: float
    S_H: int
    S_A: int
    delta_S: int
    X: int
    engine_phase: str
    mu_H: float
    mu_A: float
    T: float
    pricing: Optional[PricingResult] = None
    orders_allowed: bool = False
    cooldown: bool = False
    ob_freeze: bool = False
    event: Optional[str] = None  # 이 틱에서 발생한 이벤트 요약


# ═════════════════════════════════════════════════════════
# 엔진 파라미터
# ═════════════════════════════════════════════════════════

@dataclass
class EngineParams:
    """
    Phase 2에서 넘어오는 파라미터 패키지.

    Phase 2의 MatchState에서 필요한 필드만 추출하여 구성.
    """
    # 기본 강도
    a_H: float
    a_A: float

    # Phase 1 NLL 파라미터
    b: np.ndarray            # shape (6,)
    gamma: np.ndarray        # shape (4,) — [0, γ₁, γ₂, γ₁+γ₂]
    delta_H: np.ndarray      # shape (5,) — ΔS=0 삽입된 버전
    delta_A: np.ndarray      # shape (5,)

    # 마르코프 모델
    P_grid: Dict[int, np.ndarray]    # dt → exp(Q·dt), shape (4,4)
    Q_diag: np.ndarray               # shape (4,) — Q 대각
    Q_off: np.ndarray                # shape (4,4) — 정규화된 비대각 전이

    # 시간
    T_exp: float                     # 기대 경기 시간 (~95분)
    basis_bounds: np.ndarray         # shape (7,) — 기저함수 경계

    # 프리매치 예측 (검증용)
    mu_H_prematch: float = 0.0
    mu_A_prematch: float = 0.0

    # MC 설정
    mc_simulations: int = 50_000

    # 쿨다운
    cooldown_seconds: float = 15.0

    # 틱 간격 (데이터 폴링에 맞춤, 기본 3초)
    tick_interval: float = 3.0

    @classmethod
    def from_phase2(cls, state) -> EngineParams:
        """
        Phase 2 MatchState에서 EngineParams를 생성한다.

        Args:
            state: initializer.py의 MatchState 객체
        """
        b = np.array(state.b)
        gamma = build_gamma_array(state.gamma_1, state.gamma_2)
        delta_H = build_delta_array(state.delta_H)
        delta_A = build_delta_array(state.delta_A)
        Q_diag, Q_off = build_Q_diag_and_off(state.Q)

        basis_bounds = build_basis_bounds(
            first_half_end=47.0,   # TODO: Phase 2에서 전달받기
            T_m=state.T_exp,
        )

        return cls(
            a_H=state.a_H,
            a_A=state.a_A,
            b=b,
            gamma=gamma,
            delta_H=delta_H,
            delta_A=delta_A,
            P_grid=state.P_grid,
            Q_diag=Q_diag,
            Q_off=Q_off,
            T_exp=state.T_exp,
            basis_bounds=basis_bounds,
            mu_H_prematch=state.mu_H,
            mu_A_prematch=state.mu_A,
        )


# ═════════════════════════════════════════════════════════
# Live Trading Engine
# ═════════════════════════════════════════════════════════

class LiveTradingEngine:
    """
    Phase 3 메인 오케스트레이션 엔진.

    두 개의 비동기 태스크가 동시에 실행된다:
      1. tick_loop:     매 틱 시간 전진 + μ 계산 + 프라이싱
      2. event_loop:    EventSource에서 이벤트 수신 + 상태 업데이트

    Args:
        event_source: EventSource 구현체 (ReplaySource / GoalserveSource)
        params:       Phase 2에서 전달된 EngineParams
        on_tick:      매 틱 콜백 (Optional). TickSnapshot을 받는다.
    """

    def __init__(
        self,
        event_source: EventSource,
        params: EngineParams,
        on_tick: Optional[Callable[[TickSnapshot], None]] = None,
    ):
        self.source = event_source
        self.params = params
        self.on_tick = on_tick

        # 컴포넌트 초기화
        self.handler = EventHandler(cooldown_seconds=params.cooldown_seconds)
        self.stoppage = StoppageTimeManager(T_exp=params.T_exp)
        self.state = GameState()

        # 틱 카운터 & 기록
        self.tick_count = 0
        self.snapshots: List[TickSnapshot] = []

        # 이벤트 큐 (event_loop → tick_loop)
        self._event_queue: asyncio.Queue[NormalizedEvent] = asyncio.Queue()

        # 현재 μ, pricing
        self._mu_H = params.mu_H_prematch
        self._mu_A = params.mu_A_prematch
        self._pricing: Optional[PricingResult] = None

        # 초기 상태 오버라이드 (mid-game join 용)
        self._initial_state: Optional[Dict] = None

    def set_initial_state(self, state: Dict) -> None:
        """
        Mid-game join 시 초기 상태 설정.

        run() 호출 전에 실행해야 함.

        Args:
            state: {
                "score_home": int,
                "score_away": int,
                "minute": float,
                "phase": str,  # "FIRST_HALF", "HALFTIME", "SECOND_HALF"
                "X": int,      # 마르코프 상태
            }
        """
        self._initial_state = state
        logger.info(
            f"엔진 초기 상태 설정: "
            f"{state.get('score_home',0)}-{state.get('score_away',0)} "
            f"{state.get('minute',0):.0f}' ({state.get('phase','?')}) "
            f"X={state.get('X',0)}"
        )

    async def run(self, match_id: str) -> List[TickSnapshot]:
        """
        엔진 실행. 경기 종료까지 틱 루프를 돌린다.

        Args:
            match_id: fixture_id (문자열)

        Returns:
            전체 틱 스냅샷 리스트
        """
        logger.info(f"엔진 시작: match_id={match_id}")

        # EventSource 연결
        await self.source.connect(match_id)

        # GoalserveSource의 초기 상태 자동 적용 (mid-game join)
        if self._initial_state is None and hasattr(self.source, 'initial_state'):
            if self.source.initial_state:
                self.set_initial_state(self.source.initial_state)

        try:
            # 두 태스크를 동시에 실행
            event_task = asyncio.create_task(self._event_loop())
            tick_task = asyncio.create_task(self._tick_loop())

            # tick_loop가 FINISHED로 끝날 때까지 대기
            await tick_task

            # event_task 정리
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

        finally:
            await self.source.disconnect()

        logger.info(
            f"엔진 종료: {self.state.S_H}-{self.state.S_A}, "
            f"틱={self.tick_count}, 이벤트={self.handler.event_count}개"
        )

        return self.snapshots

    # ─── 이벤트 루프 ──────────────────────────────────

    async def _event_loop(self) -> None:
        """EventSource에서 이벤트를 수신하여 큐에 넣는다."""
        try:
            async for event in self.source.listen():
                await self._event_queue.put(event)
        except asyncio.CancelledError:
            pass

    # ─── 틱 루프 ──────────────────────────────────────

    async def _tick_loop(self) -> None:
        """매 틱 메인 사이클."""

        # 초기 상태 적용 (mid-game join)
        if self._initial_state:
            s = self._initial_state
            self.state.S_H = s.get("score_home", 0)
            self.state.S_A = s.get("score_away", 0)
            self.state.delta_S = self.state.S_H - self.state.S_A
            self.state.X = s.get("X", 0)
            self.state.current_minute = s.get("minute", 0.0)
            self.state.engine_phase = s.get("phase", "FIRST_HALF")

            # 추가시간 매니저도 동기화
            if self.state.current_minute > 45:
                self.stoppage.reset_for_second_half()

            logger.info(
                f"Mid-game join: {self.state.S_H}-{self.state.S_A} "
                f"{self.state.current_minute:.0f}' "
                f"({self.state.engine_phase}) X={self.state.X}"
            )
        else:
            self.state.engine_phase = "FIRST_HALF"

        while self.state.engine_phase != "FINISHED":
            # ① 큐에 쌓인 이벤트 모두 처리 (이벤트 기반 시간 전진)
            events_this_tick = []
            while True:
                event = await self._get_next_event()
                if event is None:
                    break
                self._process_event(event)
                events_this_tick.append(str(event))

            event_str = " | ".join(events_this_tick) if events_this_tick else None

            # ③ 시간 & 계산 (활성 플레이 중일 때)
            if self.state.engine_phase in ("FIRST_HALF", "SECOND_HALF"):
                # T 업데이트 (추가시간 관리)
                T = self.stoppage.update(
                    self.state.current_minute,
                    self.state.engine_phase,
                )

                # μ 계산
                self._mu_H, self._mu_A = compute_remaining_mu(
                    t=self.state.current_minute,
                    T=T,
                    X=self.state.X,
                    delta_S=self.state.delta_S,
                    a_H=self.params.a_H,
                    a_A=self.params.a_A,
                    b=self.params.b,
                    gamma=self.params.gamma,
                    delta_H=self.params.delta_H,
                    delta_A=self.params.delta_A,
                    P_grid=self.params.P_grid,
                    basis_bounds=self.params.basis_bounds,
                )

                # 프라이싱
                self._pricing = self._compute_pricing()

            # ④ 스냅샷 기록
            snapshot = TickSnapshot(
                tick=self.tick_count,
                minute=self.state.current_minute,
                S_H=self.state.S_H,
                S_A=self.state.S_A,
                delta_S=self.state.delta_S,
                X=self.state.X,
                engine_phase=self.state.engine_phase,
                mu_H=self._mu_H,
                mu_A=self._mu_A,
                T=self.stoppage.T,
                pricing=self._pricing,
                orders_allowed=self.state.orders_allowed,
                cooldown=self.state.cooldown,
                ob_freeze=self.state.ob_freeze,
                event=event_str,
            )
            self.snapshots.append(snapshot)

            if self.on_tick:
                self.on_tick(snapshot)

            self.tick_count += 1

            if self.state.engine_phase == "FINISHED":
                break

            # 데이터 폴링 간격에 맞춰 대기 (기본 3초)
            # 이벤트가 있으면 _get_next_event에서 즉시 반환되므로
            # 실질적으로 이벤트 직후에는 빠르게 반응
            await asyncio.sleep(self.params.tick_interval)

    # ─── 이벤트 처리 ──────────────────────────────────

    async def _get_next_event(self) -> Optional[NormalizedEvent]:
        """큐에서 이벤트 하나를 가져온다. 없으면 즉시 반환."""
        try:
            return self._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _process_event(self, event: NormalizedEvent) -> None:
        """단일 이벤트를 처리한다."""

        # EventHandler에 위임 (S, X, ΔS, engine_phase 업데이트)
        self.handler.handle(event, self.state)

        # 추가시간 진입 이벤트는 StoppageTimeManager에도 전달
        if event.event_type == EventType.STOPPAGE_ENTERED:
            half = event.raw.get("half", "unknown")
            self.stoppage.on_stoppage_entered(half)

        # 후반 시작 시 StoppageTimeManager 리셋
        if event.event_type == EventType.SECOND_HALF_START:
            self.stoppage.reset_for_second_half()

    # ─── 프라이싱 ─────────────────────────────────────

    def _compute_pricing(self) -> PricingResult:
        """현재 상태에서 P_true를 계산한다."""

        use_mc = (self.state.X != 0 or self.state.delta_S != 0)

        if not use_mc:
            # 해석적 (빠름)
            return analytical_pricing(
                mu_H=self._mu_H,
                mu_A=self._mu_A,
                S_H=self.state.S_H,
                S_A=self.state.S_A,
            )
        else:
            # MC 시뮬레이션
            seed = int(time.time() * 1000) % (2**31)
            final_scores = mc_simulate_remaining(
                t_now=self.state.current_minute,
                T_end=self.stoppage.T,
                S_H=self.state.S_H,
                S_A=self.state.S_A,
                state=self.state.X,
                score_diff=self.state.delta_S,
                a_H=self.params.a_H,
                a_A=self.params.a_A,
                b=self.params.b,
                gamma=self.params.gamma,
                delta_H=self.params.delta_H,
                delta_A=self.params.delta_A,
                Q_diag=self.params.Q_diag,
                Q_off=self.params.Q_off,
                basis_bounds=self.params.basis_bounds,
                N=self.params.mc_simulations,
                seed=seed,
            )
            return aggregate_mc_results(final_scores)

    # ─── 편의 메서드 ──────────────────────────────────

    def summary(self) -> str:
        """엔진 실행 결과 요약."""
        if not self.snapshots:
            return "엔진: 실행 전"

        last = self.snapshots[-1]
        goals = [
            s for s in self.snapshots if s.event and "goal" in s.event.lower()
        ]
        reds = [
            s for s in self.snapshots if s.event and "red_card" in s.event.lower()
        ]

        lines = [
            f"═══ Phase 3 Engine 결과 ═══",
            f"최종 스코어: {last.S_H}-{last.S_A}",
            f"마르코프 상태: X={last.X}",
            f"총 틱: {self.tick_count}",
            f"골: {len(goals)}개, 퇴장: {len(reds)}개",
            f"이벤트 총: {self.handler.event_count}개",
        ]

        if last.pricing:
            p = last.pricing
            lines.extend([
                f"최종 P_true:",
                f"  Home={p.home_win:.3f} Draw={p.draw:.3f} Away={p.away_win:.3f}",
                f"  Over2.5={p.over_25:.3f} (mode={p.mode})",
            ])

        return "\n".join(lines)
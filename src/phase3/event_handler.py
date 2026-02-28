"""
Phase 3 Step 3.3: 불연속적 충격 처리 (event_handler.py)

이벤트 소스(ReplaySource/GoalserveSource)에서 NormalizedEvent가 도착하면,
연속적이던 μ 계산에 불연속적 충격(Shock)을 가한다.

처리하는 이벤트:
  - GOAL:              S, ΔS 업데이트 → λ 점프 → 쿨다운 활성화
  - RED_CARD:          X 마르코프 전이 → γ 변경 → 쿨다운 활성화
  - HALFTIME:          engine_phase → HALFTIME, 프라이싱 동결
  - SECOND_HALF_START: engine_phase → SECOND_HALF, 프라이싱 재개
  - STOPPAGE_ENTERED:  StoppageTimeManager에 위임
  - MATCH_END:         engine_phase → FINISHED, 최종 정산

사용법:
  handler = EventHandler(cooldown_seconds=15.0)
  new_state = handler.handle(event, current_state)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.phase3.event_source import EventType, NormalizedEvent

logger = logging.getLogger(__name__)

# 상수
DEFAULT_COOLDOWN_SECONDS = 15.0


# ═════════════════════════════════════════════════════════
# 게임 상태
# ═════════════════════════════════════════════════════════

@dataclass
class GameState:
    """
    Phase 3 엔진의 전체 게임 상태.

    engine.py의 매 틱마다 참조되며,
    이벤트 발생 시 EventHandler가 업데이트한다.

    Attributes:
        S_H, S_A:       현재 스코어
        delta_S:         S_H - S_A
        X:               마르코프 상태 (0=11v11, 1=홈퇴장, 2=원정퇴장, 3=양쪽)
        engine_phase:    "PRE_MATCH", "FIRST_HALF", "SECOND_HALF",
                         "HALFTIME", "FINISHED"
        cooldown:        True이면 주문 차단 (P_true 계산은 계속)
        cooldown_until:  쿨다운 해제 시각 (time.monotonic 기준)
        ob_freeze:       호가 이상 감지에 의한 동결
        current_minute:  현재 경기 시간 (분)
    """
    S_H: int = 0
    S_A: int = 0
    delta_S: int = 0
    X: int = 0
    engine_phase: str = "PRE_MATCH"
    cooldown: bool = False
    cooldown_until: float = 0.0
    ob_freeze: bool = False
    current_minute: float = 0.0

    def update_cooldown(self) -> None:
        """시간 경과에 따라 쿨다운을 자동 해제한다."""
        if self.cooldown and time.monotonic() >= self.cooldown_until:
            self.cooldown = False

    @property
    def orders_allowed(self) -> bool:
        """주문 가능 여부 (쿨다운도 아니고 ob_freeze도 아닐 때)"""
        self.update_cooldown()
        return (
            not self.cooldown
            and not self.ob_freeze
            and self.engine_phase in ("FIRST_HALF", "SECOND_HALF")
        )

    def __str__(self) -> str:
        return (
            f"GameState({self.S_H}-{self.S_A}, X={self.X}, "
            f"phase={self.engine_phase}, "
            f"cd={'Y' if self.cooldown else 'N'}, "
            f"ob={'Y' if self.ob_freeze else 'N'}, "
            f"t={self.current_minute:.1f})"
        )


# ═════════════════════════════════════════════════════════
# 이벤트 핸들러
# ═════════════════════════════════════════════════════════

class EventHandler:
    """
    NormalizedEvent를 받아 GameState를 업데이트하는 핸들러.

    Args:
        cooldown_seconds: 골/퇴장 후 주문 차단 시간 (기본 15초)
    """

    def __init__(self, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._event_log: list = []

    def handle(self, event: NormalizedEvent, state: GameState) -> GameState:
        """
        이벤트를 처리하고 업데이트된 GameState를 반환한다.

        Args:
            event: 처리할 NormalizedEvent
            state: 현재 GameState (in-place로 수정됨)

        Returns:
            수정된 GameState (동일 객체)
        """
        state.current_minute = event.minute

        handler_map = {
            EventType.GOAL: self._handle_goal,
            EventType.RED_CARD: self._handle_red_card,
            EventType.HALFTIME: self._handle_halftime,
            EventType.SECOND_HALF_START: self._handle_second_half_start,
            EventType.STOPPAGE_ENTERED: self._handle_stoppage_entered,
            EventType.MATCH_END: self._handle_match_end,
        }

        handler = handler_map.get(event.event_type)
        if handler:
            handler(event, state)
            self._event_log.append(event)

        return state

    # ─── 이벤트별 핸들러 ──────────────────────────────

    def _handle_goal(self, event: NormalizedEvent, state: GameState) -> None:
        """
        골 처리: S, ΔS 업데이트 + 쿨다운 활성화.

        득점팀에 따라 스코어를 올리고, ΔS(S_H - S_A)를 갱신한다.
        δ(ΔS) 변경으로 인해 λ_H, λ_A가 동시에 점프한다.
        """
        if event.team == "home":
            state.S_H += 1
        elif event.team == "away":
            state.S_A += 1

        state.delta_S = state.S_H - state.S_A

        # 쿨다운 활성화
        self._activate_cooldown(state)

        logger.info(
            f"⚽ 골! {event.team} @ {event.minute:.0f}분 → "
            f"{state.S_H}-{state.S_A} (ΔS={state.delta_S})"
        )

    def _handle_red_card(self, event: NormalizedEvent, state: GameState) -> None:
        """
        레드카드 처리: X 마르코프 상태 전이 + 쿨다운 활성화.

        전이 규칙:
          홈 퇴장: 0→1, 2→3
          원정 퇴장: 0→2, 1→3
        """
        old_X = state.X

        if event.team == "home":
            if state.X == 0:
                state.X = 1    # 11v11 → 10v11
            elif state.X == 2:
                state.X = 3    # 11v10 → 10v10
        elif event.team == "away":
            if state.X == 0:
                state.X = 2    # 11v11 → 11v10
            elif state.X == 1:
                state.X = 3    # 10v11 → 10v10

        # 쿨다운 활성화
        self._activate_cooldown(state)

        logger.info(
            f"🟥 퇴장! {event.team} @ {event.minute:.0f}분 → "
            f"X: {old_X}→{state.X}"
        )

    def _handle_halftime(self, event: NormalizedEvent, state: GameState) -> None:
        """하프타임 진입: 프라이싱 동결, 주문 차단."""
        state.engine_phase = "HALFTIME"
        logger.info(
            f"⏸️ 하프타임 → {state.S_H}-{state.S_A}"
        )

    def _handle_second_half_start(
        self, event: NormalizedEvent, state: GameState
    ) -> None:
        """후반 시작: 프라이싱 재개."""
        state.engine_phase = "SECOND_HALF"
        logger.info(
            f"▶️ 후반 시작 @ {event.minute:.1f}분"
        )

    def _handle_stoppage_entered(
        self, event: NormalizedEvent, state: GameState
    ) -> None:
        """
        추가시간 진입: StoppageTimeManager에 위임.

        직접 T를 수정하지 않음 — engine.py가 StoppageTimeManager.on_stoppage_entered()
        를 호출하도록 시그널만 남긴다.
        """
        half = event.raw.get("half", "unknown")
        logger.info(
            f"⏱️ 추가시간 진입 ({half}) @ {event.minute:.1f}분"
        )

    def _handle_match_end(
        self, event: NormalizedEvent, state: GameState
    ) -> None:
        """경기 종료: 최종 정산."""
        state.engine_phase = "FINISHED"
        logger.info(
            f"🏁 경기 종료 → 최종 {state.S_H}-{state.S_A} "
            f"@ {event.minute:.0f}분"
        )

    # ─── 유틸 ─────────────────────────────────────────

    def _activate_cooldown(self, state: GameState) -> None:
        """쿨다운 활성화."""
        state.cooldown = True
        state.cooldown_until = time.monotonic() + self.cooldown_seconds

    @property
    def event_count(self) -> int:
        """처리한 이벤트 수."""
        return len(self._event_log)

    def get_event_log(self) -> list:
        """처리된 이벤트 로그 반환 (디버깅용)."""
        return list(self._event_log)
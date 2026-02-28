"""
Phase 3 Step 3.5: 추가시간 실시간 처리 (stoppage.py)

Phase 2에서 설정한 T_exp를 경기 진행에 따라 실시간으로 보정한다.

Goalserve는 공식 추가시간 발표 이벤트를 제공하지 않으므로,
minute 필드(예: "90+3")에서 추가시간 진입을 감지하고
롤링 방식으로 T를 업데이트한다.

상태 전이:
  Phase A (정규시간)  → minute > threshold → Phase C (롤링)
  Phase C: T = t + rolling_horizon (매 틱/폴링마다 갱신)

사용법:
  mgr = StoppageTimeManager(T_exp=95.0)
  T = mgr.update(current_minute=91.0, engine_phase="SECOND_HALF")
  # → Phase C 진입, T = 92.5 (91 + 1.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 상수
FIRST_HALF_THRESHOLD = 45.0   # 전반 정규 시간 경계
SECOND_HALF_THRESHOLD = 90.0  # 후반 정규 시간 경계
DEFAULT_ROLLING_HORIZON = 1.5 # 추가시간 롤링 여유 (분)


@dataclass
class StoppageTimeManager:
    """
    경기 종료 예정 시간 T를 실시간으로 관리하는 상태 머신.

    Attributes:
        T_exp:           Phase 2에서 계산한 기대 경기 시간 (기본 ~95분)
        rolling_horizon: Phase C에서 T = t + rolling_horizon (기본 1.5분)
        phase:           현재 상태 ("A" 또는 "C")
        T:               현재 적용 중인 경기 종료 예정 시간
        first_half_stoppage_entered: 전반 추가시간 진입 여부
        second_half_stoppage_entered: 후반 추가시간 진입 여부
    """
    T_exp: float = 95.0
    rolling_horizon: float = DEFAULT_ROLLING_HORIZON

    # 내부 상태
    phase: str = "A"
    T: float = 0.0
    first_half_stoppage_entered: bool = False
    second_half_stoppage_entered: bool = False

    def __post_init__(self):
        self.T = self.T_exp

    def update(self, current_minute: float, engine_phase: str) -> float:
        """
        현재 경기 시간과 엔진 상태를 받아 T를 업데이트한다.

        Args:
            current_minute: 현재 경기 시간 (분). 추가시간 포함된 값.
                           예: 93.0 = 후반 90+3
            engine_phase:   엔진 상태. "FIRST_HALF", "SECOND_HALF" 등.

        Returns:
            갱신된 T (경기 종료 예정 시간)

        동작:
          Phase A: T = T_exp 유지. minute > threshold 감지 시 Phase C 전환.
          Phase C: T = current_minute + rolling_horizon (매 호출마다 갱신).
        """
        if engine_phase == "FIRST_HALF":
            return self._handle_first_half(current_minute)
        elif engine_phase == "SECOND_HALF":
            return self._handle_second_half(current_minute)
        else:
            # HALFTIME, FINISHED 등 — T 변경 없음
            return self.T

    def on_stoppage_entered(self, half: str) -> None:
        """
        EventSource에서 STOPPAGE_ENTERED 이벤트 수신 시 호출.

        Args:
            half: "first" 또는 "second"
        """
        if half == "first" and not self.first_half_stoppage_entered:
            self.first_half_stoppage_entered = True
            logger.info("전반 추가시간 진입 감지")
        elif half == "second" and not self.second_half_stoppage_entered:
            self.second_half_stoppage_entered = True
            self.phase = "C"
            logger.info("후반 추가시간 진입 → Phase C (롤링 모드)")

    def reset_for_second_half(self) -> None:
        """
        후반 시작 시 호출. 전반 추가시간 상태를 리셋하고 Phase A로 복귀.
        """
        self.phase = "A"
        self.T = self.T_exp
        logger.info(f"후반 시작 → Phase A, T={self.T:.1f}")

    # ─── 내부 로직 ────────────────────────────────────

    def _handle_first_half(self, current_minute: float) -> float:
        """전반 추가시간 처리. T 자체는 변경하지 않음 (전반 종료 ≠ 경기 종료)."""
        # 전반 추가시간은 하프타임으로 이어지므로
        # T(경기 종료)에는 영향 없음. 감지 마킹만 수행.
        if current_minute > FIRST_HALF_THRESHOLD:
            if not self.first_half_stoppage_entered:
                self.first_half_stoppage_entered = True
                logger.info(f"전반 추가시간 자동 감지: minute={current_minute:.1f}")
        return self.T

    def _handle_second_half(self, current_minute: float) -> float:
        """후반 추가시간 처리. Phase C 진입 시 T를 롤링 업데이트."""
        if self.phase == "A":
            if current_minute > SECOND_HALF_THRESHOLD:
                # Phase A → Phase C 전환
                self.phase = "C"
                self.second_half_stoppage_entered = True
                self.T = current_minute + self.rolling_horizon
                logger.info(
                    f"후반 추가시간 감지 → Phase C: "
                    f"minute={current_minute:.1f}, T={self.T:.1f}"
                )
            # Phase A: T_exp 유지
            return self.T

        elif self.phase == "C":
            # Phase C: 매 호출마다 T 갱신
            new_T = current_minute + self.rolling_horizon
            # T는 단조 증가만 허용 (시간이 되돌아가지 않음)
            self.T = max(self.T, new_T)
            return self.T

        return self.T

    # ─── 편의 메서드 ──────────────────────────────────

    @property
    def is_stoppage_time(self) -> bool:
        """현재 추가시간 구간인지 여부"""
        return self.phase == "C"

    def __str__(self) -> str:
        return (
            f"StoppageTimeManager(phase={self.phase}, T={self.T:.1f}, "
            f"T_exp={self.T_exp:.1f}, "
            f"1H_stoppage={'Y' if self.first_half_stoppage_entered else 'N'}, "
            f"2H_stoppage={'Y' if self.second_half_stoppage_entered else 'N'})"
        )
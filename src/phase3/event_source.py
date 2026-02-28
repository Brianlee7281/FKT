"""
Phase 3: Event Source Abstraction (event_source.py)

라이브 엔진이 경기 이벤트를 수신하는 공통 인터페이스.
구체적인 데이터 소스(ReplaySource, GoalserveSource)는 이 ABC를 구현한다.

엔진(engine.py)은 EventSource.listen()만 호출하며,
이벤트가 DB 재생인지 Goalserve REST 폴링인지 알 필요 없다.

이벤트 흐름:
  [데이터 소스] → EventSource.listen() → NormalizedEvent → engine.handle_event()

구현체:
  - ReplaySource   : DB에서 과거 경기 이벤트 재생 (개발/테스트용)
  - GoalserveSource: Goalserve REST 폴링 + 상태 diff (라이브용)

사용법:
  # 개발 시 (과거 경기 재생)
  source = ReplaySource(db_path="data/kalshi_football.db", speed=60.0)

  # 라이브 시
  source = GoalserveSource(api_key="...", poll_interval=3)

  # 엔진은 동일한 코드
  async for event in source.listen():
      engine.handle_event(event)
"""

from __future__ import annotations

import enum
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 이벤트 타입 열거형
# ═════════════════════════════════════════════════════════

class EventType(str, enum.Enum):
    """
    Phase 3 엔진이 처리하는 이벤트 유형.

    각 이벤트가 엔진에 미치는 영향:
      GOAL               → S(t) 업데이트, ΔS 변경, δ(ΔS) 점프
      RED_CARD            → X(t) 마르코프 전이, γ 변경
      HALFTIME            → engine_phase → HALFTIME, 프라이싱/주문 동결
      SECOND_HALF_START   → engine_phase → SECOND_HALF, 프라이싱 재개
      STOPPAGE_ENTERED    → T를 롤링 모드로 전환 (T = t + 1.5)
      MATCH_END           → engine_phase → FINISHED, 최종 정산
    """
    GOAL = "goal"
    RED_CARD = "red_card"
    HALFTIME = "halftime"
    SECOND_HALF_START = "second_half_start"
    STOPPAGE_ENTERED = "stoppage_entered"
    MATCH_END = "match_end"


# ═════════════════════════════════════════════════════════
# 정규화된 이벤트 데이터 클래스
# ═════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NormalizedEvent:
    """
    모든 이벤트 소스에서 통일된 형식으로 변환된 경기 이벤트.

    엔진은 이 데이터 클래스만 처리한다.
    원본 데이터 소스(Goalserve JSON, DB row 등)는 raw 필드에 보존.

    Attributes:
        event_type: 이벤트 유형 (EventType enum)
        team:       이벤트 발생 팀 ("home" / "away").
                    HALFTIME, SECOND_HALF_START, MATCH_END에서는 빈 문자열.
        minute:     경기 시간 (분). 추가시간 포함 (예: 90+3 → 93.0)
        timestamp:  시스템 수신 시각 (time.time()). 지연 측정용.
        raw:        원본 데이터 (디버깅/로깅용). 소스마다 구조 다름.

    Examples:
        # 홈팀 골 (72분)
        NormalizedEvent(
            event_type=EventType.GOAL,
            team="home",
            minute=72.0,
            timestamp=1709100000.0,
            raw={"scorer": "Haaland", "assist": "De Bruyne"}
        )

        # 어웨이팀 퇴장 (55분)
        NormalizedEvent(
            event_type=EventType.RED_CARD,
            team="away",
            minute=55.0,
            timestamp=1709100000.0,
            raw={"player": "Casemiro", "reason": "Second Yellow card"}
        )

        # 추가시간 진입 (후반 91분)
        NormalizedEvent(
            event_type=EventType.STOPPAGE_ENTERED,
            team="",
            minute=91.0,
            timestamp=1709100000.0,
            raw={"half": "second", "detected_from": "minute_field"}
        )
    """
    event_type: EventType
    team: str               # "home", "away", or ""
    minute: float
    timestamp: float = field(default_factory=time.time)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """입력 유효성 검증"""
        # team 유효성
        if self.event_type in (EventType.GOAL, EventType.RED_CARD):
            if self.team not in ("home", "away"):
                raise ValueError(
                    f"{self.event_type.value} 이벤트는 team이 "
                    f"'home' 또는 'away'여야 합니다. 받은 값: '{self.team}'"
                )
        # minute 범위 (음수 방지, 200분 이상은 비정상)
        if self.minute < 0 or self.minute > 200:
            raise ValueError(
                f"minute 범위 초과: {self.minute} (0~200 허용)"
            )

    @property
    def is_discrete_shock(self) -> bool:
        """
        엔진의 μ 재계산을 트리거하는 불연속 충격인지 여부.
        GOAL, RED_CARD는 강도 함수 λ를 즉각 변경하므로 True.
        """
        return self.event_type in (EventType.GOAL, EventType.RED_CARD)

    @property
    def triggers_cooldown(self) -> bool:
        """
        이벤트 후 쿨다운(주문 차단)을 활성화해야 하는지 여부.
        모든 경기 상태 변경 이벤트에서 쿨다운 진입.
        HALFTIME, MATCH_END는 이미 주문이 차단되므로 쿨다운 불필요.
        """
        return self.event_type in (
            EventType.GOAL,
            EventType.RED_CARD,
            EventType.STOPPAGE_ENTERED,
        )

    def __str__(self) -> str:
        team_str = f" [{self.team}]" if self.team else ""
        return f"Event({self.event_type.value}{team_str} @ {self.minute:.1f}min)"


# ═════════════════════════════════════════════════════════
# 이벤트 소스 추상 베이스 클래스
# ═════════════════════════════════════════════════════════

class EventSource(ABC):
    """
    경기 이벤트를 제공하는 데이터 소스의 공통 인터페이스.

    라이프사이클:
        source = SomeEventSource(...)
        await source.connect(match_id)
        try:
            async for event in source.listen():
                engine.handle_event(event)
        finally:
            await source.disconnect()

    구현 시 주의사항:
        - listen()은 MATCH_END를 yield한 후 반드시 종료해야 한다.
        - connect()에서 데이터 소스 접근을 검증한다 (DB 존재, API 응답 등).
        - disconnect()에서 리소스를 정리한다 (DB 커넥션, HTTP 클라이언트 등).
        - 중복 이벤트를 필터링하는 것은 구현체의 책임이다.
    """

    @abstractmethod
    async def connect(self, match_id: str) -> None:
        """
        데이터 소스에 연결하고 지정 경기를 추적할 준비를 한다.

        Args:
            match_id: 경기 식별자.
                      ReplaySource: fixture_id (str로 변환된 정수)
                      GoalserveSource: Goalserve 내부 match ID

        Raises:
            ConnectionError: 데이터 소스에 접근할 수 없을 때
            ValueError: match_id가 유효하지 않을 때
        """
        ...

    @abstractmethod
    def listen(self) -> AsyncIterator[NormalizedEvent]:
        """
        이벤트 스트림을 비동기 이터레이터로 반환한다.

        Yields:
            NormalizedEvent: 정규화된 경기 이벤트.
                            시간순으로 yield되어야 한다.
                            마지막 이벤트는 반드시 MATCH_END.

        Notes:
            - ReplaySource: DB에서 읽은 이벤트를 speed에 맞춰 yield
            - GoalserveSource: 3초 폴링 → 상태 diff → 변화 시 yield
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        데이터 소스와의 연결을 정리한다.

        - ReplaySource: DB 커넥션 닫기
        - GoalserveSource: httpx.AsyncClient 닫기
        """
        ...

    # ─── 편의 메서드 (공통) ───

    async def __aenter__(self) -> EventSource:
        """async with 지원"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """async with 종료 시 자동 disconnect"""
        await self.disconnect()


# ═════════════════════════════════════════════════════════
# 유틸리티: minute 문자열 파싱
# ═════════════════════════════════════════════════════════

def parse_minute(minute_str: str) -> float:
    """
    Goalserve/API-Football의 minute 문자열을 float로 변환.

    Goalserve는 추가시간을 "90+3" 형태로 표시.
    API-Football은 minute=90, extra=3으로 분리 저장.
    이 함수는 두 형식을 모두 처리.

    Args:
        minute_str: "72", "90+3", "45+2" 등

    Returns:
        float: 실제 경기 시간 (분). "90+3" → 93.0

    Examples:
        >>> parse_minute("72")
        72.0
        >>> parse_minute("90+3")
        93.0
        >>> parse_minute("45+2")
        47.0
    """
    minute_str = str(minute_str).strip()

    if "+" in minute_str:
        parts = minute_str.split("+", 1)
        try:
            base = float(parts[0])
            extra_str = parts[1].strip() if len(parts) > 1 else ""
            extra = float(extra_str) if extra_str else 0.0
            return base + extra
        except (ValueError, IndexError):
            logger.warning(f"minute 파싱 실패: '{minute_str}', 0.0 반환")
            return 0.0
    else:
        try:
            return float(minute_str)
        except ValueError:
            logger.warning(f"minute 파싱 실패: '{minute_str}', 0.0 반환")
            return 0.0
"""
Phase 3: Replay Event Source (replay.py)

DB에 저장된 과거 경기 이벤트를 시간순으로 재생하여
Phase 3 엔진을 테스트하는 EventSource 구현체.

Goalserve 라이브 폴링 없이도 엔진의 전체 파이프라인을 검증할 수 있다:
  - 골 발생 시 S, ΔS, δ 업데이트 정상 여부
  - 퇴장 발생 시 X 마르코프 전이 정상 여부
  - 하프타임 동결/재개 전환
  - 추가시간 롤링 T 전환
  - 경기 종료 후 최종 정산

데이터 소스:
  - matches 테이블: fixture_id, home/away team_id, 스코어, elapsed_minutes
  - match_events 테이블: 골, 레드카드 등 분 단위 이벤트

사용법:
  source = ReplaySource(db_path="data/kalshi_football.db")
  await source.connect("1035068")   # fixture_id
  async for event in source.listen():
      print(event)                   # Event(goal [home] @ 23.0min)
  await source.disconnect()

  # 속도 조절
  source = ReplaySource(db_path="data/kalshi_football.db", speed=60.0)  # 60배속
  source = ReplaySource(db_path="data/kalshi_football.db", speed=0.0)   # 즉시 (대기 없음)
"""

from __future__ import annotations

import asyncio
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from src.phase3.event_source import (
    EventSource,
    EventType,
    NormalizedEvent,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────

FIRST_HALF_REGULAR_END = 45    # 전반 정규 시간 종료 (분)
SECOND_HALF_API_START = 46     # API-Football에서 후반 이벤트 시작 분
DEFAULT_FIRST_HALF_STOPPAGE = 2
DEFAULT_SECOND_HALF_STOPPAGE = 3
DEFAULT_ELAPSED = 95           # elapsed_minutes 없을 때 기본값


# ─────────────────────────────────────────────────────────
# DB에서 읽은 원시 데이터 구조
# ─────────────────────────────────────────────────────────

@dataclass
class _MatchInfo:
    """matches 테이블에서 읽은 경기 메타데이터"""
    fixture_id: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_goals_ht: Optional[int]
    away_goals_ht: Optional[int]
    home_goals_ft: Optional[int]
    away_goals_ft: Optional[int]
    elapsed_minutes: Optional[int]


@dataclass
class _RawEvent:
    """match_events 테이블에서 읽은 원시 이벤트"""
    event_minute: int
    event_extra: Optional[int]
    event_type: str         # "Goal", "Card", "subst"
    event_detail: str       # "Normal Goal", "Own Goal", "Red Card", "Second Yellow card"
    team_id: int
    team_name: str
    player_name: Optional[str]

    @property
    def actual_minute(self) -> float:
        """실제 경기 시각. 90+3이면 minute=90, extra=3 → 93.0"""
        base = self.event_minute or 0
        extra = self.event_extra or 0
        return float(base + extra)

    @property
    def is_first_half(self) -> bool:
        return self.event_minute <= FIRST_HALF_REGULAR_END

    @property
    def is_goal(self) -> bool:
        return self.event_type == "Goal"

    @property
    def is_red_card(self) -> bool:
        return (
            self.event_type == "Card"
            and self.event_detail in ("Red Card", "Second Yellow card")
        )


# ═════════════════════════════════════════════════════════
# ReplaySource
# ═════════════════════════════════════════════════════════

class ReplaySource(EventSource):
    """
    DB에서 과거 경기 이벤트를 시간순으로 재생하는 EventSource.

    Args:
        db_path: SQLite 데이터베이스 경로 (data/kalshi_football.db)
        speed:   재생 속도 배율
                 0.0  → 즉시 재생 (대기 없음, 단위 테스트용)
                 1.0  → 실시간 (90분 소요)
                 60.0 → 60배속 (90초 만에 한 경기)

    이벤트 생성 규칙:
        1. 골: match_events에서 event_type="Goal" 필터.
           자책골(Own Goal)은 상대팀 득점으로 변환.
        2. 레드카드: event_type="Card" AND detail in ("Red Card", "Second Yellow card")
        3. 하프타임: 전반 마지막 이벤트 시각 기준으로 합성.
           DB에 명시적 하프타임 이벤트가 없으므로 자동 생성.
        4. 후반 시작: 하프타임 직후 합성.
        5. 추가시간 진입: minute > 45(전반) 또는 > 90(후반) 감지 시 합성.
        6. 경기 종료: elapsed_minutes 기반으로 합성.

    Notes:
        - preprocessor.py의 시간 변환 로직과 달리, 여기서는
          API-Football 원본 시간(actual_minute)을 그대로 사용한다.
          실효 시간 변환(하프타임 제거)은 engine.py의 책임이다.
        - match_events의 시간순이 실제 경기 시간순과 일치한다고 가정한다.
    """

    def __init__(
        self,
        db_path: str | Path = "data/kalshi_football.db",
        speed: float = 0.0,
    ):
        self.db_path = Path(db_path)
        self.speed = speed

        self._conn: Optional[sqlite3.Connection] = None
        self._match: Optional[_MatchInfo] = None
        self._events: List[NormalizedEvent] = []
        self._connected = False

    # ─── EventSource ABC 구현 ─────────────────────────────

    async def connect(self, match_id: str) -> None:
        """
        DB에서 지정 경기의 이벤트를 로드하고 NormalizedEvent 시퀀스를 구성한다.

        Args:
            match_id: fixture_id (문자열). 예: "1035068"

        Raises:
            ConnectionError: DB 파일이 없을 때
            ValueError: fixture_id가 DB에 없을 때
        """
        fixture_id = int(match_id)

        # DB 연결
        if not self.db_path.exists():
            raise ConnectionError(f"DB 파일 없음: {self.db_path}")

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # 경기 메타데이터 로드
        self._match = self._load_match(fixture_id)
        if self._match is None:
            raise ValueError(f"fixture_id {fixture_id}가 DB에 없습니다")

        # 원시 이벤트 로드
        raw_events = self._load_events(fixture_id)

        # NormalizedEvent 시퀀스 구성
        self._events = self._build_event_sequence(raw_events)

        self._connected = True
        logger.info(
            f"ReplaySource 연결 완료: fixture={fixture_id}, "
            f"{self._match.home_team_name} vs {self._match.away_team_name}, "
            f"이벤트 {len(self._events)}개, speed={self.speed}x"
        )

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        """
        구성된 이벤트를 시간순으로 yield한다.
        speed > 0이면 이벤트 간 시간 간격만큼 대기한다.
        """
        if not self._connected:
            raise RuntimeError("connect()를 먼저 호출하세요")

        prev_minute = 0.0

        for event in self._events:
            # 속도 조절: 이벤트 간 경기 시간 차이만큼 대기
            if self.speed > 0:
                gap_minutes = max(0, event.minute - prev_minute)
                gap_seconds = (gap_minutes * 60) / self.speed
                if gap_seconds > 0:
                    await asyncio.sleep(gap_seconds)

            prev_minute = event.minute
            yield event

    async def disconnect(self) -> None:
        """DB 연결 닫기"""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._connected = False
        self._events = []
        self._match = None

    # ─── DB 로드 ──────────────────────────────────────────

    def _load_match(self, fixture_id: int) -> Optional[_MatchInfo]:
        """matches 테이블에서 경기 메타데이터 로드"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT fixture_id, home_team_id, home_team_name,
                   away_team_id, away_team_name,
                   home_goals_ht, away_goals_ht,
                   home_goals_ft, away_goals_ft,
                   elapsed_minutes
            FROM matches WHERE fixture_id = ?
        """, (fixture_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return _MatchInfo(
            fixture_id=row["fixture_id"],
            home_team_id=row["home_team_id"],
            home_team_name=row["home_team_name"],
            away_team_id=row["away_team_id"],
            away_team_name=row["away_team_name"],
            home_goals_ht=row["home_goals_ht"],
            away_goals_ht=row["away_goals_ht"],
            home_goals_ft=row["home_goals_ft"],
            away_goals_ft=row["away_goals_ft"],
            elapsed_minutes=row["elapsed_minutes"],
        )

    def _load_events(self, fixture_id: int) -> List[_RawEvent]:
        """match_events 테이블에서 골 + 레드카드 이벤트 로드"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT event_minute, event_extra, event_type, event_detail,
                   team_id, team_name, player_name
            FROM match_events
            WHERE fixture_id = ?
              AND (
                  event_type = 'Goal'
                  OR (event_type = 'Card'
                      AND event_detail IN ('Red Card', 'Second Yellow card'))
              )
            ORDER BY event_minute, event_extra
        """, (fixture_id,))

        events = []
        for row in cursor.fetchall():
            events.append(_RawEvent(
                event_minute=row["event_minute"] or 0,
                event_extra=row["event_extra"],
                event_type=row["event_type"] or "",
                event_detail=row["event_detail"] or "",
                team_id=row["team_id"] or 0,
                team_name=row["team_name"] or "",
                player_name=row["player_name"],
            ))

        return events

    # ─── 이벤트 시퀀스 구성 ────────────────────────────────

    def _team_side(self, team_id: int) -> str:
        """team_id → 'home' 또는 'away'"""
        if team_id == self._match.home_team_id:
            return "home"
        return "away"

    def _build_event_sequence(
        self, raw_events: List[_RawEvent]
    ) -> List[NormalizedEvent]:
        """
        원시 이벤트를 NormalizedEvent 시퀀스로 변환한다.

        처리 순서:
        1. 골/레드카드를 NormalizedEvent로 변환
        2. 전반 추가시간 진입 이벤트 합성 (있으면)
        3. 하프타임 / 후반 시작 합성
        4. 후반 추가시간 진입 이벤트 합성 (있으면)
        5. 경기 종료 합성
        6. 시간순 정렬 (같은 분이면: 퇴장 → 골 → 구조 이벤트)
        """
        events: List[NormalizedEvent] = []
        match = self._match

        # ── 1. 골/레드카드 변환 ────────────────────────────

        for raw in raw_events:
            if raw.is_goal:
                events.append(self._convert_goal(raw))
            elif raw.is_red_card:
                events.append(self._convert_red_card(raw))

        # ── 2. 전반 추가시간 추정 ──────────────────────────

        first_half_max = FIRST_HALF_REGULAR_END
        for raw in raw_events:
            if raw.is_first_half and raw.actual_minute > first_half_max:
                first_half_max = raw.actual_minute

        first_half_stoppage = first_half_max - FIRST_HALF_REGULAR_END
        if first_half_stoppage <= 0:
            first_half_stoppage = DEFAULT_FIRST_HALF_STOPPAGE

        first_half_end = FIRST_HALF_REGULAR_END + first_half_stoppage

        # 전반 추가시간 진입
        if first_half_stoppage > 0:
            events.append(NormalizedEvent(
                event_type=EventType.STOPPAGE_ENTERED,
                team="",
                minute=FIRST_HALF_REGULAR_END + 0.5,  # 45분 직후
                raw={"half": "first", "source": "replay_synthesized"},
            ))

        # ── 3. 하프타임 / 후반 시작 ────────────────────────

        events.append(NormalizedEvent(
            event_type=EventType.HALFTIME,
            team="",
            minute=first_half_end,
            raw={"source": "replay_synthesized"},
        ))

        events.append(NormalizedEvent(
            event_type=EventType.SECOND_HALF_START,
            team="",
            minute=first_half_end + 0.1,  # 하프타임 직후 (정렬용 미세 오프셋)
            raw={"source": "replay_synthesized"},
        ))

        # ── 4. 후반 추가시간 ───────────────────────────────

        # 후반 이벤트 중 90분 이후가 있는지 확인
        second_half_events = [
            e for e in raw_events if not e.is_first_half
        ]
        has_second_half_stoppage = any(
            e.actual_minute > 90 for e in second_half_events
        )

        if has_second_half_stoppage:
            events.append(NormalizedEvent(
                event_type=EventType.STOPPAGE_ENTERED,
                team="",
                minute=90.5,  # 90분 직후
                raw={"half": "second", "source": "replay_synthesized"},
            ))

        # ── 5. 경기 종료 ──────────────────────────────────

        # elapsed_minutes로부터 종료 시각 결정
# 경기 종료 시각 결정
        # elapsed_minutes와 마지막 이벤트 시각 중 더 큰 값 + 여유분
        last_event_minute = max(
            (e.actual_minute for e in raw_events), default=0.0
        )

        if match.elapsed_minutes and match.elapsed_minutes > 0:
            end_minute = max(float(match.elapsed_minutes), last_event_minute + 1.0)
        else:
            end_minute = last_event_minute + 1.0 if last_event_minute > 0 else DEFAULT_ELAPSED
            
        events.append(NormalizedEvent(
            event_type=EventType.MATCH_END,
            team="",
            minute=end_minute,
            raw={
                "final_score": f"{match.home_goals_ft}-{match.away_goals_ft}",
                "source": "replay_synthesized",
            },
        ))

        # ── 6. 시간순 정렬 ────────────────────────────────
        # 같은 분: 퇴장(0) → 골(1) → 추가시간(2) → 하프타임(3) → 후반시작(4) → 종료(5)
        sort_priority = {
            EventType.RED_CARD: 0,
            EventType.GOAL: 1,
            EventType.STOPPAGE_ENTERED: 2,
            EventType.HALFTIME: 3,
            EventType.SECOND_HALF_START: 4,
            EventType.MATCH_END: 5,
        }

        events.sort(key=lambda e: (e.minute, sort_priority.get(e.event_type, 9)))

        return events

    def _convert_goal(self, raw: _RawEvent) -> NormalizedEvent:
        """
        원시 골 이벤트를 NormalizedEvent로 변환.
        자책골(Own Goal)은 상대팀 득점으로 변환한다.
        """
        if raw.event_detail == "Own Goal":
            # 자책골: team_id의 팀이 자책 → 상대팀 득점
            side = self._team_side(raw.team_id)
            actual_team = "away" if side == "home" else "home"
        else:
            actual_team = self._team_side(raw.team_id)

        return NormalizedEvent(
            event_type=EventType.GOAL,
            team=actual_team,
            minute=raw.actual_minute,
            raw={
                "player": raw.player_name or "",
                "detail": raw.event_detail,
                "original_team": raw.team_name,
                "source": "replay_db",
            },
        )

    def _convert_red_card(self, raw: _RawEvent) -> NormalizedEvent:
        """원시 카드 이벤트를 RED_CARD NormalizedEvent로 변환"""
        return NormalizedEvent(
            event_type=EventType.RED_CARD,
            team=self._team_side(raw.team_id),
            minute=raw.actual_minute,
            raw={
                "player": raw.player_name or "",
                "detail": raw.event_detail,  # "Red Card" or "Second Yellow card"
                "source": "replay_db",
            },
        )

    # ─── 편의 메서드 ──────────────────────────────────────

    @property
    def match_info(self) -> Optional[_MatchInfo]:
        """연결된 경기의 메타데이터 (디버깅용)"""
        return self._match

    @property
    def event_count(self) -> int:
        """구성된 이벤트 수"""
        return len(self._events)

    def summary(self) -> str:
        """연결된 경기의 이벤트 요약 문자열"""
        if not self._connected or not self._match:
            return "ReplaySource: 미연결"

        m = self._match
        lines = [
            f"ReplaySource: {m.home_team_name} vs {m.away_team_name}",
            f"  fixture_id: {m.fixture_id}",
            f"  FT: {m.home_goals_ft}-{m.away_goals_ft}",
            f"  HT: {m.home_goals_ht}-{m.away_goals_ht}",
            f"  이벤트: {self.event_count}개",
            "  ---",
        ]
        for evt in self._events:
            lines.append(f"  {evt}")

        return "\n".join(lines)
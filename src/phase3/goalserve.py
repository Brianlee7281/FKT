"""
Phase 3: Goalserve REST 폴링 기반 라이브 이벤트 소스 (goalserve.py)

3초 간격으로 Goalserve /soccernew/live 엔드포인트를 폴링하여
이전 상태와의 diff로 이벤트를 감지한다.

Goalserve JSON 구조:
  scores.category[].matches.match[]:
    @status:  분 숫자 (예: "85") / "HT" / "FT"
    @timer:   분 숫자
    localteam:  { @name, @goals, @id }
    visitorteam: { @name, @goals, @id }
    events.event[]:
      @type:      "goal" / "redcard" / "yellowred" / "yellowcard" / "subst" / "var"
      @minute:    "45"
      @extra_min: "3"  (추가시간이면)
      @team:      "localteam" / "visitorteam"
      @player:    "M. Salah"
      @result:    "[1 - 0]"
      @eventid:   고유 이벤트 ID

사용법:
  source = GoalserveSource(api_key="...", match_id="6678380", poll_interval=3.0)
  async with source:
      async for event in source.listen():
          print(event)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Dict, List, Optional, Set

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None  # type: ignore

from src.phase3.event_source import EventSource, EventType, NormalizedEvent

logger = logging.getLogger(__name__)

# Goalserve 상수
GOALSERVE_BASE_URL = "http://www.goalserve.com/getfeed"
DEFAULT_POLL_INTERVAL = 3.0  # 초


class GoalserveSource(EventSource):
    """
    Goalserve REST 폴링 기반 라이브 이벤트 소스.

    3초마다 /soccernew/live?json=1 을 폴링하여
    이전 스냅샷과의 diff로 이벤트를 감지한다.

    Args:
        api_key:        Goalserve API 키 (URL에 포함됨)
        match_id:       Goalserve 경기 ID (@id 필드)
        poll_interval:  폴링 간격 (초, 기본 3.0)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        match_id: Optional[str] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self.api_key = api_key or os.getenv("GOALSERVE_API_KEY", "")
        self.match_id = match_id or ""
        self.poll_interval = poll_interval

        # HTTP 클라이언트
        self._client: Optional[httpx.AsyncClient] = None

        # 상태 추적
        self._seen_event_ids: Set[str] = set()
        self._prev_status: Optional[str] = None
        self._prev_timer: Optional[str] = None
        self._prev_home_goals: int = 0
        self._prev_away_goals: int = 0
        self._stoppage_entered: Dict[str, bool] = {"first": False, "second": False}
        self._halftime_sent: bool = False
        self._second_half_sent: bool = False
        self._match_end_sent: bool = False

        # 경기 정보 (connect 시 채움)
        self._home_team_id: str = ""
        self._away_team_id: str = ""
        self._home_team_name: str = ""
        self._away_team_name: str = ""

    # ─── EventSource 인터페이스 ─────────────────────────

    async def connect(self, match_id: str) -> None:
        """HTTP 클라이언트 초기화 + 첫 폴링으로 경기 정보 수집."""
        self.match_id = match_id
        if HAS_HTTPX:
            self._client = httpx.AsyncClient(timeout=60.0)
        else:
            self._client = None
            logger.warning("httpx 미설치 — 실제 폴링 불가 (테스트 전용 모드)")
        logger.info(f"GoalserveSource: 연결 match_id={match_id}")

        # 첫 폴링으로 경기 정보 확인 (최대 3회 재시도)
        match_data = None
        for attempt in range(3):
            match_data = await self._poll_match()
            if match_data:
                break
            logger.warning(f"GoalserveSource: 연결 시도 {attempt+1}/3 실패, 재시도...")
            await asyncio.sleep(2)
        if match_data:
            self._home_team_name = match_data.get("localteam", {}).get("@name", "Home")
            self._away_team_name = match_data.get("visitorteam", {}).get("@name", "Away")
            self._home_team_id = match_data.get("localteam", {}).get("@id", "")
            self._away_team_id = match_data.get("visitorteam", {}).get("@id", "")

            # 기존 이벤트 마킹 (이미 발생한 이벤트는 무시)
            self._seed_existing_events(match_data)

            logger.info(
                f"GoalserveSource: {self._home_team_name} vs {self._away_team_name}"
            )
        else:
            logger.warning(f"GoalserveSource: match_id={match_id} 못 찾음")

    async def disconnect(self) -> None:
        """HTTP 클라이언트 정리."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("GoalserveSource: 연결 해제")

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        """
        3초마다 폴링 + diff로 이벤트 생성.

        Yields:
            NormalizedEvent (GOAL, RED_CARD, HALFTIME, SECOND_HALF_START,
                           STOPPAGE_ENTERED, MATCH_END)
        """
        while not self._match_end_sent:
            match_data = await self._poll_match()

            if match_data:
                # 이벤트 diff로 새 이벤트 감지
                async for event in self._diff(match_data):
                    yield event
            else:
                logger.info("GoalserveSource: 경기 데이터 없음 (FT 또는 네트워크 오류)")

            await asyncio.sleep(self.poll_interval)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    # ─── 폴링 ─────────────────────────────────────────

    async def _poll_match(self) -> Optional[Dict]:
        """Goalserve live 엔드포인트를 폴링하여 해당 경기 데이터를 찾는다."""
        if not self._client:
            return None

        url = f"{GOALSERVE_BASE_URL}/{self.api_key}/soccernew/live?json=1"

        for attempt in range(3):
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
                return self._find_match(resp.json(), self.match_id)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                logger.warning(f"GoalserveSource 폴링 실패 (3회 재시도 후): {type(e).__name__}: {e}")
                return None

    @staticmethod
    def _find_match(data: Dict, match_id: str) -> Optional[Dict]:
        """JSON 응답에서 match_id에 해당하는 경기를 찾는다."""
        categories = data.get("scores", {}).get("category", [])
        if isinstance(categories, dict):
            categories = [categories]
        if not categories:
            return None

        for cat in categories:
            matches_container = cat.get("matches")
            if not matches_container or not isinstance(matches_container, dict):
                continue
            matches = matches_container.get("match", [])
            if matches is None:
                continue
            if isinstance(matches, dict):
                matches = [matches]
            for m in matches:
                if m.get("@id") == match_id:
                    return m

        return None

    # ─── Diff 로직 ────────────────────────────────────

    async def _diff(self, match_data: Dict) -> AsyncIterator[NormalizedEvent]:
        """
        이전 폴링 상태와 현재를 비교하여 새 이벤트를 yield한다.

        감지 순서:
          1. 새로운 이벤트 (eventid 기반 diff)
          2. 상태 변화 (HT, FT, 추가시간 진입)
        """
        status = match_data.get("@status", "")
        timer = match_data.get("@timer", "")
        home_goals = int(match_data.get("localteam", {}).get("@goals", "0"))
        away_goals = int(match_data.get("visitorteam", {}).get("@goals", "0"))

        # ── 1. 이벤트 리스트 diff ────────────────────

        events_data = match_data.get("events")
        if events_data and events_data is not None:
            event_list = events_data.get("event", [])
            if isinstance(event_list, dict):
                event_list = [event_list]

            for evt in event_list:
                event_id = evt.get("@eventid", "")
                if event_id and event_id not in self._seen_event_ids:
                    self._seen_event_ids.add(event_id)

                    normalized = self._parse_event(evt)
                    if normalized:
                        yield normalized

        # ── 2. 하프타임 감지 ─────────────────────────

        if status == "HT" and not self._halftime_sent and not self._second_half_sent:
            self._halftime_sent = True
            minute = self._parse_minute_from_timer(self._prev_status or "45")
            yield NormalizedEvent(
                event_type=EventType.HALFTIME,
                team="",
                minute=max(minute, 45.0),
                raw={"status": status},
            )

        # ── 3. 후반 시작 감지 ────────────────────────

        if (self._prev_status == "HT" and status != "HT"
                and status != "FT" and not self._second_half_sent):
            self._second_half_sent = True
            yield NormalizedEvent(
                event_type=EventType.SECOND_HALF_START,
                team="",
                minute=45.0,
                raw={"status": status},
            )

        # ── 4. 추가시간 진입 감지 ────────────────────

        try:
            current_minute = self._parse_minute_from_timer(timer)
        except Exception:
            current_minute = 0.0

        if (current_minute > 45 and status != "HT" and status != "FT"
                and not self._stoppage_entered["first"]
                and not self._halftime_sent):
            self._stoppage_entered["first"] = True
            yield NormalizedEvent(
                event_type=EventType.STOPPAGE_ENTERED,
                team="",
                minute=current_minute,
                raw={"half": "first"},
            )

        if (current_minute > 90 and status != "FT"
                and not self._stoppage_entered["second"]):
            self._stoppage_entered["second"] = True
            yield NormalizedEvent(
                event_type=EventType.STOPPAGE_ENTERED,
                team="",
                minute=current_minute,
                raw={"half": "second"},
            )

        # ── 5. 경기 종료 감지 ────────────────────────

        if status == "FT" and not self._match_end_sent:
            self._match_end_sent = True
            yield NormalizedEvent(
                event_type=EventType.MATCH_END,
                team="",
                minute=max(current_minute, 90.0),
                raw={
                    "status": "FT",
                    "final_score": f"{home_goals}-{away_goals}",
                },
            )

        # 상태 갱신
        self._prev_status = status
        self._prev_timer = timer
        self._prev_home_goals = home_goals
        self._prev_away_goals = away_goals

    # ─── 이벤트 파싱 ──────────────────────────────────

    def _parse_event(self, evt: Dict) -> Optional[NormalizedEvent]:
        """Goalserve 이벤트를 NormalizedEvent로 변환한다."""
        evt_type = evt.get("@type", "")
        team_raw = evt.get("@team", "")
        minute_str = evt.get("@minute", "0")
        extra_min = evt.get("@extra_min", "")
        player = evt.get("@player", "")

        # 분 계산
        try:
            minute = float(minute_str)
            extra = 0
            if extra_min:
                extra = int(extra_min)
                minute += extra  # 45+2 → 47.0 (실효 시간)
        except (ValueError, TypeError):
            minute = 0.0
            extra = 0

        # 팀 매핑
        team = self._map_team(team_raw)

        # 표시용 분 (45+2 형태)
        if extra:
            display_minute = f"{minute_str}+{extra}"
        else:
            display_minute = minute_str

        # 이벤트 타입 매핑
        if evt_type == "goal":
            return NormalizedEvent(
                event_type=EventType.GOAL,
                team=team,
                minute=minute,
                raw={
                    "player": player,
                    "result": evt.get("@result", ""),
                    "own_goal": "(o.g.)" in player,
                    "eventid": evt.get("@eventid", ""),
                    "display_minute": display_minute,
                },
            )
        elif evt_type in ("redcard", "yellowred"):
            return NormalizedEvent(
                event_type=EventType.RED_CARD,
                team=team,
                minute=minute,
                raw={
                    "player": player,
                    "detail": evt_type,
                    "eventid": evt.get("@eventid", ""),
                    "display_minute": display_minute,
                },
            )

        # yellowcard, subst, var → 무시 (우리 모델에 영향 없음)
        return None

    def _map_team(self, team_raw: str) -> str:
        """Goalserve 팀 식별자를 'home'/'away'로 변환."""
        if team_raw == "localteam":
            return "home"
        elif team_raw == "visitorteam":
            return "away"
        return ""

    def _seed_existing_events(self, match_data: Dict) -> None:
        """이미 발생한 이벤트의 ID를 마킹하여 중복 방지."""
        events_data = match_data.get("events")
        if events_data and events_data is not None:
            event_list = events_data.get("event", [])
            if isinstance(event_list, dict):
                event_list = [event_list]
            for evt in event_list:
                eid = evt.get("@eventid", "")
                if eid:
                    self._seen_event_ids.add(eid)

        # 현재 상태 초기화
        status = match_data.get("@status", "")
        self._prev_status = status
        self._prev_timer = match_data.get("@timer", "")
        self._prev_home_goals = int(match_data.get("localteam", {}).get("@goals", "0"))
        self._prev_away_goals = int(match_data.get("visitorteam", {}).get("@goals", "0"))

        # 이미 HT를 지났으면 마킹
        if status != "HT" and self._prev_home_goals + self._prev_away_goals > 0:
            # 후반이면 이미 하프타임 지남
            try:
                timer = self._parse_minute_from_timer(match_data.get("@timer", "0"))
                if timer > 45:
                    self._halftime_sent = True
                    self._second_half_sent = True
                    self._stoppage_entered["first"] = True
            except (ValueError, TypeError):
                pass
        if status == "HT":
            self._halftime_sent = True

    @staticmethod
    def _parse_minute_from_timer(timer_str: str) -> float:
        """타이머 문자열을 분으로 변환. '45+', '90+' 형태 처리."""
        if not timer_str:
            return 0.0
        try:
            # "45+", "90+" → 45.5, 90.5
            cleaned = timer_str.replace("+", ".5")
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0

    # ─── 편의 메서드 ──────────────────────────────────

    @staticmethod
    def find_all_live_matches(data: Dict) -> List[Dict]:
        """
        Goalserve 응답에서 현재 진행 중인 모든 경기를 찾는다.

        Returns:
            경기 데이터 리스트 (각각 league, match 정보 포함)
        """
        results = []
        categories = data.get("scores", {}).get("category", [])
        if isinstance(categories, dict):
            categories = [categories]

        for cat in categories:
            league = cat.get("@name", "")
            matches_container = cat.get("matches")
            if not matches_container or not isinstance(matches_container, dict):
                continue
            matches = matches_container.get("match", [])
            if matches is None:
                continue
            if isinstance(matches, dict):
                matches = [matches]
            for m in matches:
                status = m.get("@status", "")
                # 숫자 상태 또는 HT = 진행 중
                if status not in ("FT", "Postp.", "Canc.", "Awarded", ""):
                    results.append({
                        "league": league,
                        "match_id": m.get("@id", ""),
                        "home": m.get("localteam", {}).get("@name", ""),
                        "away": m.get("visitorteam", {}).get("@name", ""),
                        "score": f"{m.get('localteam', {}).get('@goals', 0)}-{m.get('visitorteam', {}).get('@goals', 0)}",
                        "status": status,
                        "timer": m.get("@timer", ""),
                    })

        return results
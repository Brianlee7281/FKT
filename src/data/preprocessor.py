"""
Phase 1.1: 시계열 이벤트 분할 및 구간 데이터화 (Data Engineering).

API-Football의 점(Point) 이벤트를 λ가 일정한 연속 구간(Interval)으로 변환한다.
강도 함수 λ는 (X(t), ΔS(t))에 의존하므로,
이 두 변수 중 하나라도 변하는 시점에서 구간을 잘라야 한다.

입력:
  - data/kalshi_football.db의 matches, match_events 테이블
    (api_football.py가 다운로드한 데이터)

출력:
  - intervals 테이블: NLL 적분항의 입력
  - goal_events 테이블: NLL 점 이벤트항의 입력
  - match_meta 테이블: 경기별 메타 (T_m, 기저함수 경계 등)

사용법:
  # 전체 처리
  python -m src.data.preprocessor

  # 특정 리그-시즌만
  python -m src.data.preprocessor --league 39 --season 2023

  # 처리 결과 검증
  python -m src.data.preprocessor --verify

  # 특정 경기의 구간 분할 결과를 상세히 출력 (디버깅용)
  python -m src.data.preprocessor --debug-match 868124
"""

import sqlite3
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

DB_PATH = Path("data/kalshi_football.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocessor.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# 기저함수 빈 길이 (분)
BASIS_BIN_MINUTES = 15
NUM_BASIS = 6   # 전반 3개 + 후반 3개

# 하프타임 경계 (API-Football 기준)
FIRST_HALF_REGULAR_END = 45   # 정규 전반 종료 분
SECOND_HALF_API_START = 46    # API-Football의 후반 시작 분


# ─────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────

@dataclass
class RawEvent:
    """DB에서 읽은 원시 이벤트"""
    fixture_id: int
    event_minute: int
    event_extra: Optional[int]
    event_type: str          # "Goal", "Card", "subst"
    event_detail: str        # "Normal Goal", "Own Goal", "Red Card" 등
    team_id: int
    team_name: str

    @property
    def actual_minute(self) -> float:
        """
        실제 경기 시각 (분).
        API-Football에서 90+3이면 minute=90, extra=3 → 93분.
        """
        base = self.event_minute or 0
        extra = self.event_extra or 0
        return float(base + extra)


@dataclass
class MatchInfo:
    """DB에서 읽은 경기 메타데이터"""
    fixture_id: int
    league_id: int
    season: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_goals_ft: int
    away_goals_ft: int
    home_goals_ht: Optional[int]
    away_goals_ht: Optional[int]
    elapsed_minutes: Optional[int]


@dataclass
class Interval:
    """
    λ가 일정한 하나의 구간.

    NLL 적분항에 사용:
    μ_k = exp(a_m + b_{basis_idx} + γ_{state_X} + δ(delta_S)) × duration
    """
    fixture_id: int
    t_start: float           # 구간 시작 (실효 플레이 분)
    t_end: float             # 구간 종료
    state_X: int             # 마르코프 상태 0~3
    delta_S: int             # 홈 기준 득점차
    basis_idx: int           # 기저함수 인덱스 0~5
    is_halftime: bool        # True면 NLL에서 제외
    T_m: float               # 경기 실제 종료 시간 (실효 분)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class GoalEvent:
    """
    골 시점의 NLL 점 이벤트 기여.

    ln λ(t_g) = a_m + b_{basis_idx} + γ_{state_X} + δ(delta_S_before)

    인과관계 주의: delta_S_before는 골 직전의 ΔS!
    """
    fixture_id: int
    t_eff: float             # 실효 플레이 시각
    team: str                # "home" 또는 "away" (실제 득점팀)
    state_X: int             # 골 시점의 마르코프 상태
    delta_S_before: int      # 골 직전의 ΔS (인과관계 핵심!)
    basis_idx: int           # 기저함수 인덱스 0~5


@dataclass
class MatchMeta:
    """경기별 메타데이터 (기저함수 경계 포함)"""
    fixture_id: int
    T_m: float               # 실효 경기 종료 시각
    first_half_end: float    # 전반 종료 실효 시각 (추가시간 포함)
    second_half_start: float # 후반 시작 실효 시각
    basis_boundaries: List[float]  # 7개 경계점 [0, 15, 30, fh_end, fh_end+15, ...]
    total_goals: int
    total_red_cards: int


# ─────────────────────────────────────────────────────────
# 출력 테이블 스키마
# ─────────────────────────────────────────────────────────

OUTPUT_SCHEMA_SQL = """
-- Phase 1 NLL 입력: 구간 데이터
CREATE TABLE IF NOT EXISTS intervals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    t_start         REAL NOT NULL,        -- 실효 플레이 시간 (분)
    t_end           REAL NOT NULL,
    duration        REAL NOT NULL,        -- t_end - t_start
    state_X         INTEGER NOT NULL,     -- 마르코프 상태 0~3
    delta_S         INTEGER NOT NULL,     -- 홈 기준 득점차
    basis_idx       INTEGER NOT NULL,     -- 기저함수 인덱스 0~5
    is_halftime     INTEGER NOT NULL DEFAULT 0,  -- 1이면 NLL 제외
    T_m             REAL NOT NULL,        -- 경기 실효 종료 시간

    FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);

-- Phase 1 NLL 입력: 골 이벤트
CREATE TABLE IF NOT EXISTS goal_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    t_eff           REAL NOT NULL,        -- 실효 플레이 시간 (분)
    team            TEXT NOT NULL,        -- "home" or "away"
    state_X         INTEGER NOT NULL,     -- 골 시점 마르코프 상태
    delta_S_before  INTEGER NOT NULL,     -- 골 직전 ΔS (인과관계!)
    basis_idx       INTEGER NOT NULL,     -- 기저함수 인덱스

    FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);

-- 경기별 메타데이터 (기저함수 경계 등)
CREATE TABLE IF NOT EXISTS match_meta (
    fixture_id          INTEGER PRIMARY KEY,
    T_m                 REAL NOT NULL,
    first_half_end      REAL NOT NULL,
    second_half_start   REAL NOT NULL,
    basis_boundaries    TEXT NOT NULL,     -- JSON 배열 [0, 15, 30, ...]
    total_goals         INTEGER NOT NULL,
    total_red_cards     INTEGER NOT NULL,

    FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);

CREATE INDEX IF NOT EXISTS idx_intervals_fixture
    ON intervals(fixture_id);
CREATE INDEX IF NOT EXISTS idx_intervals_basis
    ON intervals(basis_idx, is_halftime);
CREATE INDEX IF NOT EXISTS idx_goals_fixture
    ON goal_events(fixture_id);
"""


# ─────────────────────────────────────────────────────────
# 핵심 로직: 단일 경기 처리
# ─────────────────────────────────────────────────────────

def process_match(
    match: MatchInfo,
    raw_events: List[RawEvent],
) -> Tuple[List[Interval], List[GoalEvent], MatchMeta]:
    """
    한 경기의 원시 이벤트를 구간 + 골 이벤트로 변환한다.

    Phase 1.1 설계 문서의 구간 분할 규칙:
    - 레드카드 → X 변경 → 분할
    - 골 → ΔS 변경 → 분할
    - 하프타임 → 적분 제외 마킹
    - 경기 종료 → 마지막 구간 닫기

    시간 변환:
    - API-Football의 시간(45→46 점프)을 실효 플레이 시간으로 변환
    - 하프타임을 제거하여 연속적인 시간축을 구성
    """
    fixture_id = match.fixture_id

    # ── 1. 관련 이벤트 필터링 및 분류 ─────────────────────

    goals: List[RawEvent] = []
    red_cards: List[RawEvent] = []

    for evt in raw_events:
        if evt.event_type == "Goal" and evt.event_detail != "Missed Penalty":
            goals.append(evt)
        elif evt.event_type == "Card" and evt.event_detail in ("Red Card", "Second Yellow card"):
            red_cards.append(evt)

    # ── 2. 시간축 구성 (실효 플레이 시간) ─────────────────

    # 전반 추가시간 추정
    # 전반 이벤트 중 45분 이후의 최대 시각을 찾는다
    first_half_events = [
        e for e in raw_events
        if e.event_minute is not None and e.event_minute <= FIRST_HALF_REGULAR_END
    ]
    first_half_stoppage = 0
    for e in first_half_events:
        extra = e.event_extra or 0
        if e.event_minute == FIRST_HALF_REGULAR_END and extra > first_half_stoppage:
            first_half_stoppage = extra

    # 하프타임 스코어로 검증: 전반에 골이 있었는데 전반 이벤트에 안 잡힌 경우 대비
    # (추가시간 2~3분이 보편적)
    if first_half_stoppage == 0:
        first_half_stoppage = 2   # 기본값

    first_half_end = FIRST_HALF_REGULAR_END + first_half_stoppage
    # 후반은 전반 직후 시작 (하프타임 제거)
    second_half_start = first_half_end

    # T_m (실효 경기 종료 시간)
    # 방법: matches.elapsed_minutes에서 계산 or 마지막 이벤트에서 추정
    if match.elapsed_minutes and match.elapsed_minutes > 0:
        # API-Football의 elapsed는 90+α₂ 형태 → 후반 부분만 추출
        second_half_play = match.elapsed_minutes - FIRST_HALF_REGULAR_END
        T_m = first_half_end + second_half_play
    else:
        # fallback: 마지막 이벤트 시각 + 1분
        second_half_events = [
            e for e in raw_events
            if e.event_minute is not None and e.event_minute >= SECOND_HALF_API_START
        ]
        if second_half_events:
            last_api_min = max(e.actual_minute for e in second_half_events)
            second_half_play = last_api_min - FIRST_HALF_REGULAR_END
            T_m = first_half_end + second_half_play + 1
        else:
            # 후반 이벤트 없음 → 기본 경기 시간
            T_m = first_half_end + 45 + 3  # 90+3+2 ≈ 95분

    # 기저함수 경계 (7개 점 → 6개 구간)
    basis_boundaries = [
        0.0,
        BASIS_BIN_MINUTES,                                          # 15
        2 * BASIS_BIN_MINUTES,                                      # 30
        first_half_end,                                             # ~47
        second_half_start + BASIS_BIN_MINUTES,                      # ~62
        second_half_start + 2 * BASIS_BIN_MINUTES,                  # ~77
        T_m,                                                        # ~95
    ]

    # ── 3. 이벤트를 실효 시간으로 변환 ────────────────────

    def to_effective_time(evt: RawEvent) -> float:
        """API-Football 시각 → 실효 플레이 시간"""
        api_min = evt.actual_minute

        if evt.event_minute <= FIRST_HALF_REGULAR_END:
            # 전반 이벤트: 그대로
            return api_min
        else:
            # 후반 이벤트: 하프타임 제거
            # API 46분 → 실효 first_half_end + 1
            return first_half_end + (api_min - FIRST_HALF_REGULAR_END)

    # ── 4. 분할 이벤트 목록 구성 ─────────────────────────

    @dataclass
    class SplitEvent:
        """구간 분할을 유발하는 이벤트"""
        t_eff: float
        event_type: str          # "goal", "red_card", "halftime_start", "halftime_end"
        team_id: Optional[int]
        team_name: Optional[str]
        detail: Optional[str]    # "Own Goal" 등

    split_events: List[SplitEvent] = []

    # 골
    for g in goals:
        split_events.append(SplitEvent(
            t_eff=to_effective_time(g),
            event_type="goal",
            team_id=g.team_id,
            team_name=g.team_name,
            detail=g.event_detail,
        ))

    # 레드카드
    for rc in red_cards:
        split_events.append(SplitEvent(
            t_eff=to_effective_time(rc),
            event_type="red_card",
            team_id=rc.team_id,
            team_name=rc.team_name,
            detail=rc.event_detail,
        ))

    # 하프타임 (전반 종료 ~ 후반 시작, 실효 시간으로는 길이 0)
    # 하프타임은 실효 시간에서 제거되므로 길이=0인 마커만 남긴다.
    # → 하프타임 구간은 별도로 만들지 않음 (실효 시간에 포함 안 됨)

    # 시간순 정렬 (같은 시각이면 레드카드를 골보다 먼저 처리)
    event_order = {"red_card": 0, "goal": 1}
    split_events.sort(key=lambda e: (e.t_eff, event_order.get(e.event_type, 2)))

    # ── 5. 상태 추적하며 구간 분할 ────────────────────────

    intervals: List[Interval] = []
    goal_events_out: List[GoalEvent] = []

    # 현재 상태
    state_X = 0
    score_home = 0
    score_away = 0
    delta_S = 0
    current_t = 0.0

    # 홈/어웨이 판별 함수
    def is_home_team(team_id: int) -> bool:
        return team_id == match.home_team_id

    def get_basis_idx(t: float) -> int:
        """실효 시간 → 기저함수 인덱스 (0~5)"""
        for i in range(NUM_BASIS):
            if basis_boundaries[i] <= t < basis_boundaries[i + 1]:
                return i
        # t가 T_m 이상이면 마지막 빈
        return NUM_BASIS - 1

    def close_interval(t_end: float):
        """현재 구간을 닫고 intervals에 추가"""
        nonlocal current_t
        if t_end <= current_t:
            return   # 길이 0인 구간은 건너뜀

        # T_m을 넘는 이벤트가 있을 수 있음 (추가시간 골 등)
        # → T_m을 이벤트 시각까지 확장
        nonlocal T_m, basis_boundaries
        if t_end > T_m:
            T_m = t_end
            basis_boundaries[-1] = T_m

        # 구간이 기저함수 경계를 걸치면 쪼개야 한다
        t = current_t
        while t < t_end:
            bi = get_basis_idx(t)
            # 이 기저함수 빈의 끝
            bin_end = basis_boundaries[bi + 1] if bi + 1 < len(basis_boundaries) else T_m
            seg_end = min(t_end, bin_end)

            # 진행 불가 방지 (부동소수점 오차 등)
            if seg_end <= t:
                break

            intervals.append(Interval(
                fixture_id=fixture_id,
                t_start=t,
                t_end=seg_end,
                state_X=state_X,
                delta_S=delta_S,
                basis_idx=bi,
                is_halftime=False,
                T_m=T_m,
            ))
            t = seg_end

        current_t = t_end

    # ── 이벤트 순회 ──────────────────────────────────────

    for evt in split_events:
        t = evt.t_eff

        # 시간 역전 방지 (같은 분에 여러 이벤트)
        t = max(t, current_t)

        if evt.event_type == "red_card":
            # 1. 현재 구간 닫기
            close_interval(t)

            # 2. 상태 전이
            team_is_home = is_home_team(evt.team_id)
            if team_is_home:
                if state_X == 0:
                    state_X = 1    # 11v11 → 10v11 (홈 퇴장)
                elif state_X == 2:
                    state_X = 3    # 11v10 → 10v10
            else:
                if state_X == 0:
                    state_X = 2    # 11v11 → 11v10 (어웨이 퇴장)
                elif state_X == 1:
                    state_X = 3    # 10v11 → 10v10

            # 3. 새 구간 시작 (current_t는 이미 t)

        elif evt.event_type == "goal":
            # 1. 현재 구간 닫기
            close_interval(t)

            # 2. 골 점 이벤트 기록 — ΔS는 직전 값! (인과관계)
            #    API-Football에서 골 이벤트의 team_id는 자책골 포함 항상
            #    득점을 인정받는 팀(수혜팀)을 가리킨다. 특수 처리 불필요.
            actual_scoring_team = "home" if is_home_team(evt.team_id) else "away"

            goal_events_out.append(GoalEvent(
                fixture_id=fixture_id,
                t_eff=t,
                team=actual_scoring_team,
                state_X=state_X,
                delta_S_before=delta_S,    # ← 인과관계 핵심
                basis_idx=get_basis_idx(t),
            ))

            # 3. 스코어 및 ΔS 업데이트 (골 이후)
            if actual_scoring_team == "home":
                score_home += 1
            else:
                score_away += 1
            delta_S = score_home - score_away

            # 4. 새 구간 시작 (새 ΔS 적용)

    # ── 마지막 구간 닫기 (경기 종료까지) ──────────────────

    close_interval(T_m)

    # ── 6. 검증: 이벤트 기반 스코어 vs DB 스코어 ─────────

    if match.home_goals_ft is not None and match.away_goals_ft is not None:
        if score_home != match.home_goals_ft or score_away != match.away_goals_ft:
            logger.warning(
                f"  ⚠️  스코어 불일치 fixture {fixture_id}: "
                f"이벤트={score_home}-{score_away}, "
                f"DB={match.home_goals_ft}-{match.away_goals_ft} "
                f"(자책골 등 수동 확인 필요)"
            )

    # ── 7. 메타데이터 구성 ────────────────────────────────

    meta = MatchMeta(
        fixture_id=fixture_id,
        T_m=T_m,
        first_half_end=first_half_end,
        second_half_start=second_half_start,
        basis_boundaries=basis_boundaries,
        total_goals=len(goal_events_out),
        total_red_cards=len(red_cards),
    )

    return intervals, goal_events_out, meta


# ─────────────────────────────────────────────────────────
# 배치 처리 + DB 쓰기
# ─────────────────────────────────────────────────────────

class Preprocessor:
    """Phase 1.1 전처리 파이프라인"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def _connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(OUTPUT_SCHEMA_SQL)
        self.conn.commit()

    def _close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _load_match(self, fixture_id: int) -> Optional[MatchInfo]:
        """DB에서 경기 메타데이터 로드"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fixture_id, league_id, season,
                   home_team_id, home_team_name,
                   away_team_id, away_team_name,
                   home_goals_ft, away_goals_ft,
                   home_goals_ht, away_goals_ht,
                   elapsed_minutes
            FROM matches WHERE fixture_id = ?
        """, (fixture_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return MatchInfo(*row)

    def _load_events(self, fixture_id: int) -> List[RawEvent]:
        """DB에서 경기 이벤트 로드"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fixture_id, event_minute, event_extra,
                   event_type, event_detail,
                   team_id, team_name
            FROM match_events
            WHERE fixture_id = ?
            ORDER BY event_minute, event_extra
        """, (fixture_id,))

        events = []
        for row in cursor.fetchall():
            # NULL 처리
            fid, minute, extra, etype, detail, tid, tname = row
            if minute is None or etype is None:
                continue
            events.append(RawEvent(
                fixture_id=fid,
                event_minute=minute,
                event_extra=extra,
                event_type=etype or "",
                event_detail=detail or "",
                team_id=tid or 0,
                team_name=tname or "",
            ))

        return events

    def _get_fixture_ids(
        self, league_id: Optional[int] = None, season: Optional[int] = None
    ) -> List[int]:
        """처리 대상 경기 목록 조회"""
        cursor = self.conn.cursor()

        query = """
            SELECT m.fixture_id
            FROM matches m
            INNER JOIN download_progress dp
                ON m.fixture_id = dp.fixture_id
                AND dp.data_type = 'events'
                AND dp.status = 'done'
            WHERE 1=1
        """
        params = []

        if league_id is not None:
            query += " AND m.league_id = ?"
            params.append(league_id)
        if season is not None:
            query += " AND m.season = ?"
            params.append(season)

        query += " ORDER BY m.match_date"
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def _save_results(
        self,
        intervals: List[Interval],
        goal_events: List[GoalEvent],
        meta: MatchMeta,
    ):
        """처리 결과를 DB에 저장"""
        import json
        cursor = self.conn.cursor()
        fixture_id = meta.fixture_id

        # 기존 데이터 삭제 (재처리 대비)
        cursor.execute("DELETE FROM intervals WHERE fixture_id = ?", (fixture_id,))
        cursor.execute("DELETE FROM goal_events WHERE fixture_id = ?", (fixture_id,))
        cursor.execute("DELETE FROM match_meta WHERE fixture_id = ?", (fixture_id,))

        # 구간 저장
        for iv in intervals:
            cursor.execute("""
                INSERT INTO intervals
                    (fixture_id, t_start, t_end, duration,
                     state_X, delta_S, basis_idx, is_halftime, T_m)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                iv.fixture_id, iv.t_start, iv.t_end, iv.duration,
                iv.state_X, iv.delta_S, iv.basis_idx,
                1 if iv.is_halftime else 0, iv.T_m,
            ))

        # 골 이벤트 저장
        for g in goal_events:
            cursor.execute("""
                INSERT INTO goal_events
                    (fixture_id, t_eff, team, state_X, delta_S_before, basis_idx)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                g.fixture_id, g.t_eff, g.team,
                g.state_X, g.delta_S_before, g.basis_idx,
            ))

        # 메타 저장
        cursor.execute("""
            INSERT INTO match_meta
                (fixture_id, T_m, first_half_end, second_half_start,
                 basis_boundaries, total_goals, total_red_cards)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            meta.fixture_id, meta.T_m,
            meta.first_half_end, meta.second_half_start,
            json.dumps(meta.basis_boundaries),
            meta.total_goals, meta.total_red_cards,
        ))

    def process_all(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
    ):
        """전체 경기 배치 처리"""
        self._connect()

        try:
            fixture_ids = self._get_fixture_ids(league_id, season)
            total = len(fixture_ids)

            if total == 0:
                logger.warning("처리할 경기가 없습니다. api_football.py를 먼저 실행하세요.")
                return

            logger.info(f"📐 Phase 1.1 전처리 시작: {total}경기")

            total_intervals = 0
            total_goals = 0
            total_reds = 0
            errors = 0

            for i, fid in enumerate(fixture_ids, 1):
                if i % 500 == 0 or i == total:
                    logger.info(f"  [{i:5d}/{total}] ({i/total*100:5.1f}%)")

                try:
                    match = self._load_match(fid)
                    if not match:
                        continue

                    events = self._load_events(fid)
                    intervals, goal_events, meta = process_match(match, events)
                    self._save_results(intervals, goal_events, meta)

                    total_intervals += len(intervals)
                    total_goals += len(goal_events)
                    total_reds += meta.total_red_cards

                except Exception as e:
                    errors += 1
                    logger.warning(f"  ⚠️  fixture {fid} 처리 실패: {e}")
                    if errors <= 5:
                        import traceback
                        traceback.print_exc()

            self.conn.commit()

            logger.info(f"\n{'='*50}")
            logger.info(f"✅ 전처리 완료!")
            logger.info(f"  경기: {total} (오류: {errors})")
            logger.info(f"  구간: {total_intervals:,} (평균 {total_intervals/max(total-errors,1):.1f}/경기)")
            logger.info(f"  골:   {total_goals:,} (평균 {total_goals/max(total-errors,1):.2f}/경기)")
            logger.info(f"  퇴장: {total_reds:,}")
            logger.info(f"{'='*50}\n")

        finally:
            self._close()

    # ── 검증 ─────────────────────────────────────────────

    def verify(self):
        """전처리 결과의 무결성 검증"""
        self._connect()
        try:
            cursor = self.conn.cursor()

            print("\n" + "="*60)
            print("🔍 Phase 1.1 전처리 결과 검증")
            print("="*60)

            # 1. 기본 통계
            cursor.execute("SELECT COUNT(*) FROM match_meta")
            n_matches = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM intervals WHERE is_halftime = 0")
            n_intervals = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM goal_events")
            n_goals = cursor.fetchone()[0]

            print(f"\n  처리된 경기: {n_matches:,}")
            print(f"  구간 (active): {n_intervals:,}")
            print(f"  골 이벤트: {n_goals:,}")

            if n_matches == 0:
                print("\n  ❌ 처리된 데이터 없음. preprocessor --all을 먼저 실행하세요.")
                return

            # 2. 구간 연속성 검증: 각 경기에서 구간이 빈틈/겹침 없이 이어지는지
            cursor.execute("""
                SELECT fixture_id, t_start, t_end
                FROM intervals
                WHERE is_halftime = 0
                ORDER BY fixture_id, t_start
            """)

            gap_errors = 0
            overlap_errors = 0
            prev_fid = None
            prev_end = 0.0

            for fid, t_start, t_end in cursor.fetchall():
                if fid != prev_fid:
                    prev_fid = fid
                    prev_end = 0.0  # 첫 구간은 0부터 시작해야 함

                if abs(t_start - prev_end) > 0.01:
                    if t_start > prev_end + 0.01:
                        gap_errors += 1
                    elif t_start < prev_end - 0.01:
                        overlap_errors += 1
                prev_end = t_end

            if gap_errors == 0 and overlap_errors == 0:
                print("\n  ✅ 구간 연속성: 빈틈/겹침 없음")
            else:
                print(f"\n  ⚠️  구간 연속성 문제:")
                if gap_errors:
                    print(f"    빈틈(gap): {gap_errors}건")
                if overlap_errors:
                    print(f"    겹침(overlap): {overlap_errors}건")

            # 3. 상태 X 분포
            cursor.execute("""
                SELECT state_X, COUNT(*), SUM(duration)
                FROM intervals WHERE is_halftime = 0
                GROUP BY state_X
            """)
            print(f"\n  마르코프 상태 분포:")
            state_names = {0: "11v11", 1: "10v11(홈↓)", 2: "11v10(어웨이↓)", 3: "10v10"}
            for state_x, count, total_dur in cursor.fetchall():
                name = state_names.get(state_x, f"상태{state_x}")
                print(f"    X={state_x} ({name}): {count:,}구간, {total_dur:,.0f}분")

            # 4. ΔS 분포
            cursor.execute("""
                SELECT delta_S, COUNT(*), SUM(duration)
                FROM intervals WHERE is_halftime = 0
                GROUP BY delta_S
                ORDER BY delta_S
            """)
            print(f"\n  ΔS (득점차) 분포:")
            for ds, count, total_dur in cursor.fetchall():
                print(f"    ΔS={ds:+d}: {count:,}구간, {total_dur:,.0f}분")

            # 5. 기저함수 빈별 분포
            cursor.execute("""
                SELECT basis_idx, COUNT(*), SUM(duration)
                FROM intervals WHERE is_halftime = 0
                GROUP BY basis_idx
                ORDER BY basis_idx
            """)
            bin_names = ["전반 초(0-15)", "전반 중(15-30)", "전반 말(30-HT)",
                         "후반 초(HT-+15)", "후반 중(+15-+30)", "후반 말(+30-FT)"]
            print(f"\n  기저함수 빈별 분포:")
            for bi, count, total_dur in cursor.fetchall():
                name = bin_names[bi] if bi < len(bin_names) else f"빈{bi}"
                print(f"    b_{bi} ({name}): {count:,}구간, {total_dur:,.0f}분")

            # 6. 골 인과관계 검증: delta_S_before가 합리적인지
            cursor.execute("""
                SELECT team, delta_S_before, COUNT(*)
                FROM goal_events
                GROUP BY team, delta_S_before
                ORDER BY team, delta_S_before
            """)
            print(f"\n  골 이벤트 분포 (팀 × 직전ΔS):")
            for team, ds_before, count in cursor.fetchall():
                print(f"    {team} 팀 득점, 직전 ΔS={ds_before:+d}: {count:,}골")

            # 7. 경기당 평균 득점 (sanity check)
            cursor.execute("""
                SELECT AVG(total_goals) FROM match_meta
            """)
            avg_goals = cursor.fetchone()[0]
            print(f"\n  경기당 평균 골: {avg_goals:.2f}")
            if 2.2 <= avg_goals <= 3.2:
                print("  ✅ 합리적인 범위 (유럽 5대 리그 평균 ~2.7)")
            else:
                print("  ⚠️  예상 범위(2.2~3.2) 밖 — 데이터 확인 필요")

            # 8. T_m 분포
            cursor.execute("""
                SELECT MIN(T_m), AVG(T_m), MAX(T_m) FROM match_meta
            """)
            t_min, t_avg, t_max = cursor.fetchone()
            print(f"\n  T_m (실효 종료 시간): "
                  f"min={t_min:.1f}, avg={t_avg:.1f}, max={t_max:.1f}")
            if 90 <= t_avg <= 100:
                print("  ✅ 합리적인 범위")
            else:
                print("  ⚠️  예상 범위(90~100) 밖")

            print()

        finally:
            self._close()

    # ── 디버그: 단일 경기 상세 출력 ───────────────────────

    def debug_match(self, fixture_id: int):
        """특정 경기의 구간 분할 결과를 상세히 출력"""
        self._connect()
        try:
            match = self._load_match(fixture_id)
            if not match:
                print(f"❌ fixture {fixture_id}를 찾을 수 없습니다.")
                return

            events = self._load_events(fixture_id)

            print(f"\n{'='*70}")
            print(f"🔍 경기 상세: {match.home_team_name} vs {match.away_team_name}")
            print(f"   최종 스코어: {match.home_goals_ft}-{match.away_goals_ft}")
            print(f"{'='*70}")

            # 원시 이벤트 출력
            print(f"\n📋 원시 이벤트 ({len(events)}개):")
            for e in events:
                extra_str = f"+{e.event_extra}" if e.event_extra else ""
                print(f"  {e.event_minute}{extra_str}분  "
                      f"{e.event_type:<6} {e.event_detail:<20} "
                      f"{e.team_name}")

            # 구간 분할 실행
            intervals, goal_events, meta = process_match(match, events)

            print(f"\n📐 구간 분할 결과 ({len(intervals)}개):")
            state_names = {0: "11v11", 1: "10v11", 2: "11v10", 3: "10v10"}
            print(f"  {'구간':<4} {'시간 범위':<18} {'X':<7} {'ΔS':>3} {'빈':>2} {'길이':>6}")
            print(f"  {'-'*50}")
            for i, iv in enumerate(intervals, 1):
                ht_mark = " [HT]" if iv.is_halftime else ""
                print(f"  {i:<4} [{iv.t_start:5.1f}, {iv.t_end:5.1f})"
                      f"  {state_names[iv.state_X]:<7} {iv.delta_S:+3d}"
                      f"  b{iv.basis_idx}"
                      f"  {iv.duration:5.1f}분"
                      f"{ht_mark}")

            print(f"\n⚽ 골 이벤트 ({len(goal_events)}개):")
            print(f"  {'시각':>6} {'팀':<6} {'X':<7} {'직전ΔS':>6} {'빈':>2}")
            print(f"  {'-'*35}")
            for g in goal_events:
                print(f"  {g.t_eff:5.1f}분 {g.team:<6} "
                      f"{state_names[g.state_X]:<7} {g.delta_S_before:+5d}"
                      f"  b{g.basis_idx}")

            print(f"\n📊 메타:")
            print(f"  T_m = {meta.T_m:.1f}분")
            print(f"  전반 종료 = {meta.first_half_end:.1f}분")
            print(f"  기저함수 경계 = {[f'{b:.1f}' for b in meta.basis_boundaries]}")
            print()

        finally:
            self._close()


# ─────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.1: 시계열 이벤트 분할 및 구간 데이터화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.data.preprocessor                           # 전체 처리
  python -m src.data.preprocessor --league 39 --season 2023 # PL 2023-24
  python -m src.data.preprocessor --verify                  # 결과 검증
  python -m src.data.preprocessor --debug-match 868124      # 특정 경기 상세
        """,
    )

    parser.add_argument("--league", type=int,
                        help="리그 ID (예: 39 = Premier League)")
    parser.add_argument("--season", type=int,
                        help="시즌 연도 (예: 2023)")
    parser.add_argument("--verify", action="store_true",
                        help="전처리 결과 무결성 검증")
    parser.add_argument("--debug-match", type=int, metavar="FIXTURE_ID",
                        help="특정 경기의 구간 분할을 상세히 출력")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help=f"SQLite DB 경로 (기본: {DB_PATH})")

    args = parser.parse_args()
    proc = Preprocessor(db_path=Path(args.db))

    if args.verify:
        proc.verify()
    elif args.debug_match:
        proc.debug_match(args.debug_match)
    else:
        proc.process_all(league_id=args.league, season=args.season)


if __name__ == "__main__":
    main()
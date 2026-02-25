"""
API-Football 데이터 수집 클라이언트.

Phase 1에 필요한 과거 경기 데이터를 API-Football에서 다운로드하여
SQLite 데이터베이스에 저장한다.

필요한 데이터:
  - /fixtures          : 경기 메타데이터 (스코어, 종료시간 등)
  - /fixtures/events   : 분 단위 이벤트 (골, 레드카드) → Phase 1.1 구간 분할
  - /fixtures/statistics: 경기별 스탯 (점유율, xG 등)  → Phase 1.3 XGBoost 피처
  - /odds              : 프리매치 배당률                → Phase 1.3 피처 + Phase 2.4 Sanity Check

사용법:
  # 프리미어리그 2023-24 시즌 다운로드
  python -m src.data.api_football --league 39 --season 2023

  # 5대 리그 × 최근 5시즌 전체 다운로드
  python -m src.data.api_football --all

  # 다운로드 진행 상태 확인
  python -m src.data.api_football --status

  # 중단 후 이어받기 (자동으로 이미 받은 것은 건너뜀)
  python -m src.data.api_football --all
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

# .env 파일에서 API 키 로드
load_dotenv()

# 5대 리그 코드 (API-Football league ID)
LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    135: "Serie A",
    78:  "Bundesliga",
    61:  "Ligue 1",
}

# 다운로드 대상 시즌
SEASONS = [2020, 2021, 2022, 2023, 2024]

# API 기본 설정
API_BASE_URL = "https://v3.football.api-sports.io"

# 데이터 경로
DB_PATH = Path("data/kalshi_football.db")
RAW_DIR = Path("data/raw")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api_football.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 데이터베이스 스키마
# ─────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- 경기 메타데이터
-- Phase 1.1: T_m (실제 종료 시간), 스코어
-- Phase 1.3: 타겟 변수 (총 득점)
CREATE TABLE IF NOT EXISTS matches (
    fixture_id      INTEGER PRIMARY KEY,
    league_id       INTEGER NOT NULL,
    league_name     TEXT,
    season          INTEGER NOT NULL,
    match_date      TEXT NOT NULL,           -- ISO 형식
    round           TEXT,                    -- "Regular Season - 1" 등

    home_team_id    INTEGER NOT NULL,
    home_team_name  TEXT NOT NULL,
    away_team_id    INTEGER NOT NULL,
    away_team_name  TEXT NOT NULL,

    home_goals_ft   INTEGER,                -- 풀타임 홈 득점
    away_goals_ft   INTEGER,                -- 풀타임 어웨이 득점
    home_goals_ht   INTEGER,                -- 하프타임 홈 득점
    away_goals_ht   INTEGER,                -- 하프타임 어웨이 득점

    elapsed_minutes INTEGER,                -- 경기 경과 시간 (status.elapsed)
    venue_name      TEXT,
    referee         TEXT,

    downloaded_at   TEXT DEFAULT (datetime('now'))
);

-- 경기 이벤트 (골, 레드카드 등)
-- Phase 1.1: 구간 분할의 핵심 입력
CREATE TABLE IF NOT EXISTS match_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    event_minute    INTEGER NOT NULL,        -- 이벤트 발생 분
    event_extra     INTEGER,                 -- 추가시간 내 분 (예: 90+3이면 extra=3)
    event_type      TEXT NOT NULL,           -- "Goal", "Card", "subst" 등
    event_detail    TEXT,                    -- "Normal Goal", "Red Card", "Yellow Card" 등
    team_id         INTEGER,
    team_name       TEXT,
    player_id       INTEGER,
    player_name     TEXT,
    assist_id       INTEGER,
    assist_name     TEXT,
    comments        TEXT,

    FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);

-- 경기 통계 (팀별)
-- Phase 1.3: XGBoost 피처 (롤링 평균의 원천)
CREATE TABLE IF NOT EXISTS match_statistics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    team_name       TEXT,
    stat_type       TEXT NOT NULL,           -- "Ball Possession", "Expected Goals" 등
    stat_value      TEXT,                    -- 원본 값 (문자열, "55%" 또는 "1.85")

    FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);

-- 프리매치 배당률
-- Phase 1.3: 피처 (시장 내재 확률)
-- Phase 2.4: Sanity Check
CREATE TABLE IF NOT EXISTS match_odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL,
    bookmaker_id    INTEGER,
    bookmaker_name  TEXT,
    bet_type        TEXT NOT NULL,           -- "Match Winner", "Goals Over/Under" 등
    bet_value       TEXT NOT NULL,           -- "Home", "Draw", "Away", "Over 2.5" 등
    odd_value       REAL,                    -- 배당률 (decimal)

    FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id)
);

-- 다운로드 진행 상태 추적 (이어받기용)
CREATE TABLE IF NOT EXISTS download_progress (
    fixture_id      INTEGER NOT NULL,
    data_type       TEXT NOT NULL,           -- "events", "statistics", "odds"
    status          TEXT DEFAULT 'pending',  -- "pending", "done", "error"
    error_message   TEXT,
    downloaded_at   TEXT,
    PRIMARY KEY (fixture_id, data_type)
);

-- 인덱스 (조회 속도)
CREATE INDEX IF NOT EXISTS idx_matches_league_season
    ON matches(league_id, season);
CREATE INDEX IF NOT EXISTS idx_events_fixture
    ON match_events(fixture_id);
CREATE INDEX IF NOT EXISTS idx_events_type
    ON match_events(fixture_id, event_type);
CREATE INDEX IF NOT EXISTS idx_stats_fixture
    ON match_statistics(fixture_id);
CREATE INDEX IF NOT EXISTS idx_odds_fixture
    ON match_odds(fixture_id);
"""


# ─────────────────────────────────────────────────────────
# API 클라이언트
# ─────────────────────────────────────────────────────────

class APIFootballClient:
    """
    API-Football v3 REST 클라이언트.

    기능:
    - 속도 제한(rate limit) 자동 준수
    - 실패 시 자동 재시도 (최대 3회)
    - 원본 JSON 로컬 백업
    - 이어받기 (이미 다운로드한 데이터는 건너뜀)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            raise ValueError(
                "API_FOOTBALL_KEY가 .env 파일에 없습니다.\n"
                ".env 파일에 다음을 추가하세요:\n"
                "API_FOOTBALL_KEY=여기에_실제_키"
            )

        self.headers = {"x-apisports-key": self.api_key}
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

        # 원본 JSON 백업 디렉토리
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # 로그 디렉토리
        Path("logs").mkdir(exist_ok=True)

        # 속도 제한 추적
        self._requests_remaining = None
        self._last_request_time = 0

    # ── 데이터베이스 ──────────────────────────────────────

    def _connect_db(self):
        """SQLite 연결 + 스키마 생성"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")      # 동시 읽기 성능
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def _close_db(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ── HTTP 요청 ────────────────────────────────────────

    def _request(self, endpoint: str, params: dict, max_retries: int = 3) -> dict:
        """
        API-Football에 GET 요청을 보낸다.

        - 속도 제한 자동 준수 (응답 헤더의 남은 요청 수 확인)
        - 실패 시 지수 백오프로 재시도
        - 원본 응답을 로컬에 백업
        """
        url = f"{API_BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            # 속도 제한: 최소 요청 간격 보장
            elapsed = time.time() - self._last_request_time
            min_interval = 1.2   # 초 (분당 ~50회, 안전 마진 포함)
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )
                self._last_request_time = time.time()

                # 속도 제한 정보 추출
                remaining = response.headers.get("x-ratelimit-requests-remaining")
                if remaining is not None:
                    self._requests_remaining = int(remaining)
                    if self._requests_remaining <= 2:
                        logger.warning(
                            f"⚠️  일일 요청 한도 거의 소진: 남은 {self._requests_remaining}회"
                        )
                        if self._requests_remaining <= 0:
                            logger.error("❌ 일일 요청 한도 소진. 내일 다시 실행하세요.")
                            raise SystemExit(1)

                # HTTP 에러 확인
                if response.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"429 Too Many Requests. {wait}초 대기 후 재시도...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # API 레벨 에러 확인
                if data.get("errors"):
                    errors = data["errors"]
                    # 빈 딕셔너리 {}도 errors로 올 수 있음
                    if isinstance(errors, dict) and errors:
                        logger.warning(f"API 에러: {errors}")
                        if "rateLimit" in str(errors):
                            time.sleep(60)
                            continue
                    elif isinstance(errors, list) and errors:
                        logger.warning(f"API 에러: {errors}")

                return data

            except requests.exceptions.Timeout:
                logger.warning(f"타임아웃 (시도 {attempt+1}/{max_retries})")
                time.sleep(5 * (attempt + 1))

            except requests.exceptions.ConnectionError:
                logger.warning(f"연결 실패 (시도 {attempt+1}/{max_retries})")
                time.sleep(10 * (attempt + 1))

            except requests.exceptions.RequestException as e:
                logger.error(f"요청 실패: {e}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError(f"API 요청 {max_retries}회 실패: {endpoint} {params}")

    def _save_raw_json(self, filename: str, data: dict):
        """원본 JSON을 로컬 파일로 백업 (디버깅 + 재처리용)"""
        filepath = RAW_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 경기 목록 다운로드 ────────────────────────────────

    def download_fixtures(self, league_id: int, season: int) -> list:
        """
        한 시즌의 모든 완료된 경기를 다운로드하여 DB에 저장한다.

        Returns:
            경기 fixture_id 목록
        """
        logger.info(f"📥 경기 목록 다운로드: {LEAGUES.get(league_id, league_id)} {season}")

        data = self._request("fixtures", {
            "league": league_id,
            "season": season,
            "status": "FT",    # 종료된 경기만
        })
        self._save_raw_json(f"fixtures_{league_id}_{season}.json", data)

        fixtures = data.get("response", [])
        if not fixtures:
            logger.warning(f"  경기 데이터 없음: league={league_id}, season={season}")
            return []

        fixture_ids = []
        cursor = self.conn.cursor()

        for match in fixtures:
            fx = match["fixture"]
            league = match["league"]
            teams = match["teams"]
            goals = match["goals"]
            score = match["score"]

            fixture_id = fx["id"]
            fixture_ids.append(fixture_id)

            # 하프타임 스코어 (없을 수 있음)
            ht = score.get("halftime", {})
            ht_home = ht.get("home")
            ht_away = ht.get("away")

            # 경기 경과 시간 (T_m 추정에 사용)
            elapsed = fx.get("status", {}).get("elapsed")

            cursor.execute("""
                INSERT OR REPLACE INTO matches (
                    fixture_id, league_id, league_name, season, match_date,
                    round, home_team_id, home_team_name,
                    away_team_id, away_team_name,
                    home_goals_ft, away_goals_ft,
                    home_goals_ht, away_goals_ht,
                    elapsed_minutes, venue_name, referee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id,
                league.get("id"),
                league.get("name"),
                league.get("season"),
                fx.get("date"),
                league.get("round"),
                teams["home"]["id"],
                teams["home"]["name"],
                teams["away"]["id"],
                teams["away"]["name"],
                goals.get("home"),
                goals.get("away"),
                ht_home,
                ht_away,
                elapsed,
                fx.get("venue", {}).get("name"),
                fx.get("referee"),
            ))

            # 상세 데이터 다운로드 진행 상태 초기화
            for dtype in ("events", "statistics", "odds"):
                cursor.execute("""
                    INSERT OR IGNORE INTO download_progress
                        (fixture_id, data_type, status)
                    VALUES (?, ?, 'pending')
                """, (fixture_id, dtype))

        self.conn.commit()
        logger.info(f"  ✅ {len(fixture_ids)}경기 저장 완료")
        return fixture_ids

    # ── 경기별 상세 데이터 다운로드 ───────────────────────

    def _is_downloaded(self, fixture_id: int, data_type: str) -> bool:
        """이미 다운로드했는지 확인 (이어받기용)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT status FROM download_progress
            WHERE fixture_id = ? AND data_type = ?
        """, (fixture_id, data_type))
        row = cursor.fetchone()
        return row is not None and row[0] == "done"

    def _mark_done(self, fixture_id: int, data_type: str):
        """다운로드 완료 표시"""
        self.conn.execute("""
            UPDATE download_progress
            SET status = 'done', downloaded_at = datetime('now')
            WHERE fixture_id = ? AND data_type = ?
        """, (fixture_id, data_type))
        self.conn.commit()

    def _mark_error(self, fixture_id: int, data_type: str, error_msg: str):
        """다운로드 실패 표시"""
        self.conn.execute("""
            UPDATE download_progress
            SET status = 'error', error_message = ?
            WHERE fixture_id = ? AND data_type = ?
        """, (error_msg, fixture_id, data_type))
        self.conn.commit()

    def download_events(self, fixture_id: int):
        """
        경기별 이벤트 (골, 레드카드, 교체 등) 다운로드.

        Phase 1.1 구간 분할에 사용:
        - Goal → ΔS 변경 → 구간 분할
        - Red Card → X 변경 → 구간 분할
        """
        if self._is_downloaded(fixture_id, "events"):
            return

        try:
            data = self._request("fixtures/events", {"fixture": fixture_id})
            self._save_raw_json(f"events_{fixture_id}.json", data)

            events = data.get("response", [])
            cursor = self.conn.cursor()

            for evt in events:
                time_info = evt.get("time", {})
                team = evt.get("team", {})
                player = evt.get("player", {})
                assist = evt.get("assist", {})

                cursor.execute("""
                    INSERT INTO match_events (
                        fixture_id, event_minute, event_extra,
                        event_type, event_detail,
                        team_id, team_name,
                        player_id, player_name,
                        assist_id, assist_name,
                        comments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fixture_id,
                    time_info.get("elapsed"),
                    time_info.get("extra"),
                    evt.get("type"),
                    evt.get("detail"),
                    team.get("id"),
                    team.get("name"),
                    player.get("id"),
                    player.get("name"),
                    assist.get("id"),
                    assist.get("name"),
                    evt.get("comments"),
                ))

            self.conn.commit()
            self._mark_done(fixture_id, "events")

        except Exception as e:
            self._mark_error(fixture_id, "events", str(e))
            logger.warning(f"  ⚠️  events 실패 (fixture {fixture_id}): {e}")

    def download_statistics(self, fixture_id: int):
        """
        경기별 통계 (점유율, xG, 유효슈팅 등) 다운로드.

        Phase 1.3 XGBoost 피처의 원천 데이터:
        - Ball Possession, Expected Goals, Shots on Goal
        - Total Passes, Pass Accuracy, PPDA 등
        """
        if self._is_downloaded(fixture_id, "statistics"):
            return

        try:
            data = self._request("fixtures/statistics", {"fixture": fixture_id})
            self._save_raw_json(f"stats_{fixture_id}.json", data)

            teams_stats = data.get("response", [])
            cursor = self.conn.cursor()

            for team_data in teams_stats:
                team = team_data.get("team", {})
                stats = team_data.get("statistics", [])

                for stat in stats:
                    cursor.execute("""
                        INSERT INTO match_statistics (
                            fixture_id, team_id, team_name,
                            stat_type, stat_value
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        fixture_id,
                        team.get("id"),
                        team.get("name"),
                        stat.get("type"),
                        str(stat.get("value")) if stat.get("value") is not None else None,
                    ))

            self.conn.commit()
            self._mark_done(fixture_id, "statistics")

        except Exception as e:
            self._mark_error(fixture_id, "statistics", str(e))
            logger.warning(f"  ⚠️  statistics 실패 (fixture {fixture_id}): {e}")

    def download_odds(self, fixture_id: int):
        """
        프리매치 배당률 다운로드.

        Phase 1.3: 시장 내재 확률을 XGBoost 피처로 사용
        Phase 2.4: Sanity Check (모델 vs 시장 괴리도)

        주요 베팅 타입:
        - Match Winner (1X2)
        - Goals Over/Under 2.5
        - Both Teams Score
        """
        if self._is_downloaded(fixture_id, "odds"):
            return

        try:
            data = self._request("odds", {"fixture": fixture_id})
            self._save_raw_json(f"odds_{fixture_id}.json", data)

            odds_list = data.get("response", [])
            cursor = self.conn.cursor()

            for odds_data in odds_list:
                bookmakers = odds_data.get("bookmakers", [])
                for bookie in bookmakers:
                    bookie_id = bookie.get("id")
                    bookie_name = bookie.get("name")

                    for bet in bookie.get("bets", []):
                        bet_name = bet.get("name")
                        for value in bet.get("values", []):
                            cursor.execute("""
                                INSERT INTO match_odds (
                                    fixture_id, bookmaker_id, bookmaker_name,
                                    bet_type, bet_value, odd_value
                                ) VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                fixture_id,
                                bookie_id,
                                bookie_name,
                                bet_name,
                                value.get("value"),
                                self._parse_odd(value.get("odd")),
                            ))

            self.conn.commit()
            self._mark_done(fixture_id, "odds")

        except Exception as e:
            self._mark_error(fixture_id, "odds", str(e))
            logger.warning(f"  ⚠️  odds 실패 (fixture {fixture_id}): {e}")

    @staticmethod
    def _parse_odd(value) -> Optional[float]:
        """배당률 문자열을 float로 변환"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # ── 전체 다운로드 오케스트레이션 ──────────────────────

    def download_season(self, league_id: int, season: int):
        """
        한 리그-시즌의 전체 데이터를 다운로드한다.

        순서:
        1. 경기 목록 다운로드 → matches 테이블
        2. 각 경기별 이벤트 다운로드 → match_events 테이블
        3. 각 경기별 통계 다운로드 → match_statistics 테이블
        4. 각 경기별 배당률 다운로드 → match_odds 테이블

        중단 후 재실행하면 이미 받은 데이터는 건너뛴다.
        """
        league_name = LEAGUES.get(league_id, f"League {league_id}")
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 {league_name} {season}-{season+1} 시즌 다운로드")
        logger.info(f"{'='*60}")

        # 1. 경기 목록
        fixture_ids = self.download_fixtures(league_id, season)
        if not fixture_ids:
            return

        total = len(fixture_ids)

        # 2~4. 각 경기별 상세 데이터
        for i, fid in enumerate(fixture_ids, 1):
            # 진행 상태 표시
            pct = i / total * 100
            remaining_str = self._estimate_remaining(i, total)

            logger.info(
                f"  [{i:3d}/{total}] ({pct:5.1f}%) "
                f"fixture {fid} {remaining_str}"
            )

            self.download_events(fid)
            self.download_statistics(fid)
            self.download_odds(fid)

        logger.info(f"✅ {league_name} {season} 완료!\n")

    def download_all(self):
        """5대 리그 × 5시즌 전체 다운로드"""
        total_seasons = len(LEAGUES) * len(SEASONS)
        current = 0

        for league_id, league_name in LEAGUES.items():
            for season in SEASONS:
                current += 1
                logger.info(
                    f"\n📊 전체 진행: [{current}/{total_seasons}] "
                    f"{league_name} {season}"
                )
                self.download_season(league_id, season)

        logger.info("\n" + "="*60)
        logger.info("🎉 전체 다운로드 완료!")
        logger.info("="*60)
        self.print_status()

    def _estimate_remaining(self, current: int, total: int) -> str:
        """남은 시간 추정"""
        # 경기당 3회 API 호출 × ~1.2초 = ~3.6초
        remaining_matches = total - current
        remaining_seconds = remaining_matches * 3.6
        if remaining_seconds < 60:
            return f"(~{remaining_seconds:.0f}초 남음)"
        elif remaining_seconds < 3600:
            return f"(~{remaining_seconds/60:.0f}분 남음)"
        else:
            return f"(~{remaining_seconds/3600:.1f}시간 남음)"

    # ── 진행 상태 조회 ───────────────────────────────────

    def print_status(self):
        """다운로드 진행 상태를 출력한다."""
        cursor = self.conn.cursor()

        # 리그-시즌별 경기 수
        cursor.execute("""
            SELECT league_name, season, COUNT(*) as cnt
            FROM matches
            GROUP BY league_id, season
            ORDER BY league_id, season
        """)
        rows = cursor.fetchall()

        print("\n" + "="*60)
        print("📊 다운로드 현황")
        print("="*60)

        if not rows:
            print("  아직 다운로드된 데이터가 없습니다.")
            print("  실행: python -m src.data.api_football --all")
            return

        print(f"\n{'리그':<20} {'시즌':<8} {'경기 수':<8}")
        print("-" * 40)
        total_matches = 0
        for league_name, season, count in rows:
            print(f"  {league_name:<18} {season:<8} {count:<8}")
            total_matches += count
        print("-" * 40)
        print(f"  {'합계':<18} {'':8} {total_matches:<8}")

        # 상세 데이터 진행률
        for dtype in ("events", "statistics", "odds"):
            cursor.execute("""
                SELECT status, COUNT(*) FROM download_progress
                WHERE data_type = ?
                GROUP BY status
            """, (dtype,))
            status_counts = dict(cursor.fetchall())
            done = status_counts.get("done", 0)
            pending = status_counts.get("pending", 0)
            error = status_counts.get("error", 0)
            total = done + pending + error

            if total > 0:
                pct = done / total * 100
                bar_len = 30
                filled = int(bar_len * done / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\n  {dtype:<12} [{bar}] {pct:5.1f}%  "
                      f"({done}/{total}, 오류 {error})")
        # 남은 API 요청 수
        if self._requests_remaining is not None:
            print(f"\n  📡 남은 API 요청: {self._requests_remaining}회")

        print()

    # ── 데이터 검증 ──────────────────────────────────────

    def verify_data(self):
        """
        다운로드된 데이터의 기본적인 무결성을 검증한다.

        검증 항목:
        1. 각 경기에 이벤트 데이터가 있는가?
        2. 이벤트의 골 수가 최종 스코어와 일치하는가?
        3. 통계 데이터에 핵심 지표가 있는가?
        """
        cursor = self.conn.cursor()

        print("\n" + "="*60)
        print("🔍 데이터 검증")
        print("="*60)

        # 1. 이벤트 누락 경기
        cursor.execute("""
            SELECT m.fixture_id, m.home_team_name, m.away_team_name
            FROM matches m
            LEFT JOIN download_progress dp
                ON m.fixture_id = dp.fixture_id AND dp.data_type = 'events'
            WHERE dp.status != 'done' OR dp.status IS NULL
            LIMIT 10
        """)
        missing_events = cursor.fetchall()
        if missing_events:
            print(f"\n  ⚠️  이벤트 미다운로드 경기 {len(missing_events)}개 (상위 10개):")
            for fid, home, away in missing_events:
                print(f"    fixture {fid}: {home} vs {away}")
        else:
            print("\n  ✅ 모든 경기에 이벤트 데이터 있음")

        # 2. 골 수 일치 검증
        cursor.execute("""
            SELECT
                m.fixture_id,
                m.home_team_name, m.away_team_name,
                m.home_goals_ft, m.away_goals_ft,
                (SELECT COUNT(*) FROM match_events e
                 WHERE e.fixture_id = m.fixture_id
                 AND e.event_type = 'Goal'
                 AND e.team_id = m.home_team_id) as event_home_goals,
                (SELECT COUNT(*) FROM match_events e
                 WHERE e.fixture_id = m.fixture_id
                 AND e.event_type = 'Goal'
                 AND e.team_id = m.away_team_id) as event_away_goals
            FROM matches m
            WHERE m.fixture_id IN (
                SELECT fixture_id FROM download_progress
                WHERE data_type = 'events' AND status = 'done'
            )
        """)
        mismatch_count = 0
        for row in cursor.fetchall():
            fid, home, away, ft_h, ft_a, evt_h, evt_a = row
            if ft_h != evt_h or ft_a != evt_a:
                mismatch_count += 1
                if mismatch_count <= 5:
                    print(
                        f"  ⚠️  골 수 불일치 fixture {fid}: "
                        f"스코어={ft_h}-{ft_a}, 이벤트={evt_h}-{evt_a}"
                    )

        if mismatch_count == 0:
            print("  ✅ 이벤트 골 수와 최종 스코어 일치")
        else:
            print(f"  ⚠️  총 {mismatch_count}경기에서 골 수 불일치 "
                  "(자책골, OG 등이 원인일 수 있음 — 수동 확인 필요)")

        # 3. 핵심 통계 지표 존재 확인
        key_stats = [
            "Ball Possession",
            "Total Shots",
            "Shots on Goal",
            "Expected Goals",
        ]
        cursor.execute("""
            SELECT DISTINCT stat_type FROM match_statistics
        """)
        available_stats = {row[0] for row in cursor.fetchall()}

        print(f"\n  통계 지표 확인:")
        for stat in key_stats:
            if stat in available_stats:
                print(f"    ✅ {stat}")
            else:
                print(f"    ❌ {stat} — 없음!")

        # xG가 없는 시즌 확인 (2017 이전에는 xG 미제공)
        cursor.execute("""
            SELECT m.season, COUNT(DISTINCT m.fixture_id) as total,
                   COUNT(DISTINCT CASE WHEN ms.stat_type = 'Expected Goals'
                         THEN m.fixture_id END) as with_xg
            FROM matches m
            LEFT JOIN match_statistics ms ON m.fixture_id = ms.fixture_id
            GROUP BY m.season
            ORDER BY m.season
        """)
        print(f"\n  시즌별 xG 커버리지:")
        for season, total, with_xg in cursor.fetchall():
            pct = (with_xg / total * 100) if total > 0 else 0
            status = "✅" if pct > 90 else "⚠️"
            print(f"    {status} {season}: {with_xg}/{total} ({pct:.0f}%)")

        print()

    # ── 진입점 ───────────────────────────────────────────

    def run(self, args):
        """CLI 인자에 따라 실행"""
        self._connect_db()

        try:
            if args.status:
                self.print_status()
            elif args.verify:
                self.verify_data()
            elif args.all:
                self.download_all()
            elif args.league and args.season:
                self.download_season(args.league, args.season)
            elif args.retry_errors:
                self.retry_errors()
            else:
                print("사용법:")
                print("  python -m src.data.api_football --all              # 전체 다운로드")
                print("  python -m src.data.api_football --league 39 --season 2023")
                print("  python -m src.data.api_football --status           # 진행 상태")
                print("  python -m src.data.api_football --verify           # 데이터 검증")
                print("  python -m src.data.api_football --retry-errors     # 실패 항목 재시도")
        finally:
            self._close_db()

    def retry_errors(self):
        """이전에 실패한 다운로드를 재시도한다."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fixture_id, data_type FROM download_progress
            WHERE status = 'error'
        """)
        errors = cursor.fetchall()

        if not errors:
            print("✅ 재시도할 오류가 없습니다.")
            return

        logger.info(f"🔄 {len(errors)}개 실패 항목 재시도")

        # 상태를 pending으로 리셋
        for fixture_id, data_type in errors:
            self.conn.execute("""
                UPDATE download_progress SET status = 'pending', error_message = NULL
                WHERE fixture_id = ? AND data_type = ?
            """, (fixture_id, data_type))
        self.conn.commit()

        # 재다운로드
        for i, (fixture_id, data_type) in enumerate(errors, 1):
            logger.info(f"  [{i}/{len(errors)}] fixture {fixture_id} / {data_type}")
            if data_type == "events":
                self.download_events(fixture_id)
            elif data_type == "statistics":
                self.download_statistics(fixture_id)
            elif data_type == "odds":
                self.download_odds(fixture_id)

        logger.info("🔄 재시도 완료")
        self.print_status()


# ─────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="API-Football 과거 데이터 다운로드 (Phase 1 학습용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.data.api_football --all                  # 5대 리그 × 5시즌 전체
  python -m src.data.api_football --league 39 --season 2023  # PL 2023-24
  python -m src.data.api_football --status               # 진행 상태 확인
  python -m src.data.api_football --verify               # 데이터 무결성 검증
  python -m src.data.api_football --retry-errors         # 실패 항목 재시도

리그 코드:
  39  = Premier League
  140 = La Liga
  135 = Serie A
  78  = Bundesliga
  61  = Ligue 1
        """,
    )

    parser.add_argument("--all", action="store_true",
                        help="5대 리그 × 5시즌 전체 다운로드")
    parser.add_argument("--league", type=int,
                        help="리그 ID (예: 39 = Premier League)")
    parser.add_argument("--season", type=int,
                        help="시즌 연도 (예: 2023)")
    parser.add_argument("--status", action="store_true",
                        help="다운로드 진행 상태 확인")
    parser.add_argument("--verify", action="store_true",
                        help="데이터 무결성 검증")
    parser.add_argument("--retry-errors", action="store_true",
                        help="이전에 실패한 다운로드 재시도")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help=f"SQLite DB 경로 (기본: {DB_PATH})")

    args = parser.parse_args()

    client = APIFootballClient(db_path=Path(args.db))
    client.run(args)


if __name__ == "__main__":
    main()
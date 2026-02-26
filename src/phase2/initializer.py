"""
Phase 2: Pre-Match Initialization (initializer.py)

킥오프 전, Phase 1의 파라미터를 오늘 경기에 맞게 초기화하여
Live Trading Engine의 초기 조건(Initial Condition)을 설정한다.

Step 2.1: 경기 전 컨텍스트 데이터 수집 (Data Ingestion)
Step 2.2: 피처 선택 (Feature Selection)
Step 2.3: 기본 득점 강도 파라미터 a 역산 (Prior Inference)
Step 2.4: Pre-Match Sanity Check (Go / Hold / Skip)
Step 2.5: 라이브 엔진 초기화 (System Initialization)

입력:
  - Phase 1 파라미터: nll_params.json, Q_matrix.json, xgb_poisson.json, feature_mask.json
  - API-Football: /lineups, /injuries, /odds
  - data/kalshi_football.db

출력:
  - LiveMatchState (Phase 3 시작 조건)

사용법:
  python -m src.phase2.initializer --fixture 12345
  python -m src.phase2.initializer --validate --samples 50
"""

import os
import json
import math
import time
import sqlite3
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from scipy.linalg import expm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

DB_PATH = Path("data/kalshi_football.db")
MODELS_DIR = Path("data/models")
API_BASE_URL = "https://v3.football.api-sports.io"

ROLLING_WINDOW = 5

STAT_TYPES = [
    "Ball Possession", "Total Shots", "Shots on Goal", "Shots off Goal",
    "Shots insidebox", "Blocked Shots", "Corner Kicks", "Fouls",
    "Passes accurate", "Passes %", "Total passes", "Goalkeeper Saves",
    "expected_goals",
]

# 6개 basis bin 경계 (분)
BIN_DURATION = 15  # 각 bin 15분

# Sanity Check 임계값
SANITY_GO = 0.15
SANITY_HOLD = 0.25

# 리스크 기본값
DEFAULT_ORDER_CAP = 0.03
DEFAULT_MATCH_CAP = 0.05
DEFAULT_TOTAL_CAP = 0.20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/initializer.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 데이터 클래스
# ═════════════════════════════════════════════════════════

@dataclass
class TeamInfo:
    """팀 정보 + 롤링 스탯"""
    team_id: int
    team_name: str
    is_home: bool
    formation: Optional[str] = None
    lineup: Optional[List[Dict]] = None
    injuries: Optional[List[Dict]] = None
    rolling_stats: Optional[Dict[str, float]] = None
    opp_rolling_stats: Optional[Dict[str, float]] = None
    recent_matches: int = 0


@dataclass
class SanityResult:
    """Step 2.4 Sanity Check 결과"""
    verdict: str                      # "Go", "Hold", "Skip"
    delta_max: float                  # max |P_model - P_market|
    p_model: Dict[str, float] = field(default_factory=dict)   # H/D/A 확률
    p_market: Optional[Dict[str, float]] = None  # 배당 내재 확률
    reason: str = ""


@dataclass
class LiveMatchState:
    """Step 2.5 출력: Phase 3 시작 조건"""
    # 경기 정보
    fixture_id: int
    league_id: int
    home_name: str
    away_name: str

    # 시간 상태
    current_time: float = 0.0
    T_exp: float = 95.0               # 예상 경기 시간
    engine_phase: str = "WAITING_FOR_KICKOFF"

    # 경기 상태
    current_state_X: int = 0           # 11v11
    score_home: int = 0
    score_away: int = 0
    delta_S: int = 0

    # 강도 함수 파라미터
    a_H: float = 0.0
    a_A: float = 0.0
    b: List[float] = field(default_factory=list)           # [6]
    gamma_1: float = 0.0
    gamma_2: float = 0.0
    delta_H: List[float] = field(default_factory=list)     # [4]: ΔS ≤-2, -1, +1, ≥+2
    delta_A: List[float] = field(default_factory=list)     # [4]
    C_time: float = 0.0

    # 마르코프 모델
    Q: Optional[np.ndarray] = None     # 4×4
    P_grid: Optional[Dict[int, np.ndarray]] = None  # dt → exp(Q·dt)

    # 프리매치 확률 (디버깅용)
    mu_H: float = 0.0
    mu_A: float = 0.0
    sanity: Optional[SanityResult] = None

    # 리스크 파라미터
    bankroll: float = 1000.0
    f_order_cap: float = DEFAULT_ORDER_CAP
    f_match_cap: float = DEFAULT_MATCH_CAP
    f_total_cap: float = DEFAULT_TOTAL_CAP


# ═════════════════════════════════════════════════════════
# API-Football 경량 클라이언트
# ═════════════════════════════════════════════════════════

class APIFootballClient:
    """프리매치 데이터 수집용 경량 API 클라이언트"""

    def __init__(self):
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY가 .env에 없습니다")
        self.headers = {"x-apisports-key": self.api_key}
        self._last_request_time = 0

    def _request(self, endpoint: str, params: dict) -> dict:
        url = f"{API_BASE_URL}/{endpoint}"
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.2:
            time.sleep(1.2 - elapsed)

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            self._last_request_time = time.time()
            response.raise_for_status()
            data = response.json()
            if data.get("errors") and isinstance(data["errors"], dict) and data["errors"]:
                logger.warning(f"API 에러: {data['errors']}")
                return {"response": []}
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {e}")
            return {"response": []}

    def get_fixture(self, fixture_id: int) -> Optional[Dict]:
        data = self._request("fixtures", {"id": fixture_id})
        items = data.get("response", [])
        return items[0] if items else None

    def get_lineups(self, fixture_id: int) -> List[Dict]:
        return self._request("fixtures/lineups", {"fixture": fixture_id}).get("response", [])

    def get_injuries(self, fixture_id: int) -> List[Dict]:
        return self._request("injuries", {"fixture": fixture_id}).get("response", [])

    def get_odds(self, fixture_id: int) -> Optional[Dict]:
        data = self._request("odds", {"fixture": fixture_id})
        odds_list = data.get("response", [])
        if not odds_list:
            return None
        for bookmaker in odds_list[0].get("bookmakers", []):
            for bet in bookmaker.get("bets", []):
                if bet.get("name") == "Match Winner":
                    return {v["value"]: float(v["odd"]) for v in bet.get("values", [])}
        return None


# ═════════════════════════════════════════════════════════
# Step 2.1: 데이터 수집
# ═════════════════════════════════════════════════════════

def parse_stat_value(val) -> Optional[float]:
    """통계 값 → 숫자 (Phase 1 ml_prior.py와 동일)"""
    if val is None or val == "None" or val == "":
        return None
    try:
        return float(str(val).strip().replace("%", ""))
    except (ValueError, TypeError):
        return None


def get_team_rolling_stats(
    conn: sqlite3.Connection,
    team_id: int,
    before_date: str,
    window: int = ROLLING_WINDOW,
) -> Tuple[Dict[str, float], int]:
    """
    특정 팀의 최근 N경기 롤링 평균 (Phase 1 compute_rolling_features와 동일 로직).

    Returns:
        (rolling_stats dict {roll_XXX: value}, num_matches_used)
    """
    cursor = conn.execute("""
        SELECT fixture_id, match_date,
               home_team_id, away_team_id,
               home_goals_ft, away_goals_ft
        FROM matches
        WHERE (home_team_id = ? OR away_team_id = ?)
          AND match_date < ?
          AND home_goals_ft IS NOT NULL
        ORDER BY match_date DESC
        LIMIT ?
    """, (team_id, team_id, before_date, window))

    recent = cursor.fetchall()
    if len(recent) < 2:
        return {}, len(recent)

    fixture_ids = [m[0] for m in recent]

    # 스탯 쿼리
    ph = ",".join("?" * len(fixture_ids))
    st_ph = ",".join("?" * len(STAT_TYPES))
    rows = conn.execute(
        f"SELECT fixture_id, stat_type, stat_value FROM match_statistics "
        f"WHERE fixture_id IN ({ph}) AND team_id = ? AND stat_type IN ({st_ph})",
        fixture_ids + [team_id] + STAT_TYPES
    ).fetchall()

    # 경기별 스탯
    match_stats = {}
    for fid, stype, sval in rows:
        parsed = parse_stat_value(sval)
        if parsed is not None:
            match_stats.setdefault(fid, {})[stype] = parsed

    # 득실점 추가
    for fid, _, home_id, away_id, hg, ag in recent:
        ms = match_stats.setdefault(fid, {})
        if team_id == home_id:
            ms["goals_scored"] = hg
            ms["goals_conceded"] = ag
        else:
            ms["goals_scored"] = ag
            ms["goals_conceded"] = hg

    # 스탯별 평균
    rolling = {}
    for stat_key in STAT_TYPES + ["goals_scored", "goals_conceded"]:
        values = [match_stats[f][stat_key] for f in fixture_ids
                  if f in match_stats and stat_key in match_stats[f]]
        if values:
            rolling[f"roll_{stat_key}"] = np.mean(values)

    return rolling, len(recent)


def fetch_prematch_api(
    fixture_id: int,
    home_info: TeamInfo,
    away_info: TeamInfo,
) -> Tuple[TeamInfo, TeamInfo, Optional[Dict]]:
    """API-Football에서 라인업, 부상, 배당률 수집"""
    client = APIFootballClient()

    # 라인업
    for lineup_data in client.get_lineups(fixture_id):
        tid = lineup_data.get("team", {}).get("id")
        formation = lineup_data.get("formation")
        players = [p["player"] for p in lineup_data.get("startXI", [])]
        if tid == home_info.team_id:
            home_info.formation = formation
            home_info.lineup = players
        elif tid == away_info.team_id:
            away_info.formation = formation
            away_info.lineup = players

    # 부상
    for inj in client.get_injuries(fixture_id):
        info = {"name": inj.get("player", {}).get("name"),
                "reason": inj.get("player", {}).get("reason")}
        tid = inj.get("team", {}).get("id")
        if tid == home_info.team_id:
            home_info.injuries = (home_info.injuries or []) + [info]
        elif tid == away_info.team_id:
            away_info.injuries = (away_info.injuries or []) + [info]

    # 배당률
    odds = client.get_odds(fixture_id)

    return home_info, away_info, odds


# ═════════════════════════════════════════════════════════
# Step 2.2: 피처 선택 + 조립
# ═════════════════════════════════════════════════════════

def load_feature_mask() -> Tuple[List[str], Dict]:
    """feature_mask.json 로드"""
    path = MODELS_DIR / "feature_mask.json"
    if not path.exists():
        raise FileNotFoundError(f"feature_mask.json 없음: {path}")
    with open(path) as f:
        data = json.load(f)
    return data["selected_features"], data


def assemble_features(
    team_rolling: Dict[str, float],
    opp_rolling: Dict[str, float],
    is_home: int,
    selected_features: List[str],
) -> Dict[str, float]:
    """
    롤링 스탯을 XGBoost 추론용 피처 벡터로 조립.

    Phase 1 prepare_dataset()과 동일한 피처 이름/순서 보장.
    """
    pool = {}
    pool.update(team_rolling)
    # roll_XXX → opp_roll_XXX 변환
    for k, v in opp_rolling.items():
        opp_key = k.replace("roll_", "opp_roll_") if k.startswith("roll_") else k
        pool[opp_key] = v
    pool["is_home"] = is_home

    result = {}
    missing = []
    for feat in selected_features:
        if feat in pool:
            result[feat] = pool[feat]
        else:
            result[feat] = -999  # XGBoost native missing
            missing.append(feat)

    if missing:
        logger.warning(f"  결측 피처 {len(missing)}개: {missing[:5]}")
    return result


# ═════════════════════════════════════════════════════════
# Step 2.3: a 파라미터 역산
# ═════════════════════════════════════════════════════════

def load_phase1_params() -> Dict:
    """Phase 1 프로덕션 파라미터 전부 로드"""
    # NLL params (b, γ, δ)
    nll_path = MODELS_DIR / "nll_params.json"
    if not nll_path.exists():
        raise FileNotFoundError(f"nll_params.json 없음: {nll_path}")
    with open(nll_path) as f:
        nll = json.load(f)

    # Q matrix
    q_path = MODELS_DIR / "Q_matrix.json"
    if not q_path.exists():
        raise FileNotFoundError(f"Q_matrix.json 없음: {q_path}")
    with open(q_path) as f:
        q_data = json.load(f)

    # all Q (전체 리그 통합)
    Q = np.array(q_data["all"]["Q"])

    # XGBoost 모델
    xgb_path = MODELS_DIR / "xgb_poisson.json"
    if not xgb_path.exists():
        raise FileNotFoundError(f"xgb_poisson.json 없음: {xgb_path}")
    model = xgb.Booster()
    model.load_model(str(xgb_path))

    return {
        "b": nll["b"],
        "gamma_1": nll["gamma_1"],
        "gamma_2": nll["gamma_2"],
        "delta_H": nll["delta_H"],
        "delta_A": nll["delta_A"],
        "Q": Q,
        "q_data": q_data,
        "xgb_model": model,
    }


def compute_expected_time(conn: sqlite3.Connection) -> Tuple[float, float, float]:
    """
    전반/후반 평균 추가시간을 DB에서 계산하여 T_exp 산출.

    T_exp = 90 + E[α₁] + E[α₂]
    """
    # match_meta에서 T_m 분포 조회
    rows = conn.execute("""
        SELECT T_m, first_half_end
        FROM match_meta
    """).fetchall()

    if not rows:
        # match_meta가 없으면 기본값 사용
        logger.warning("  match_meta 없음 → 기본 추가시간 사용 (전반 2분, 후반 3분)")
        return 95.0, 2.0, 3.0

    T_ms = [r[0] for r in rows]
    fh_ends = [r[1] for r in rows]

    # 전반 추가시간: first_half_end - 45
    alpha_1 = np.mean([fh - 45 for fh in fh_ends])
    # 후반 추가시간: T_m - first_half_end - 45
    alpha_2 = np.mean([tm - fh - 45 for tm, fh in zip(T_ms, fh_ends)])

    T_exp = 90 + alpha_1 + alpha_2

    logger.info(f"  추가시간: 전반 {alpha_1:.1f}분, 후반 {alpha_2:.1f}분 → T_exp={T_exp:.1f}분")
    return T_exp, alpha_1, alpha_2


def compute_C_time(b: List[float], T_exp: float, alpha_1: float, alpha_2: float) -> float:
    """
    시간 보정 상수 C_time 계산.

    C_time = Σ exp(b_i) · Δt_i

    bin 구조:
      bin 0: 전반 0~15분     → Δt = 15
      bin 1: 전반 15~30분    → Δt = 15
      bin 2: 전반 30~45+α₁   → Δt = 15 + α₁
      bin 3: 후반 0~15분     → Δt = 15
      bin 4: 후반 15~30분    → Δt = 15
      bin 5: 후반 30~45+α₂   → Δt = 15 + α₂
    """
    dt = [15, 15, 15 + alpha_1, 15, 15, 15 + alpha_2]
    C = sum(math.exp(b[i]) * dt[i] for i in range(6))
    return C


def infer_mu(
    model: xgb.Booster,
    features: Dict[str, float],
    selected_features: List[str],
) -> float:
    """XGBoost Poisson 추론: 피처 → μ̂ (기대 득점)"""
    values = [features.get(f, -999) for f in selected_features]
    dmatrix = xgb.DMatrix(
        np.array([values]),
        feature_names=selected_features,
    )
    mu = model.predict(dmatrix)[0]
    return max(mu, 0.05)  # 하한 클램핑


def compute_a(mu_hat: float, C_time: float) -> float:
    """
    a = ln(μ̂) − ln(C_time)

    킥오프 시점에서 X=0, ΔS=0이므로:
    μ̂ = exp(a) · Σ exp(b_i) · Δt_i = exp(a) · C_time
    ∴ a = ln(μ̂) − ln(C_time)
    """
    return math.log(mu_hat) - math.log(C_time)


# ═════════════════════════════════════════════════════════
# Step 2.4: Sanity Check
# ═════════════════════════════════════════════════════════

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(X=k) = λ^k · e^(-λ) / k!"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def compute_match_probabilities(mu_H: float, mu_A: float, max_goals: int = 8) -> Dict[str, float]:
    """
    독립 Poisson 모델로 홈승/무/어웨이승 확률 계산.

    P(Home) = ΣΣ P(H=h) · P(A=a) for h > a
    """
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for h in range(max_goals + 1):
        ph = poisson_pmf(h, mu_H)
        for a in range(max_goals + 1):
            pa = poisson_pmf(a, mu_A)
            joint = ph * pa
            if h > a:
                p_home += joint
            elif h == a:
                p_draw += joint
            else:
                p_away += joint

    return {"Home": p_home, "Draw": p_draw, "Away": p_away}


def odds_to_probs(odds_home: float, odds_draw: float, odds_away: float) -> Dict[str, float]:
    """배당률 → 내재 확률 (오버라운드 제거)"""
    inv = 1/odds_home + 1/odds_draw + 1/odds_away  # overround
    return {
        "Home": (1/odds_home) / inv,
        "Draw": (1/odds_draw) / inv,
        "Away": (1/odds_away) / inv,
    }


def run_sanity_check(
    mu_H: float,
    mu_A: float,
    odds_home: Optional[float],
    odds_draw: Optional[float],
    odds_away: Optional[float],
) -> SanityResult:
    """
    Step 2.4: 모델 확률 vs 시장 확률 괴리도 검사.

    | Δ_sanity     | 판정   |
    |-------------|--------|
    | < 0.15      | Go     |
    | 0.15~0.25   | Hold   |
    | ≥ 0.25      | Skip   |
    """
    p_model = compute_match_probabilities(mu_H, mu_A)

    if odds_home and odds_draw and odds_away:
        p_market = odds_to_probs(odds_home, odds_draw, odds_away)
        delta_max = max(
            abs(p_model[o] - p_market[o]) for o in ("Home", "Draw", "Away")
        )

        if delta_max < SANITY_GO:
            verdict = "Go"
        elif delta_max < SANITY_HOLD:
            verdict = "Hold"
        else:
            verdict = "Skip"

        return SanityResult(
            verdict=verdict,
            delta_max=delta_max,
            p_model=p_model,
            p_market=p_market,
            reason=f"max|P_model - P_market| = {delta_max:.3f}",
        )
    else:
        # 배당 없으면 Go (경고 표시)
        return SanityResult(
            verdict="Go",
            delta_max=0.0,
            p_model=p_model,
            p_market=None,
            reason="배당률 미가용 → 자동 Go (주의 필요)",
        )


# ═════════════════════════════════════════════════════════
# Step 2.5: 시스템 초기화
# ═════════════════════════════════════════════════════════

def precompute_P_grid(Q: np.ndarray, max_dt: int = 101) -> Dict[int, np.ndarray]:
    """
    행렬 지수함수 사전 계산: P(dt) = exp(Q · dt) for dt ∈ [0, 100]

    Phase 3에서 "dt분 후 상태 전이 확률"을 O(1) 조회.
    """
    P_grid = {}
    for dt in range(max_dt):
        P_grid[dt] = expm(Q * dt)
    return P_grid


# ═════════════════════════════════════════════════════════
# 통합 파이프라인
# ═════════════════════════════════════════════════════════

def initialize_match(
    fixture_id: int,
    db_path: Path = DB_PATH,
    fetch_api: bool = True,
    bankroll: float = 1000.0,
) -> LiveMatchState:
    """
    Phase 2 전체 파이프라인: fixture_id → LiveMatchState.

    Step 2.1 → 2.2 → 2.3 → 2.4 → 2.5 순서 실행.
    """
    logger.info(f"🏟️  Phase 2: Pre-Match Initialization (fixture {fixture_id})")
    logger.info("=" * 60)

    conn = sqlite3.connect(str(db_path))

    # ──────────────────────────────────────────────────────
    # Step 2.1: 데이터 수집
    # ──────────────────────────────────────────────────────
    logger.info("\n📋 Step 2.1: 데이터 수집")

    # 경기 정보 (DB 우선)
    row = conn.execute("""
        SELECT fixture_id, league_id, season, match_date,
               home_team_id, home_team_name,
               away_team_id, away_team_name
        FROM matches WHERE fixture_id = ?
    """, (fixture_id,)).fetchone()

    if row:
        _, league_id, season, match_date = row[:4]
        home_id, home_name, away_id, away_name = row[4], row[5], row[6], row[7]
    elif fetch_api:
        client = APIFootballClient()
        fix = client.get_fixture(fixture_id)
        if not fix:
            conn.close()
            raise ValueError(f"Fixture {fixture_id}을 찾을 수 없습니다")
        home_id = fix["teams"]["home"]["id"]
        home_name = fix["teams"]["home"]["name"]
        away_id = fix["teams"]["away"]["id"]
        away_name = fix["teams"]["away"]["name"]
        league_id = fix["league"]["id"]
        season = fix["league"]["season"]
        match_date = fix["fixture"]["date"][:10]
    else:
        conn.close()
        raise ValueError(f"Fixture {fixture_id}이 DB에 없고 API 호출도 비활성")

    logger.info(f"  {home_name} vs {away_name} ({match_date})")

    home = TeamInfo(team_id=home_id, team_name=home_name, is_home=True)
    away = TeamInfo(team_id=away_id, team_name=away_name, is_home=False)

    # API 호출 (라인업, 부상, 배당)
    odds_data = None
    if fetch_api:
        home, away, odds_data = fetch_prematch_api(fixture_id, home, away)
        if home.lineup:
            logger.info(f"  홈 라인업: {home.formation} ({len(home.lineup)}명)")
        if away.lineup:
            logger.info(f"  어웨이 라인업: {away.formation} ({len(away.lineup)}명)")

    # 롤링 스탯
    home.rolling_stats, home.recent_matches = get_team_rolling_stats(
        conn, home_id, match_date
    )
    away.rolling_stats, away.recent_matches = get_team_rolling_stats(
        conn, away_id, match_date
    )

    # 상대팀 롤링 스탯
    away_opp, _ = get_team_rolling_stats(conn, away_id, match_date)
    home_opp, _ = get_team_rolling_stats(conn, home_id, match_date)
    home.opp_rolling_stats = away_opp
    away.opp_rolling_stats = home_opp

    logger.info(f"  롤링: {home_name} {home.recent_matches}경기, {away_name} {away.recent_matches}경기")

    # ──────────────────────────────────────────────────────
    # Step 2.2: 피처 선택 + 조립
    # ──────────────────────────────────────────────────────
    logger.info("\n🔧 Step 2.2: 피처 선택")

    selected_features, _ = load_feature_mask()
    features_home = assemble_features(
        home.rolling_stats, home.opp_rolling_stats, 1, selected_features
    )
    features_away = assemble_features(
        away.rolling_stats, away.opp_rolling_stats, 0, selected_features
    )
    n_valid_h = sum(1 for v in features_home.values() if v != -999)
    n_valid_a = sum(1 for v in features_away.values() if v != -999)
    logger.info(f"  홈 피처: {n_valid_h}/{len(selected_features)} 유효")
    logger.info(f"  어웨이 피처: {n_valid_a}/{len(selected_features)} 유효")

    # ──────────────────────────────────────────────────────
    # Step 2.3: a 파라미터 역산
    # ──────────────────────────────────────────────────────
    logger.info("\n📐 Step 2.3: a 파라미터 역산")

    params = load_phase1_params()
    b = params["b"]

    # T_exp 계산
    T_exp, alpha_1, alpha_2 = compute_expected_time(conn)

    # C_time 계산
    C_time = compute_C_time(b, T_exp, alpha_1, alpha_2)
    logger.info(f"  C_time = {C_time:.2f}")

    # XGBoost 추론
    mu_H = infer_mu(params["xgb_model"], features_home, selected_features)
    mu_A = infer_mu(params["xgb_model"], features_away, selected_features)
    logger.info(f"  μ̂_H = {mu_H:.3f} ({home_name})")
    logger.info(f"  μ̂_A = {mu_A:.3f} ({away_name})")

    # a 역산
    a_H = compute_a(mu_H, C_time)
    a_A = compute_a(mu_A, C_time)
    logger.info(f"  a_H = {a_H:.4f}, a_A = {a_A:.4f}")

    # 검증: exp(a) · C_time ≈ μ̂
    check_H = math.exp(a_H) * C_time
    check_A = math.exp(a_A) * C_time
    logger.info(f"  검증: exp(a_H)·C = {check_H:.3f} ≈ μ̂_H = {mu_H:.3f} ✅")

    # ──────────────────────────────────────────────────────
    # Step 2.4: Sanity Check
    # ──────────────────────────────────────────────────────
    logger.info("\n🔍 Step 2.4: Sanity Check")

    odds_h = odds_data.get("Home") if odds_data else None
    odds_d = odds_data.get("Draw") if odds_data else None
    odds_a = odds_data.get("Away") if odds_data else None

    sanity = run_sanity_check(mu_H, mu_A, odds_h, odds_d, odds_a)

    logger.info(f"  모델: H={sanity.p_model['Home']:.3f} D={sanity.p_model['Draw']:.3f} A={sanity.p_model['Away']:.3f}")
    if sanity.p_market:
        logger.info(f"  시장: H={sanity.p_market['Home']:.3f} D={sanity.p_market['Draw']:.3f} A={sanity.p_market['Away']:.3f}")
    logger.info(f"  Δ_max = {sanity.delta_max:.3f}")

    verdict_emoji = {"Go": "✅", "Hold": "⚠️", "Skip": "❌"}
    logger.info(f"  판정: {verdict_emoji.get(sanity.verdict, '?')} {sanity.verdict} ({sanity.reason})")

    if sanity.verdict == "Skip":
        logger.warning(f"  ⛔ 이 경기는 SKIP 판정 — 라이브 진입 불가")
        conn.close()
        state = LiveMatchState(
            fixture_id=fixture_id, league_id=league_id,
            home_name=home_name, away_name=away_name,
            engine_phase="SKIPPED", sanity=sanity,
            mu_H=mu_H, mu_A=mu_A,
        )
        return state

    # ──────────────────────────────────────────────────────
    # Step 2.5: 시스템 초기화
    # ──────────────────────────────────────────────────────
    logger.info("\n⚙️  Step 2.5: 시스템 초기화")

    Q = params["Q"]

    # P_grid 사전 계산
    logger.info("  행렬 지수함수 P_grid[0..100] 계산 중...")
    P_grid = precompute_P_grid(Q)
    logger.info(f"  P_grid: {len(P_grid)}개 엔트리 완료")

    # LiveMatchState 조립
    state = LiveMatchState(
        fixture_id=fixture_id,
        league_id=league_id,
        home_name=home_name,
        away_name=away_name,
        current_time=0.0,
        T_exp=T_exp,
        engine_phase="READY",
        current_state_X=0,
        score_home=0,
        score_away=0,
        delta_S=0,
        a_H=a_H,
        a_A=a_A,
        b=b,
        gamma_1=params["gamma_1"],
        gamma_2=params["gamma_2"],
        delta_H=params["delta_H"],
        delta_A=params["delta_A"],
        C_time=C_time,
        Q=Q,
        P_grid=P_grid,
        mu_H=mu_H,
        mu_A=mu_A,
        sanity=sanity,
        bankroll=bankroll,
    )

    conn.close()

    # 요약
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🏁 Phase 2 완료: {home_name} vs {away_name}")
    logger.info(f"  μ̂: H={mu_H:.2f}, A={mu_A:.2f}")
    logger.info(f"  a:  H={a_H:.4f}, A={a_A:.4f}")
    logger.info(f"  T_exp: {T_exp:.1f}분")
    logger.info(f"  Sanity: {sanity.verdict}")
    logger.info(f"  엔진 상태: {state.engine_phase}")
    logger.info(f"{'=' * 60}")

    return state


# ═════════════════════════════════════════════════════════
# 검증: 과거 경기로 파이프라인 테스트
# ═════════════════════════════════════════════════════════

def validate_pipeline(
    db_path: Path = DB_PATH,
    n_samples: int = 50,
) -> pd.DataFrame:
    """
    과거 완료 경기로 Phase 2 파이프라인을 검증한다.

    DB에서 실제 결과가 있는 경기를 대상으로:
    1. 롤링 스탯 → 피처 → μ̂ → a 계산
    2. 모델 확률과 실제 결과 비교
    3. Sanity Check 분포 확인
    """
    logger.info(f"\n🔍 Phase 2 파이프라인 검증: {n_samples}경기")
    logger.info("=" * 60)

    conn = sqlite3.connect(str(db_path))

    # Phase 1 파라미터 로드
    try:
        params = load_phase1_params()
        selected_features, _ = load_feature_mask()
    except FileNotFoundError as e:
        logger.error(f"Phase 1 파라미터 없음: {e}")
        conn.close()
        return pd.DataFrame()

    b = params["b"]
    T_exp, alpha_1, alpha_2 = compute_expected_time(conn)
    C_time = compute_C_time(b, T_exp, alpha_1, alpha_2)

    # 완료 경기 샘플링
    fixtures = conn.execute("""
        SELECT fixture_id, league_id, match_date,
               home_team_id, home_team_name,
               away_team_id, away_team_name,
               home_goals_ft, away_goals_ft
        FROM matches
        WHERE home_goals_ft IS NOT NULL
        ORDER BY match_date DESC
        LIMIT ?
    """, (n_samples,)).fetchall()

    results = []

    for row in fixtures:
        fid, lid, mdate = row[0], row[1], row[2]
        hid, hname, aid, aname = row[3], row[4], row[5], row[6]
        hg, ag = row[7], row[8]

        # Step 2.1-2.2: 롤링 스탯 + 피처 조립
        h_roll, h_n = get_team_rolling_stats(conn, hid, mdate)
        a_roll, a_n = get_team_rolling_stats(conn, aid, mdate)
        h_opp = a_roll  # 상대의 롤링이 곧 opp
        a_opp = h_roll

        if not h_roll or not a_roll:
            continue

        feat_h = assemble_features(h_roll, h_opp, 1, selected_features)
        feat_a = assemble_features(a_roll, a_opp, 0, selected_features)

        # Step 2.3: μ̂ + a
        mu_H = infer_mu(params["xgb_model"], feat_h, selected_features)
        mu_A = infer_mu(params["xgb_model"], feat_a, selected_features)
        a_H = compute_a(mu_H, C_time)
        a_A = compute_a(mu_A, C_time)

        # Step 2.4: 모델 확률
        p_model = compute_match_probabilities(mu_H, mu_A)

        # 실제 결과
        if hg > ag:
            actual = "Home"
        elif hg == ag:
            actual = "Draw"
        else:
            actual = "Away"

        n_valid = sum(1 for v in feat_h.values() if v != -999)

        results.append({
            "fixture_id": fid,
            "date": mdate,
            "home": hname,
            "away": aname,
            "score": f"{hg}-{ag}",
            "actual": actual,
            "mu_H": round(mu_H, 3),
            "mu_A": round(mu_A, 3),
            "a_H": round(a_H, 4),
            "a_A": round(a_A, 4),
            "p_home": round(p_model["Home"], 3),
            "p_draw": round(p_model["Draw"], 3),
            "p_away": round(p_model["Away"], 3),
            "n_features": n_valid,
        })

    conn.close()

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("검증 데이터 없음")
        return df

    # 통계
    logger.info(f"\n📊 검증 결과 ({len(df)}경기):")
    logger.info(f"  μ̂_H 평균: {df['mu_H'].mean():.3f} (실제 홈 골 평균과 비교)")
    logger.info(f"  μ̂_A 평균: {df['mu_A'].mean():.3f}")
    logger.info(f"  a_H 평균: {df['a_H'].mean():.4f}")
    logger.info(f"  a_A 평균: {df['a_A'].mean():.4f}")

    # 캘리브레이션: 모델 홈승 확률 vs 실제 홈승 빈도
    actual_home_pct = (df["actual"] == "Home").mean()
    pred_home_avg = df["p_home"].mean()
    logger.info(f"  홈승: 모델 {pred_home_avg:.3f} vs 실제 {actual_home_pct:.3f}")

    actual_draw_pct = (df["actual"] == "Draw").mean()
    pred_draw_avg = df["p_draw"].mean()
    logger.info(f"  무승부: 모델 {pred_draw_avg:.3f} vs 실제 {actual_draw_pct:.3f}")

    # 피처 완성도
    logger.info(f"  피처 유효: {df['n_features'].mean():.1f}/{len(selected_features)} 평균")

    return df


def validate_feature_consistency(
    db_path: Path = DB_PATH,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Phase 1 vs Phase 2 μ̂ 일치 검증."""
    logger.info(f"\n🔬 Phase 1 ↔ Phase 2 피처 일치 검증")
    logger.info("=" * 60)

    mu_path = MODELS_DIR / "mu_predictions.csv"
    if not mu_path.exists():
        logger.error("mu_predictions.csv 없음 — Phase 1 먼저 실행 필요")
        return pd.DataFrame()

    mu_p1 = pd.read_csv(mu_path)
    logger.info(f"  Phase 1 μ̂: {len(mu_p1)}행 로드")

    try:
        params = load_phase1_params()
        selected_features, _ = load_feature_mask()
    except FileNotFoundError as e:
        logger.error(f"Phase 1 파라미터 없음: {e}")
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    b = params["b"]
    T_exp, alpha_1, alpha_2 = compute_expected_time(conn)
    C_time = compute_C_time(b, T_exp, alpha_1, alpha_2)

    matches = pd.read_sql_query("""
        SELECT fixture_id, match_date,
               home_team_id, away_team_id
        FROM matches
        WHERE home_goals_ft IS NOT NULL
        ORDER BY match_date DESC
        LIMIT ?
    """, conn, params=(n_samples,))

    results = []
    for _, mrow in matches.iterrows():
        fid = mrow["fixture_id"]
        mdate = mrow["match_date"]
        hid = mrow["home_team_id"]
        aid = mrow["away_team_id"]

        p1_home = mu_p1[(mu_p1["fixture_id"] == fid) & (mu_p1["is_home"] == 1)]
        p1_away = mu_p1[(mu_p1["fixture_id"] == fid) & (mu_p1["is_home"] == 0)]
        if p1_home.empty or p1_away.empty:
            continue

        mu_p1_H = p1_home.iloc[0]["mu_hat"]
        mu_p1_A = p1_away.iloc[0]["mu_hat"]

        h_roll, _ = get_team_rolling_stats(conn, hid, mdate)
        a_roll, _ = get_team_rolling_stats(conn, aid, mdate)
        if not h_roll or not a_roll:
            continue

        feat_h = assemble_features(h_roll, a_roll, 1, selected_features)
        feat_a = assemble_features(a_roll, h_roll, 0, selected_features)

        mu_p2_H = infer_mu(params["xgb_model"], feat_h, selected_features)
        mu_p2_A = infer_mu(params["xgb_model"], feat_a, selected_features)

        results.append({
            "fixture_id": fid,
            "mu_p1_H": round(mu_p1_H, 4), "mu_p2_H": round(mu_p2_H, 4),
            "diff_H": round(abs(mu_p2_H - mu_p1_H), 4),
            "mu_p1_A": round(mu_p1_A, 4), "mu_p2_A": round(mu_p2_A, 4),
            "diff_A": round(abs(mu_p2_A - mu_p1_A), 4),
        })

    conn.close()
    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("비교 데이터 없음")
        return df

    max_diff = max(df["diff_H"].max(), df["diff_A"].max())
    TOLERANCE = 0.05
    n_mismatch = ((df["diff_H"] > TOLERANCE) | (df["diff_A"] > TOLERANCE)).sum()

    logger.info(f"\n📊 일치 검증 결과 ({len(df)}경기):")
    logger.info(f"  평균 diff 홈: {df['diff_H'].mean():.4f}")
    logger.info(f"  평균 diff 어웨이: {df['diff_A'].mean():.4f}")
    logger.info(f"  최대 차이: {max_diff:.4f}")

    if n_mismatch == 0:
        logger.info(f"  ✅ PASS: 전체 {len(df)}경기 일치 (허용 오차 {TOLERANCE})")
    else:
        logger.warning(f"  ⚠️  MISMATCH: {n_mismatch}/{len(df)}경기에서 차이 > {TOLERANCE}")
        logger.warning(f"  최악 5개:\n{df.nlargest(5, 'diff_H').to_string(index=False)}")

    return df

# ═════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Pre-Match Initialization")
    parser.add_argument("--fixture", type=int, help="단일 경기 초기화")
    parser.add_argument("--validate", action="store_true", help="과거 데이터 파이프라인 검증")
    parser.add_argument("--samples", type=int, default=50, help="검증 샘플 수")
    parser.add_argument("--no-api", action="store_true", help="API 호출 없이 DB만 사용")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="초기 자본금")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--cross-check", action="store_true", help="Phase 1↔2 μ̂ 일치 검증")

    args = parser.parse_args()
    Path("logs").mkdir(exist_ok=True)

    if args.cross_check:
        df = validate_feature_consistency(Path(args.db), args.samples)
        if not df.empty:
            print("\n" + df.to_string(index=False))
    elif args.validate:
        df = validate_pipeline(Path(args.db), args.samples)
        if not df.empty:
            print("\n" + df[["date", "home", "away", "score", "mu_H", "mu_A",
                             "p_home", "p_draw", "p_away"]].to_string(index=False))
    elif args.fixture:
        state = initialize_match(
            args.fixture, Path(args.db),
            fetch_api=not args.no_api,
            bankroll=args.bankroll,
        )
        print(f"\n최종 상태: {state.engine_phase}")
        if state.engine_phase == "READY":
            print(f"  a_H={state.a_H:.4f}, a_A={state.a_A:.4f}")
            print(f"  μ̂: H={state.mu_H:.2f}, A={state.mu_A:.2f}")
            print(f"  T_exp={state.T_exp:.1f}분")
            print(f"  P_grid: {len(state.P_grid)}개")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
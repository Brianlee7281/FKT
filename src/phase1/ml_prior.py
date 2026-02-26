"""
Step 1.3: 프리매치 Prior 파라미터 a 학습 (XGBoost Poisson).

경기별 전력차를 반영한 기본 득점 강도의 초기 추정치 μ̂를 학습한다.
이 값은 Step 1.4 Joint NLL의 시작점(initialization)으로 사용된다.

피처: 과거 5경기 롤링 평균 (점유율, 슈팅, xG 등) + 상대팀 스탯
타겟: 해당 경기에서 해당 팀의 총 득점 수

입력:
  - matches, match_statistics 테이블

출력:
  - data/models/xgb_poisson.json (모델 가중치)
  - data/models/feature_mask.json (선택된 피처 목록)
  - data/models/mu_predictions.json (전체 경기의 μ̂ 예측값 → Step 1.4 초기값)

사용법:
  python -m src.phase1.ml_prior
  python -m src.phase1.ml_prior --validate   # CV 결과만 보기
"""

import json
import sqlite3
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

DB_PATH = Path("data/kalshi_football.db")
OUTPUT_DIR = Path("data/models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ml_prior.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# 롤링 윈도우 크기
ROLLING_WINDOW = 5

# 사용할 통계 항목 (match_statistics.stat_type)
STAT_TYPES = [
    "Ball Possession",       # 점유율 (%)
    "Total Shots",           # 총 슈팅
    "Shots on Goal",         # 유효 슈팅
    "Shots off Goal",        # 빗나간 슈팅
    "Shots insidebox",       # 박스 안 슈팅
    "Blocked Shots",         # 블록된 슈팅
    "Corner Kicks",          # 코너킥
    "Fouls",                 # 파울
    "Passes accurate",       # 정확한 패스 수
    "Passes %",              # 패스 성공률 (%)
    "Total passes",          # 총 패스
    "Goalkeeper Saves",      # GK 세이브
    "expected_goals",        # xG (일부 경기만)
]


# ─────────────────────────────────────────────────────────
# 1. 데이터 로딩
# ─────────────────────────────────────────────────────────

def load_match_data(db_path: Path) -> pd.DataFrame:
    """
    matches + match_statistics를 조인하여
    경기별-팀별 피처 테이블을 만든다.
    """
    conn = sqlite3.connect(str(db_path))

    # 경기 메타
    matches = pd.read_sql_query("""
        SELECT fixture_id, league_id, season, match_date,
               home_team_id, home_team_name,
               away_team_id, away_team_name,
               home_goals_ft, away_goals_ft
        FROM matches
        WHERE home_goals_ft IS NOT NULL
        ORDER BY match_date
    """, conn)

    # 통계 데이터
    stats = pd.read_sql_query("""
        SELECT fixture_id, team_id, stat_type, stat_value
        FROM match_statistics
        WHERE stat_type IN ({})
    """.format(",".join(f"'{s}'" for s in STAT_TYPES)), conn)

    conn.close()

    logger.info(f"  로드: {len(matches)}경기, {len(stats)}통계 레코드")

    return matches, stats


def parse_stat_value(val: str) -> Optional[float]:
    """통계 값 문자열 → 숫자 변환"""
    if val is None or val == "None" or val == "":
        return None
    try:
        # "55%" → 55.0
        val = str(val).strip().replace("%", "")
        return float(val)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────
# 2. 피처 엔지니어링
# ─────────────────────────────────────────────────────────

def build_team_stats_table(matches: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    경기별-팀별 스탯 피벗 테이블 생성.

    결과: (fixture_id, team_id, 'Ball Possession', 'Total Shots', ...)
    """
    # 값 파싱
    stats["value"] = stats["stat_value"].apply(parse_stat_value)
    stats = stats.dropna(subset=["value"])

    # 피벗: fixture_id × team_id → stat columns
    pivot = stats.pivot_table(
        index=["fixture_id", "team_id"],
        columns="stat_type",
        values="value",
        aggfunc="first",  # 중복 시 첫 번째 값
    ).reset_index()

    # 컬럼 이름 정리
    pivot.columns.name = None

    return pivot


def compute_rolling_features(matches: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    각 경기에 대해, 해당 팀의 과거 5경기 롤링 평균을 계산한다.

    중요: Lookahead bias 방지!
    경기 m의 피처는 m 이전 경기들의 통계만 사용한다.
    """
    logger.info("  롤링 피처 계산 중...")

    # 홈/어웨이를 하나의 팀-경기 행으로 풀어헤침 (unpivot)
    rows = []

    for _, m in matches.iterrows():
        fid = m["fixture_id"]
        date = m["match_date"]

        # 홈팀 행
        rows.append({
            "fixture_id": fid,
            "match_date": date,
            "league_id": m["league_id"],
            "season": m["season"],
            "team_id": m["home_team_id"],
            "team_name": m["home_team_name"],
            "opponent_id": m["away_team_id"],
            "is_home": 1,
            "goals_scored": m["home_goals_ft"],
            "goals_conceded": m["away_goals_ft"],
        })
        # 어웨이팀 행
        rows.append({
            "fixture_id": fid,
            "match_date": date,
            "league_id": m["league_id"],
            "season": m["season"],
            "team_id": m["away_team_id"],
            "team_name": m["away_team_name"],
            "opponent_id": m["home_team_id"],
            "is_home": 0,
            "goals_scored": m["away_goals_ft"],
            "goals_conceded": m["home_goals_ft"],
        })

    team_matches = pd.DataFrame(rows)

    # 팀 스탯 조인
    team_matches = team_matches.merge(
        team_stats, on=["fixture_id", "team_id"], how="left"
    )

    # 날짜순 정렬
    team_matches = team_matches.sort_values(["team_id", "match_date"]).reset_index(drop=True)

    # 롤링할 스탯 컬럼
    stat_cols = [c for c in team_stats.columns if c not in ("fixture_id", "team_id")]
    # 득실점도 롤링
    stat_cols_with_goals = stat_cols + ["goals_scored", "goals_conceded"]

    # 팀별 롤링 평균 (shift(1)로 현재 경기 제외 → lookahead bias 방지!)
    rolling_cols = {}
    for col in stat_cols_with_goals:
        rolling_cols[f"roll_{col}"] = (
            team_matches.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean())
        )

    rolling_df = pd.DataFrame(rolling_cols, index=team_matches.index)
    team_matches = pd.concat([team_matches, rolling_df], axis=1)

    # 상대팀 롤링 스탯도 가져오기
    # (상대의 최근 5경기 평균 → "강한 상대" vs "약한 상대" 구분)
    opp_features = team_matches[
        ["fixture_id", "team_id"] + [f"roll_{c}" for c in stat_cols_with_goals]
    ].copy()
    opp_features = opp_features.rename(
        columns={c: c.replace("roll_", "opp_roll_") for c in opp_features.columns if c.startswith("roll_")}
    )
    opp_features = opp_features.rename(columns={"team_id": "opponent_id"})

    team_matches = team_matches.merge(
        opp_features, on=["fixture_id", "opponent_id"], how="left"
    )

    logger.info(f"  팀-경기 행: {len(team_matches)}, 피처 컬럼: {len([c for c in team_matches.columns if 'roll_' in c])}")

    return team_matches


# ─────────────────────────────────────────────────────────
# 3. 학습 데이터셋 구성
# ─────────────────────────────────────────────────────────

def prepare_dataset(team_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    XGBoost 학습/검증에 사용할 X, y를 구성한다.

    X: 롤링 피처 + is_home
    y: goals_scored (Poisson 타겟)
    """
    feature_cols = sorted([c for c in team_matches.columns if "roll_" in c])
    feature_cols.append("is_home")

    # 결측치가 너무 많은 행 제거 (최소 첫 2경기 롤링 필요)
    mask = team_matches[feature_cols].notna().sum(axis=1) >= len(feature_cols) * 0.5
    df = team_matches[mask].copy()

    X = df[feature_cols].copy()
    y = df["goals_scored"].copy()

    # 결측치 → -999 (XGBoost는 결측치를 자동 처리함)
    X = X.fillna(-999)

    logger.info(f"  학습 데이터: {len(X)}행, {len(feature_cols)}피처")
    logger.info(f"  타겟 분포: mean={y.mean():.3f}, std={y.std():.3f}")

    # 인덱스 보존 (나중에 fixture_id 매핑용)
    X.index = df.index

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────
# 4. 모델 학습
# ─────────────────────────────────────────────────────────

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    team_matches: pd.DataFrame,
) -> Tuple[xgb.Booster, Dict]:
    """
    XGBoost Poisson 회귀 모델 학습.

    Time-Series Split으로 검증하여 과적합 방지.
    """
    logger.info("\n📈 XGBoost 학습 시작")

    # ── 시계열 분할 (데이터가 날짜순이므로 단순 인덱스 분할) ──
    n = len(X)
    train_end = int(n * 0.8)
    X_train, X_val = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_val = y.iloc[:train_end], y.iloc[train_end:]

    logger.info(f"  Train: {len(X_train)}행, Val: {len(X_val)}행")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,          # L1
        "reg_lambda": 1.0,         # L2
        "seed": 42,
        "verbosity": 0,
    }

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=100,
    )

    best_round = model.best_iteration
    train_loss = evals_result["train"]["poisson-nloglik"][best_round]
    val_loss = evals_result["val"]["poisson-nloglik"][best_round]

    logger.info(f"  Best round: {best_round}")
    logger.info(f"  Train NLL: {train_loss:.4f}")
    logger.info(f"  Val NLL:   {val_loss:.4f}")

    # ── 검증 세트 예측 품질 ──
    y_pred = model.predict(dval)
    # Poisson deviance
    mask_pos = y_val > 0
    deviance = 2 * np.sum(
        np.where(mask_pos, y_val * np.log(y_val / y_pred), 0) - (y_val - y_pred)
    )
    mean_deviance = deviance / len(y_val)

    # 단순 비교: 평균 예측 vs 실제
    logger.info(f"\n  검증 세트 성능:")
    logger.info(f"    예측 평균: {y_pred.mean():.3f}")
    logger.info(f"    실제 평균: {y_val.mean():.3f}")
    logger.info(f"    Mean Poisson Deviance: {mean_deviance:.4f}")

    # 예측 구간별 정확도
    for bucket_low, bucket_high in [(0, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 3.0)]:
        mask = (y_pred >= bucket_low) & (y_pred < bucket_high)
        if mask.sum() > 0:
            actual = y_val[mask].mean()
            pred = y_pred[mask].mean()
            logger.info(f"    μ̂ ∈ [{bucket_low}, {bucket_high}): "
                        f"pred={pred:.2f}, actual={actual:.2f}, n={mask.sum()}")

    return model, evals_result


# ─────────────────────────────────────────────────────────
# 5. 피처 중요도 + 마스크
# ─────────────────────────────────────────────────────────

def compute_feature_mask(model: xgb.Booster, feature_cols: List[str], threshold: float = 0.95) -> List[str]:
    """
    XGBoost feature importance (gain 기준)로 피처 선택.
    누적 중요도가 threshold에 도달하는 상위 피처를 선택한다.
    """
    importance = model.get_score(importance_type="gain")

    # 정렬
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in sorted_feats)

    if total_gain == 0:
        logger.warning("  ⚠️  피처 중요도가 전부 0")
        return feature_cols

    logger.info(f"\n📊 피처 중요도 (gain):")
    cumulative = 0
    selected = []
    for feat, gain in sorted_feats:
        pct = gain / total_gain * 100
        cumulative += pct
        marker = "✅" if cumulative <= threshold * 100 + pct else "  "
        logger.info(f"  {marker} {feat:<40} {gain:>10.1f} ({pct:5.1f}%, 누적 {cumulative:5.1f}%)")

        if cumulative <= threshold * 100 + pct:  # threshold까지의 피처 + 마지막 하나
            selected.append(feat)

        if cumulative >= threshold * 100 and len(selected) >= 3:
            break

    # is_home은 항상 포함
    if "is_home" not in selected:
        selected.append("is_home")

    logger.info(f"\n  선택된 피처: {len(selected)}/{len(feature_cols)}")

    return selected


# ─────────────────────────────────────────────────────────
# 6. 전체 경기 μ̂ 예측 + a 초기값 생성
# ─────────────────────────────────────────────────────────

def predict_all_mu(
    model: xgb.Booster,
    X: pd.DataFrame,
    team_matches: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    전체 경기에 대해 μ̂ (기대 득점)을 예측하고,
    a_init = ln(μ̂ / T_m)을 계산한다.

    Step 1.4의 초기값으로 사용된다.
    """
    dmatrix = xgb.DMatrix(X[feature_cols], feature_names=feature_cols)
    mu_pred = model.predict(dmatrix)

    # μ̂ 하한 클램핑 (log 계산을 위해)
    mu_pred = np.maximum(mu_pred, 0.05)

    result = team_matches.loc[X.index, [
        "fixture_id", "team_id", "team_name", "is_home", "goals_scored"
    ]].copy()
    result["mu_hat"] = mu_pred

    logger.info(f"\n📊 μ̂ 예측 분포:")
    logger.info(f"  min={mu_pred.min():.3f}, mean={mu_pred.mean():.3f}, "
                f"max={mu_pred.max():.3f}, std={mu_pred.std():.3f}")

    return result


# ─────────────────────────────────────────────────────────
# 7. Walk-Forward CV
# ─────────────────────────────────────────────────────────

def walk_forward_cv(X: pd.DataFrame, y: pd.Series, feature_cols: List[str], team_matches: pd.DataFrame):
    """
    시간순 Walk-Forward 교차 검증.
    학습 기간을 점진적으로 늘리면서 다음 시즌을 예측한다.
    """
    logger.info("\n🔄 Walk-Forward Cross-Validation")

    seasons = sorted(team_matches.loc[X.index, "season"].unique())
    if len(seasons) < 3:
        logger.warning("  시즌이 3개 미만이라 CV 스킵")
        return

    results = []

    for val_season in seasons[2:]:  # 최소 2시즌 학습 후 검증
        train_seasons = [s for s in seasons if s < val_season]

        train_idx = team_matches.loc[X.index, "season"].isin(train_seasons)
        val_idx = team_matches.loc[X.index, "season"] == val_season

        X_tr = X[train_idx.values]
        y_tr = y[train_idx.values]
        X_va = X[val_idx.values]
        y_va = y[val_idx.values]

        if len(X_tr) < 100 or len(X_va) < 100:
            continue

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dval = xgb.DMatrix(X_va, label=y_va, feature_names=feature_cols)

        params = {
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "seed": 42,
            "verbosity": 0,
        }

        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        y_pred = model.predict(dval)
        mae = np.mean(np.abs(y_va - y_pred))
        rmse = np.sqrt(np.mean((y_va - y_pred) ** 2))

        # Poisson deviance
        mask_pos = y_va > 0
        deviance = 2 * np.sum(
            np.where(mask_pos, y_va * np.log(y_va / y_pred), 0) - (y_va - y_pred)
        ) / len(y_va)

        logger.info(f"  Train {train_seasons} → Val {val_season}: "
                    f"MAE={mae:.3f}, RMSE={rmse:.3f}, "
                    f"Deviance={deviance:.4f}, "
                    f"pred_mean={y_pred.mean():.3f}, actual_mean={y_va.mean():.3f}")

        results.append({
            "val_season": val_season,
            "mae": mae,
            "rmse": rmse,
            "deviance": deviance,
            "n_train": len(X_tr),
            "n_val": len(X_va),
        })

    if results:
        avg_mae = np.mean([r["mae"] for r in results])
        avg_deviance = np.mean([r["deviance"] for r in results])
        logger.info(f"\n  평균 MAE: {avg_mae:.3f}")
        logger.info(f"  평균 Deviance: {avg_deviance:.4f}")

    return results


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def run(validate_only: bool = False):
    """Step 1.3 전체 실행"""
    logger.info("📈 Step 1.3: XGBoost Poisson 모델 학습")
    logger.info("=" * 60)

    # 1. 데이터 로딩
    matches, stats = load_match_data(DB_PATH)

    # 2. 팀별 스탯 피벗
    team_stats = build_team_stats_table(matches, stats)

    # 3. 롤링 피처 계산
    team_matches = compute_rolling_features(matches, team_stats)

    # 4. 학습 데이터셋
    X, y, feature_cols = prepare_dataset(team_matches)

    # 5. Walk-Forward CV
    cv_results = walk_forward_cv(X, y, feature_cols, team_matches)

    if validate_only:
        return

    # 6. 최종 모델 학습 (전체 데이터)
    model, _ = train_model(X, y, feature_cols, team_matches)

    # 7. 피처 마스크
    selected_features = compute_feature_mask(model, feature_cols)

    # 8. 선택된 피처로 재학습
    logger.info(f"\n🔄 선택된 {len(selected_features)}개 피처로 재학습...")
    X_selected = X[selected_features]
    model_final, _ = train_model(X_selected, y, selected_features, team_matches)

    # 9. 전체 μ̂ 예측
    mu_predictions = predict_all_mu(model_final, X, team_matches, selected_features)

    # 10. 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    model_path = OUTPUT_DIR / "xgb_poisson.json"
    model_final.save_model(str(model_path))
    logger.info(f"  모델 저장: {model_path}")

    # 피처 마스크 저장
    mask_path = OUTPUT_DIR / "feature_mask.json"
    with open(mask_path, "w") as f:
        json.dump({
            "selected_features": selected_features,
            "all_features": feature_cols,
            "rolling_window": ROLLING_WINDOW,
        }, f, indent=2)
    logger.info(f"  피처 마스크 저장: {mask_path}")

    # μ̂ 예측값 저장 (Step 1.4 초기값용)
    mu_path = OUTPUT_DIR / "mu_predictions.csv"
    mu_predictions.to_csv(mu_path, index=False)
    logger.info(f"  μ̂ 예측값 저장: {mu_path}")

    # 요약
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Step 1.3 완료!")
    logger.info(f"  피처: {len(feature_cols)} → {len(selected_features)}개 선택")
    logger.info(f"  μ̂ 범위: [{mu_predictions['mu_hat'].min():.3f}, {mu_predictions['mu_hat'].max():.3f}]")
    logger.info(f"  μ̂ 평균: {mu_predictions['mu_hat'].mean():.3f} (실제: {mu_predictions['goals_scored'].mean():.3f})")
    logger.info(f"{'='*60}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 1.3: XGBoost Poisson Prior")
    parser.add_argument("--validate", action="store_true", help="CV만 실행")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()
    run(validate_only=args.validate)


if __name__ == "__main__":
    main()
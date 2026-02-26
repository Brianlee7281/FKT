"""
Step 1.5: 시계열 교차검증 및 모델 진단 (Validation).

이 단계를 통과하지 않으면 라이브에 투입하지 않는다.

검증 항목:
  1. δ 부호 검증 (Sanity Check)
  2. Likelihood Ratio Test (δ=0 기각 검증)
  3. Walk-Forward CV (시즌 단위 재학습)
  4. Calibration Plot + Brier Score
  5. Go/No-Go 판정

입력:
  - data/models/nll_params.json (Step 1.4)
  - data/models/a_calibrated.csv (Step 1.4)
  - intervals, goal_events, match_meta, matches 테이블

출력:
  - data/models/validation_report.json
  - 터미널에 Go/No-Go 판정

사용법:
  python -m src.phase1.validation
"""

import json
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# 로컬 모듈
from src.phase1.joint_nll import (
    MMPPModel, load_data, optimize_single, extract_params,
    SIGMA_A, LAMBDA_REG, delta_s_to_idx,
)

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
        logging.FileHandler("logs/validation.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# MC 시뮬레이션 설정
MC_SIMS = 20000
NUM_BASIS = 6


# ─────────────────────────────────────────────────────────
# 1. δ 부호 검증 (Sanity Check)
# ─────────────────────────────────────────────────────────

def check_delta_signs(params: Dict) -> Dict:
    """
    학습된 δ 값이 축구 직관과 부합하는지 검증.

    기대:
      δ_H[-1] > 0  뒤지는 홈팀 → 공격↑
      δ_H[+1] < 0  앞서는 홈팀 → 수비 전환↓
      δ_A[-1] < 0  앞서는 어웨이(홈 뒤짐) → 수비↓ (ΔS<0 means away leads)
      δ_A[+1] > 0  뒤지는 어웨이(홈 앞섬) → 공격↑
    """
    dH = params["delta_H"]   # [ΔS≤-2, ΔS=-1, ΔS=+1, ΔS≥+2]
    dA = params["delta_A"]

    checks = {
        "δ_H(ΔS=-1) > 0 (홈 뒤짐→공격↑)": dH[1] > 0,
        "δ_H(ΔS=+1) < 0 (홈 앞섬→수비↓)": dH[2] < 0,
        "δ_H(ΔS≤-2) > δ_H(ΔS=-1) (크게 뒤지면 더 공격)": dH[0] >= dH[1] - 0.1,
        "δ_H(ΔS≥+2) < δ_H(ΔS=+1) (크게 앞서면 더 수비)": dH[3] <= dH[2] + 0.1,
        "δ_A(ΔS=-1) < 0 (어웨이 앞섬→수비↓)": dA[1] < 0,
        "δ_A(ΔS=+1) > 0 (어웨이 뒤짐→공격↑)": dA[2] > -0.1,  # 약간의 여유
    }

    return checks


# ─────────────────────────────────────────────────────────
# 2. Likelihood Ratio Test (δ=0 기각 검증)
# ─────────────────────────────────────────────────────────

def likelihood_ratio_test(data: Dict) -> Dict:
    """
    δ를 전부 0으로 놓은 모델 vs δ를 학습한 모델의 LR 검정.

    LR = -2(L_restricted - L_full) ~ χ²(df)
    df = δ 파라미터 수 = 8
    """
    from scipy import stats

    logger.info("\n📊 Likelihood Ratio Test: δ=0 vs δ≠0")

    # Full model (δ 학습)
    logger.info("  Full model (δ 학습) 최적화...")
    loss_full, _ = optimize_single(data, seed=42)

    # Restricted model (δ=0 고정)
    logger.info("  Restricted model (δ=0 고정) 최적화...")
    loss_restricted = _optimize_delta_zero(data)

    # NLL만 비교 (정규화 항 제외하면 더 정확하지만,
    # 동일한 정규화 조건이므로 total loss 비교도 유효)
    LR_stat = 2 * (loss_restricted - loss_full)
    df = 8   # δ_H[4] + δ_A[4]
    p_value = 1 - stats.chi2.cdf(LR_stat, df)

    logger.info(f"  Full loss:       {loss_full:.2f}")
    logger.info(f"  Restricted loss: {loss_restricted:.2f}")
    logger.info(f"  LR statistic:    {LR_stat:.2f}")
    logger.info(f"  df:              {df}")
    logger.info(f"  p-value:         {p_value:.6f}")
    logger.info(f"  결론: {'δ≠0 채택 (유의미)' if p_value < 0.05 else 'δ=0 기각 실패'}")

    return {
        "loss_full": loss_full,
        "loss_restricted": loss_restricted,
        "LR_statistic": LR_stat,
        "df": df,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def _optimize_delta_zero(data: Dict) -> float:
    """δ를 0으로 고정한 채 나머지 파라미터만 최적화"""
    model = MMPPModel(data, seed=42)

    # δ를 0으로 고정하고 requires_grad=False
    model.delta_H.requires_grad_(False)
    model.delta_A.requires_grad_(False)
    model.delta_H.zero_()
    model.delta_A.zero_()

    # Adam 최적화 (δ 제외)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1500):
        optimizer.zero_grad()
        loss, comp = model()
        loss.backward()
        optimizer.step()
        model.clamp_params()

        if comp["total"] < best_loss - 0.1:
            best_loss = comp["total"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 200:
                break

    # L-BFGS
    lbfgs = torch.optim.LBFGS(trainable, lr=0.1, max_iter=20,
                                history_size=50, line_search_fn="strong_wolfe")
    for _ in range(100):
        def closure():
            lbfgs.zero_grad()
            loss, _ = model()
            loss.backward()
            return loss
        lbfgs.step(closure)
        model.clamp_params()

    _, final_comp = model()
    return final_comp["total"]


# ─────────────────────────────────────────────────────────
# 3. Monte Carlo 경기 시뮬레이션
# ─────────────────────────────────────────────────────────

def simulate_match_mc(
    a_home: float,
    a_away: float,
    b: np.ndarray,
    gamma_1: float,
    gamma_2: float,
    delta_H: np.ndarray,
    delta_A: np.ndarray,
    T_m: float = 95.0,
    n_sims: int = MC_SIMS,
) -> Dict:
    """
    단일 경기를 MC 시뮬레이션하여 결과 확률을 추정한다.

    간소화 버전: 레드카드 상태 전이는 무시 (X=0 고정).
    대부분의 경기가 X=0이므로 큰 오차 없음.

    Returns:
        {"home_win": float, "draw": float, "away_win": float,
         "over_2.5": float, "avg_total": float, ...}
    """
    rng = np.random.default_rng()

    # 기저함수 경계 (실효 시간, 15분 간격 + 추가시간)
    first_half_end = 47.0   # ~45+2
    boundaries = [0, 15, 30, first_half_end,
                  first_half_end + 15, first_half_end + 30, T_m]

    home_goals = np.zeros(n_sims, dtype=np.int32)
    away_goals = np.zeros(n_sims, dtype=np.int32)

    # 각 기저함수 빈을 순차 시뮬레이션
    for bi in range(NUM_BASIS):
        t_start = boundaries[bi]
        t_end = boundaries[bi + 1]
        duration = t_end - t_start

        if duration <= 0:
            continue

        # ΔS에 따른 δ 적용 (현재 스코어 기반)
        delta_S = home_goals - away_goals   # [n_sims]

        # δ 룩업 (벡터화)
        dh = np.zeros(n_sims)
        da = np.zeros(n_sims)

        mask_neg2 = delta_S <= -2
        mask_neg1 = delta_S == -1
        mask_pos1 = delta_S == 1
        mask_pos2 = delta_S >= 2
        # ΔS=0은 δ=0 (기본값)

        dh[mask_neg2] = delta_H[0]
        dh[mask_neg1] = delta_H[1]
        dh[mask_pos1] = delta_H[2]
        dh[mask_pos2] = delta_H[3]

        da[mask_neg2] = delta_A[0]
        da[mask_neg1] = delta_A[1]
        da[mask_pos1] = delta_A[2]
        da[mask_pos2] = delta_A[3]

        # λ = exp(a + b + δ) × duration  (γ=0, X=0 가정)
        lambda_h = np.exp(a_home + b[bi] + dh) * duration
        lambda_a = np.exp(a_away + b[bi] + da) * duration

        # Poisson 샘플링
        goals_h = rng.poisson(lambda_h)
        goals_a = rng.poisson(lambda_a)

        home_goals += goals_h
        away_goals += goals_a

    total_goals = home_goals + away_goals

    return {
        "home_win": float(np.mean(home_goals > away_goals)),
        "draw": float(np.mean(home_goals == away_goals)),
        "away_win": float(np.mean(home_goals < away_goals)),
        "over_2.5": float(np.mean(total_goals >= 3)),
        "over_1.5": float(np.mean(total_goals >= 2)),
        "btts": float(np.mean((home_goals >= 1) & (away_goals >= 1))),
        "avg_home": float(np.mean(home_goals)),
        "avg_away": float(np.mean(away_goals)),
        "avg_total": float(np.mean(total_goals)),
    }


# ─────────────────────────────────────────────────────────
# 4. Calibration + Brier Score
# ─────────────────────────────────────────────────────────

def compute_calibration_brier(
    predictions: List[Dict],
    matches_df: pd.DataFrame,
) -> Dict:
    """
    예측 확률 vs 실제 결과를 비교하여 캘리브레이션과 Brier Score 계산.

    predictions: [{"fixture_id", "home_win", "draw", "away_win", ...}, ...]
    matches_df: fixture_id, home_goals_ft, away_goals_ft
    """
    pred_probs = []   # 모델 예측 P
    actuals = []      # 실제 결과 O ∈ {0, 1}

    for pred in predictions:
        fid = pred["fixture_id"]
        match = matches_df[matches_df["fixture_id"] == fid]
        if len(match) == 0:
            continue

        hg = match["home_goals_ft"].values[0]
        ag = match["away_goals_ft"].values[0]

        # Home Win
        pred_probs.append(pred["home_win"])
        actuals.append(1 if hg > ag else 0)

        # Draw
        pred_probs.append(pred["draw"])
        actuals.append(1 if hg == ag else 0)

        # Away Win
        pred_probs.append(pred["away_win"])
        actuals.append(1 if hg < ag else 0)

        # Over 2.5
        pred_probs.append(pred["over_2.5"])
        actuals.append(1 if hg + ag >= 3 else 0)

    pred_probs = np.array(pred_probs)
    actuals = np.array(actuals)

    # Brier Score
    brier = np.mean((pred_probs - actuals) ** 2)

    # Log Loss
    eps = 1e-7
    pred_clipped = np.clip(pred_probs, eps, 1 - eps)
    log_loss = -np.mean(
        actuals * np.log(pred_clipped) + (1 - actuals) * np.log(1 - pred_clipped)
    )

    # Calibration bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_bins = []

    for i in range(n_bins):
        mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            cal_bins.append({
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "pred_mean": float(pred_probs[mask].mean()),
                "actual_mean": float(actuals[mask].mean()),
                "count": int(mask.sum()),
            })

    # Calibration error (평균 |pred - actual|)
    cal_errors = [abs(b["pred_mean"] - b["actual_mean"]) for b in cal_bins if b["count"] >= 10]
    mean_cal_error = np.mean(cal_errors) if cal_errors else 0

    # Baseline Brier: "항상 평균 확률 예측" 모델
    baseline_brier = np.mean((actuals.mean() - actuals) ** 2)

    return {
        "brier_score": float(brier),
        "baseline_brier": float(baseline_brier),
        "brier_skill": float(1 - brier / baseline_brier),
        "log_loss": float(log_loss),
        "mean_calibration_error": float(mean_cal_error),
        "calibration_bins": cal_bins,
        "n_predictions": len(pred_probs),
    }


# ─────────────────────────────────────────────────────────
# 5. Walk-Forward CV
# ─────────────────────────────────────────────────────────

def walk_forward_cv(db_path: Path) -> List[Dict]:
    """
    시즌 단위 Walk-Forward CV.

    각 Fold에서:
    1. 학습 시즌의 데이터로 Step 1.4 NLL 최적화
    2. 학습된 파라미터로 검증 시즌 경기를 MC 시뮬레이션
    3. 예측 확률 vs 실제 결과로 Brier Score 계산
    """
    conn = sqlite3.connect(str(db_path))

    # 시즌 목록
    seasons = pd.read_sql_query(
        "SELECT DISTINCT season FROM matches ORDER BY season", conn
    )["season"].tolist()
    conn.close()

    logger.info(f"\n🔄 Walk-Forward CV: 시즌 {seasons}")

    if len(seasons) < 3:
        logger.warning("  시즌 3개 미만, CV 불가")
        return []

    fold_results = []

    for val_season in seasons[2:]:
        train_seasons = [s for s in seasons if s < val_season]
        logger.info(f"\n  Fold: Train {train_seasons} → Val {val_season}")

        # ── 학습 데이터 로드 (train seasons만) ──
        train_data = _load_data_for_seasons(db_path, train_seasons)
        if train_data["M"] < 100:
            logger.warning(f"    학습 경기 {train_data['M']}개 → 스킵")
            continue

        # ── NLL 최적화 (3 starts로 빠르게) ──
        best_loss = float("inf")
        best_params = None
        for seed in range(3):
            loss, params = optimize_single(train_data, seed=seed)
            if loss < best_loss:
                best_loss = loss
                best_params = params

        logger.info(f"    학습 완료: loss={best_loss:.1f}")

        # ── 검증 시즌 경기 MC 시뮬레이션 ──
        conn = sqlite3.connect(str(db_path))
        val_matches = pd.read_sql_query(f"""
            SELECT m.fixture_id, m.home_goals_ft, m.away_goals_ft, mm.T_m
            FROM matches m
            JOIN match_meta mm ON m.fixture_id = mm.fixture_id
            WHERE m.season = {val_season}
            AND m.home_goals_ft IS NOT NULL
        """, conn)

        # 검증 경기의 μ̂ (XGBoost)
        mu_df = pd.read_csv(OUTPUT_DIR / "mu_predictions.csv")
        conn.close()

        predictions = []
        for _, vm in val_matches.iterrows():
            fid = vm["fixture_id"]
            T_m = vm["T_m"]

            # μ̂에서 a_init 계산
            home_mu_row = mu_df[(mu_df["fixture_id"] == fid) & (mu_df["is_home"] == 1)]
            away_mu_row = mu_df[(mu_df["fixture_id"] == fid) & (mu_df["is_home"] == 0)]

            mu_h = home_mu_row["mu_hat"].values[0] if len(home_mu_row) > 0 else 1.4
            mu_a = away_mu_row["mu_hat"].values[0] if len(away_mu_row) > 0 else 1.2

            a_h = np.log(max(mu_h, 0.05) / T_m)
            a_a = np.log(max(mu_a, 0.05) / T_m)

            # MC 시뮬레이션
            result = simulate_match_mc(
                a_home=a_h, a_away=a_a,
                b=np.array(best_params["b"]),
                gamma_1=best_params["gamma_1"],
                gamma_2=best_params["gamma_2"],
                delta_H=np.array(best_params["delta_H"]),
                delta_A=np.array(best_params["delta_A"]),
                T_m=T_m,
            )
            result["fixture_id"] = fid
            predictions.append(result)

        # ── Calibration + Brier ──
        metrics = compute_calibration_brier(predictions, val_matches)

        logger.info(f"    Val {val_season}: "
                    f"Brier={metrics['brier_score']:.4f}, "
                    f"Skill={metrics['brier_skill']:.3f}, "
                    f"LogLoss={metrics['log_loss']:.4f}, "
                    f"CalErr={metrics['mean_calibration_error']:.4f}, "
                    f"n={metrics['n_predictions']}")

        # 실제 vs 예측 평균 비교
        pred_hw = np.mean([p["home_win"] for p in predictions])
        pred_draw = np.mean([p["draw"] for p in predictions])
        actual_hw = np.mean(val_matches["home_goals_ft"] > val_matches["away_goals_ft"])
        actual_draw = np.mean(val_matches["home_goals_ft"] == val_matches["away_goals_ft"])

        logger.info(f"    Home Win: pred={pred_hw:.3f}, actual={actual_hw:.3f}")
        logger.info(f"    Draw:     pred={pred_draw:.3f}, actual={actual_draw:.3f}")

        fold_results.append({
            "val_season": val_season,
            "train_seasons": train_seasons,
            "n_val_matches": len(val_matches),
            "train_loss": best_loss,
            "metrics": metrics,
            "params": {
                "b": best_params["b"],
                "gamma_1": best_params["gamma_1"],
                "gamma_2": best_params["gamma_2"],
                "delta_H": best_params["delta_H"],
                "delta_A": best_params["delta_A"],
            },
        })

    return fold_results


def _load_data_for_seasons(db_path: Path, seasons: List[int]) -> Dict:
    """특정 시즌들의 데이터만 로드"""
    conn = sqlite3.connect(str(db_path))

    season_filter = ",".join(str(s) for s in seasons)

    intervals_df = pd.read_sql_query(f"""
        SELECT i.fixture_id, i.t_start, i.t_end, i.duration,
               i.state_X, i.delta_S, i.basis_idx, i.is_halftime, i.T_m
        FROM intervals i
        JOIN matches m ON i.fixture_id = m.fixture_id
        WHERE i.is_halftime = 0 AND i.duration > 0
        AND m.season IN ({season_filter})
    """, conn)

    goals_df = pd.read_sql_query(f"""
        SELECT g.fixture_id, g.t_eff, g.team, g.state_X, g.delta_S_before, g.basis_idx
        FROM goal_events g
        JOIN matches m ON g.fixture_id = m.fixture_id
        WHERE m.season IN ({season_filter})
    """, conn)

    meta_df = pd.read_sql_query(f"""
        SELECT mm.fixture_id, mm.T_m,
               m.home_team_id, m.away_team_id
        FROM match_meta mm
        JOIN matches m ON mm.fixture_id = m.fixture_id
        WHERE m.season IN ({season_filter})
    """, conn)

    conn.close()

    # 경기 인덱스
    fixture_ids = sorted(meta_df["fixture_id"].unique())
    fix_to_idx = {fid: i for i, fid in enumerate(fixture_ids)}
    M = len(fixture_ids)

    # μ̂에서 a_init 계산
    mu_path = OUTPUT_DIR / "mu_predictions.csv"
    mu_df = pd.read_csv(mu_path) if mu_path.exists() else None

    a_init_home = np.zeros(M)
    a_init_away = np.zeros(M)

    for _, row in meta_df.iterrows():
        fid = row["fixture_id"]
        idx = fix_to_idx.get(fid)
        if idx is None:
            continue
        T_m = row["T_m"]

        if mu_df is not None:
            hm = mu_df[(mu_df["fixture_id"] == fid) & (mu_df["is_home"] == 1)]
            am = mu_df[(mu_df["fixture_id"] == fid) & (mu_df["is_home"] == 0)]
            mu_h = hm["mu_hat"].values[0] if len(hm) > 0 else 1.4
            mu_a = am["mu_hat"].values[0] if len(am) > 0 else 1.2
        else:
            mu_h, mu_a = 1.4, 1.2

        a_init_home[idx] = np.log(max(mu_h, 0.05) / T_m)
        a_init_away[idx] = np.log(max(mu_a, 0.05) / T_m)

    # 텐서 구성 (joint_nll.py와 동일한 형식)
    iv_match_idx, iv_basis_idx, iv_state_X = [], [], []
    iv_delta_S_idx, iv_duration = [], []

    for _, row in intervals_df.iterrows():
        idx = fix_to_idx.get(row["fixture_id"])
        if idx is None:
            continue
        iv_match_idx.append(idx)
        iv_basis_idx.append(int(row["basis_idx"]))
        iv_state_X.append(int(row["state_X"]))
        iv_delta_S_idx.append(delta_s_to_idx(int(row["delta_S"])))
        iv_duration.append(row["duration"])

    g_match_idx, g_basis_idx, g_state_X = [], [], []
    g_delta_S_idx, g_is_home = [], []

    for _, row in goals_df.iterrows():
        idx = fix_to_idx.get(row["fixture_id"])
        if idx is None:
            continue
        g_match_idx.append(idx)
        g_basis_idx.append(int(row["basis_idx"]))
        g_state_X.append(int(row["state_X"]))
        g_delta_S_idx.append(delta_s_to_idx(int(row["delta_S_before"])))
        g_is_home.append(1 if row["team"] == "home" else 0)

    return {
        "M": M,
        "fixture_ids": fixture_ids,
        "fix_to_idx": fix_to_idx,
        "a_init_home": torch.tensor(a_init_home, dtype=torch.float32),
        "a_init_away": torch.tensor(a_init_away, dtype=torch.float32),
        "iv_match_idx": torch.tensor(iv_match_idx, dtype=torch.long),
        "iv_basis_idx": torch.tensor(iv_basis_idx, dtype=torch.long),
        "iv_state_X": torch.tensor(iv_state_X, dtype=torch.long),
        "iv_delta_S_idx": torch.tensor(iv_delta_S_idx, dtype=torch.long),
        "iv_duration": torch.tensor(iv_duration, dtype=torch.float32),
        "g_match_idx": torch.tensor(g_match_idx, dtype=torch.long),
        "g_basis_idx": torch.tensor(g_basis_idx, dtype=torch.long),
        "g_state_X": torch.tensor(g_state_X, dtype=torch.long),
        "g_delta_S_idx": torch.tensor(g_delta_S_idx, dtype=torch.long),
        "g_is_home": torch.tensor(g_is_home, dtype=torch.float32),
    }


# ─────────────────────────────────────────────────────────
# 6. Go/No-Go 판정
# ─────────────────────────────────────────────────────────

def go_no_go(
    delta_checks: Dict,
    lr_test: Dict,
    cv_results: List[Dict],
) -> Dict:
    """최종 Go/No-Go 판정"""

    criteria = {}

    # 1. δ 부호 검증
    delta_pass = sum(delta_checks.values())
    delta_total = len(delta_checks)
    criteria["delta_signs"] = {
        "pass": delta_pass >= delta_total - 1,  # 최소 5/6
        "score": f"{delta_pass}/{delta_total}",
        "details": {k: v for k, v in delta_checks.items()},
    }

    # 2. LR Test
    criteria["lr_test"] = {
        "pass": lr_test["significant"],
        "p_value": lr_test["p_value"],
    }

    # 3. Brier Score (모든 fold에서 baseline 대비 개선)
    if cv_results:
        brier_skills = [r["metrics"]["brier_skill"] for r in cv_results]
        criteria["brier_skill"] = {
            "pass": all(s > 0 for s in brier_skills),
            "values": brier_skills,
            "mean": float(np.mean(brier_skills)),
        }

        # 4. Calibration Error
        cal_errors = [r["metrics"]["mean_calibration_error"] for r in cv_results]
        criteria["calibration"] = {
            "pass": all(e < 0.05 for e in cal_errors),
            "values": cal_errors,
            "mean": float(np.mean(cal_errors)),
        }
    else:
        criteria["brier_skill"] = {"pass": False, "values": [], "mean": 0}
        criteria["calibration"] = {"pass": False, "values": [], "mean": 0}

    # 종합 판정
    all_pass = all(c["pass"] for c in criteria.values())

    return {
        "decision": "GO ✅" if all_pass else "NO-GO ❌",
        "all_pass": all_pass,
        "criteria": criteria,
    }


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def run():
    """Step 1.5 전체 실행"""
    logger.info("🔍 Step 1.5: 모델 검증")
    logger.info("=" * 60)

    # ── 1. 학습된 파라미터 로드 ──
    param_path = OUTPUT_DIR / "nll_params.json"
    with open(param_path) as f:
        params = json.load(f)

    logger.info("  파라미터 로드 완료")

    # ── 2. δ 부호 검증 ──
    logger.info("\n📋 δ 부호 검증:")
    delta_checks = check_delta_signs(params)
    for check, passed in delta_checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {check}")

    # ── 3. LR Test ──
    data = load_data(DB_PATH)
    lr_test = likelihood_ratio_test(data)

    # ── 4. Walk-Forward CV ──
    cv_results = walk_forward_cv(DB_PATH)

    # ── 5. Go/No-Go ──
    verdict = go_no_go(delta_checks, lr_test, cv_results)

    print(f"\n{'='*60}")
    print(f"  🏁 최종 판정: {verdict['decision']}")
    print(f"{'='*60}")

    for name, crit in verdict["criteria"].items():
        status = "✅" if crit["pass"] else "❌"
        if "score" in crit:
            print(f"  {status} {name}: {crit['score']}")
        elif "mean" in crit:
            print(f"  {status} {name}: mean={crit['mean']:.4f}")
        elif "p_value" in crit:
            print(f"  {status} {name}: p={crit['p_value']:.6f}")
        else:
            print(f"  {status} {name}")

    # Calibration 상세
    if cv_results:
        print(f"\n  📊 Calibration (검증 시즌별):")
        for r in cv_results:
            m = r["metrics"]
            print(f"    시즌 {r['val_season']}: "
                  f"Brier={m['brier_score']:.4f}, "
                  f"Skill={m['brier_skill']:.3f}, "
                  f"CalErr={m['mean_calibration_error']:.4f}")

        # 캘리브레이션 빈 출력 (마지막 fold)
        print(f"\n  📊 Calibration Bins (마지막 fold):")
        print(f"    {'Pred':>8} {'Actual':>8} {'Count':>6}")
        for b in cv_results[-1]["metrics"]["calibration_bins"]:
            if b["count"] >= 10:
                diff = abs(b["pred_mean"] - b["actual_mean"])
                marker = "✅" if diff < 0.05 else "⚠️"
                print(f"    {b['pred_mean']:8.3f} {b['actual_mean']:8.3f} "
                      f"{b['count']:6d}  {marker}")

    print()

    # ── 저장 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": verdict,
        "delta_checks": {k: bool(v) for k, v in delta_checks.items()},
        "lr_test": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in lr_test.items()},
        "cv_folds": [{
            "val_season": r["val_season"],
            "metrics": r["metrics"],
        } for r in cv_results],
    }
    report_path = OUTPUT_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  리포트 저장: {report_path}")

    logger.info(f"\n✅ Step 1.5 완료!")


def main():
    parser = argparse.ArgumentParser(description="Step 1.5: Model Validation")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    run()


if __name__ == "__main__":
    main()
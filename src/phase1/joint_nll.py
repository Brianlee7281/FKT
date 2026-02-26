"""
Step 1.4: Joint NLL 최적화 (MMPP Calibration).

시간 프로파일(b), 레드카드 패널티(γ), 스코어차 효과(δ),
경기별 기본 강도(a)를 동시에 최적화한다.

NLL = -Σ_m [ Σ_g (ln λ(t_g))  -  Σ_k μ_k ]
    + (1/2σ²) Σ_m (a_m - a_m_init)²
    + λ_reg (||b||² + ||γ||² + ||δ||²)

입력:
  - intervals, goal_events, match_meta 테이블 (preprocessor.py)
  - mu_predictions.csv (Step 1.3 XGBoost)

출력:
  - data/models/nll_params.json (b, γ, δ)
  - data/models/a_calibrated.csv (보정된 경기별 a)

사용법:
  python -m src.phase1.joint_nll
  python -m src.phase1.joint_nll --starts 10    # multi-start 횟수
"""

import json
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
        logging.FileHandler("logs/joint_nll.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# 하이퍼파라미터
SIGMA_A = 0.5           # a 정규화 강도 (작을수록 ML prior에 가까움)
LAMBDA_REG = 0.01       # L2 정규화 강도
ADAM_LR = 1e-3
ADAM_EPOCHS = 1500
LBFGS_EPOCHS = 200
NUM_STARTS = 5          # Multi-start 횟수

# ΔS → δ 인덱스 매핑
# ΔS ≤ -2 → 0,  ΔS = -1 → 1,  ΔS = 0 → (고정 0),  ΔS = +1 → 2,  ΔS ≥ +2 → 3
def delta_s_to_idx(ds: int) -> int:
    """ΔS를 δ 룩업 인덱스로 변환. ΔS=0이면 -1 반환 (고정 0)."""
    if ds <= -2:
        return 0
    elif ds == -1:
        return 1
    elif ds == 0:
        return -1   # 특수값: δ(0) = 0 고정
    elif ds == 1:
        return 2
    else:  # ds >= 2
        return 3


# ─────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────

def load_data(db_path: Path) -> Dict:
    """
    DB에서 intervals, goal_events, match_meta를 로드하고,
    mu_predictions.csv에서 a_init을 계산한다.

    각 경기 m은 홈/어웨이 두 팀의 NLL에 기여한다.
    """
    conn = sqlite3.connect(str(db_path))

    # ── intervals ──
    intervals_df = pd.read_sql_query("""
        SELECT fixture_id, t_start, t_end, duration,
               state_X, delta_S, basis_idx, is_halftime, T_m
        FROM intervals
        WHERE is_halftime = 0 AND duration > 0
        ORDER BY fixture_id, t_start
    """, conn)

    # ── goal events ──
    goals_df = pd.read_sql_query("""
        SELECT fixture_id, t_eff, team, state_X, delta_S_before, basis_idx
        FROM goal_events
        ORDER BY fixture_id, t_eff
    """, conn)

    # ── match meta ──
    meta_df = pd.read_sql_query("""
        SELECT mm.fixture_id, mm.T_m,
               m.home_team_id, m.away_team_id
        FROM match_meta mm
        JOIN matches m ON mm.fixture_id = m.fixture_id
    """, conn)

    conn.close()

    # ── μ̂ from XGBoost (Step 1.3) ──
    mu_path = OUTPUT_DIR / "mu_predictions.csv"
    if mu_path.exists():
        mu_df = pd.read_csv(mu_path)
        logger.info(f"  μ̂ 로드: {len(mu_df)}행")
    else:
        mu_df = None
        logger.warning("  ⚠️ mu_predictions.csv 없음. a_init을 실제 득점에서 계산합니다.")

    # ── 경기 인덱스 매핑 ──
    fixture_ids = sorted(meta_df["fixture_id"].unique())
    fix_to_idx = {fid: i for i, fid in enumerate(fixture_ids)}
    M = len(fixture_ids)

    logger.info(f"  경기: {M}, 구간: {len(intervals_df)}, 골: {len(goals_df)}")

    # ── a_init 계산 ──
    # 경기별-팀별 a_init = ln(μ̂ / T_m)
    a_init_home = np.zeros(M)
    a_init_away = np.zeros(M)

    for _, row in meta_df.iterrows():
        fid = row["fixture_id"]
        idx = fix_to_idx.get(fid)
        if idx is None:
            continue

        T_m = row["T_m"]

        if mu_df is not None:
            # XGBoost μ̂ 사용
            home_mu = mu_df[
                (mu_df["fixture_id"] == fid) & (mu_df["is_home"] == 1)
            ]["mu_hat"]
            away_mu = mu_df[
                (mu_df["fixture_id"] == fid) & (mu_df["is_home"] == 0)
            ]["mu_hat"]

            mu_h = home_mu.values[0] if len(home_mu) > 0 else 1.4
            mu_a = away_mu.values[0] if len(away_mu) > 0 else 1.2
        else:
            # fallback: 리그 평균
            mu_h = 1.4
            mu_a = 1.2

        # a = ln(μ̂ / T_m), 클램핑으로 수치 안정
        mu_h = max(mu_h, 0.05)
        mu_a = max(mu_a, 0.05)
        a_init_home[idx] = np.log(mu_h / T_m)
        a_init_away[idx] = np.log(mu_a / T_m)

    # ── 텐서 구성: 구간 데이터 ──
    # 각 구간은 홈팀과 어웨이팀 양쪽의 NLL에 기여
    iv_match_idx = []
    iv_basis_idx = []
    iv_state_X = []
    iv_delta_S_idx = []  # δ 룩업 인덱스 (-1 = 고정 0)
    iv_duration = []

    for _, row in intervals_df.iterrows():
        fid = row["fixture_id"]
        idx = fix_to_idx.get(fid)
        if idx is None:
            continue

        iv_match_idx.append(idx)
        iv_basis_idx.append(int(row["basis_idx"]))
        iv_state_X.append(int(row["state_X"]))
        iv_delta_S_idx.append(delta_s_to_idx(int(row["delta_S"])))
        iv_duration.append(row["duration"])

    # ── 텐서 구성: 골 이벤트 ──
    g_match_idx = []
    g_basis_idx = []
    g_state_X = []
    g_delta_S_idx = []
    g_is_home = []         # 1 = 홈팀 득점, 0 = 어웨이팀 득점

    for _, row in goals_df.iterrows():
        fid = row["fixture_id"]
        idx = fix_to_idx.get(fid)
        if idx is None:
            continue

        g_match_idx.append(idx)
        g_basis_idx.append(int(row["basis_idx"]))
        g_state_X.append(int(row["state_X"]))
        g_delta_S_idx.append(delta_s_to_idx(int(row["delta_S_before"])))
        g_is_home.append(1 if row["team"] == "home" else 0)

    data = {
        "M": M,
        "fixture_ids": fixture_ids,
        "fix_to_idx": fix_to_idx,
        "a_init_home": torch.tensor(a_init_home, dtype=torch.float32),
        "a_init_away": torch.tensor(a_init_away, dtype=torch.float32),
        # 구간
        "iv_match_idx": torch.tensor(iv_match_idx, dtype=torch.long),
        "iv_basis_idx": torch.tensor(iv_basis_idx, dtype=torch.long),
        "iv_state_X": torch.tensor(iv_state_X, dtype=torch.long),
        "iv_delta_S_idx": torch.tensor(iv_delta_S_idx, dtype=torch.long),
        "iv_duration": torch.tensor(iv_duration, dtype=torch.float32),
        # 골
        "g_match_idx": torch.tensor(g_match_idx, dtype=torch.long),
        "g_basis_idx": torch.tensor(g_basis_idx, dtype=torch.long),
        "g_state_X": torch.tensor(g_state_X, dtype=torch.long),
        "g_delta_S_idx": torch.tensor(g_delta_S_idx, dtype=torch.long),
        "g_is_home": torch.tensor(g_is_home, dtype=torch.float32),
    }

    logger.info(f"  텐서: 구간 {len(iv_match_idx)}, 골 {len(g_match_idx)}")

    return data


# ─────────────────────────────────────────────────────────
# PyTorch 모델
# ─────────────────────────────────────────────────────────

class MMPPModel(nn.Module):
    """
    MMPP 파라미터 Joint NLL 모델.

    학습 가능 파라미터:
    - a_home [M]: 경기별 홈팀 기본 강도
    - a_away [M]: 경기별 어웨이팀 기본 강도
    - b [6]: 시간 구간별 프로파일
    - gamma_raw [2]: [γ_1 (홈 퇴장 패널티), γ_2 (어웨이 퇴장 패널티)]
    - delta_H [4]: 홈팀 스코어차 효과 [ΔS≤-2, ΔS=-1, ΔS=+1, ΔS≥+2]
    - delta_A [4]: 어웨이팀 스코어차 효과

    λ_H(t) = exp(a_H + b[i(t)] + γ_H[X(t)] + δ_H[ΔS(t)])
    λ_A(t) = exp(a_A + b[i(t)] + γ_A[X(t)] + δ_A[ΔS(t)])

    γ_H[X]: [0, γ_1, γ_2, γ_1+γ_2]  (홈팀 시점)
    γ_A[X]: [0, γ_2, γ_1, γ_1+γ_2]  (어웨이팀 시점: 스왑!)
    """

    def __init__(self, data: Dict, seed: int = 0):
        super().__init__()
        M = data["M"]

        torch.manual_seed(seed)
        noise_scale = 0.05

        # 경기별 기본 강도 (초기값 = XGBoost a_init)
        self.a_home = nn.Parameter(data["a_init_home"].clone())
        self.a_away = nn.Parameter(data["a_init_away"].clone())

        # 시간 프로파일 b [6]
        self.b = nn.Parameter(torch.randn(6) * noise_scale)

        # 레드카드 패널티 γ [2]: [γ_1, γ_2]
        # γ_1: 홈 퇴장 → 홈 득점력↓ (음수)
        # γ_2: 어웨이 퇴장 → 홈 득점력↑ (양수)
        self.gamma_raw = nn.Parameter(torch.randn(2) * noise_scale)

        # 스코어차 효과 δ_H [4], δ_A [4]
        # 인덱스: [ΔS≤-2, ΔS=-1, ΔS=+1, ΔS≥+2], ΔS=0은 고정 0
        self.delta_H = nn.Parameter(torch.randn(4) * noise_scale)
        self.delta_A = nn.Parameter(torch.randn(4) * noise_scale)

        # 데이터 (학습 불가)
        self.register_buffer("a_init_home", data["a_init_home"].clone())
        self.register_buffer("a_init_away", data["a_init_away"].clone())
        self.register_buffer("iv_match_idx", data["iv_match_idx"])
        self.register_buffer("iv_basis_idx", data["iv_basis_idx"])
        self.register_buffer("iv_state_X", data["iv_state_X"])
        self.register_buffer("iv_delta_S_idx", data["iv_delta_S_idx"])
        self.register_buffer("iv_duration", data["iv_duration"])
        self.register_buffer("g_match_idx", data["g_match_idx"])
        self.register_buffer("g_basis_idx", data["g_basis_idx"])
        self.register_buffer("g_state_X", data["g_state_X"])
        self.register_buffer("g_delta_S_idx", data["g_delta_S_idx"])
        self.register_buffer("g_is_home", data["g_is_home"])

    def _build_gamma(self):
        """
        γ 룩업 테이블 구성.

        γ_H[X]: 홈팀 시점
          X=0: 0  (11v11)
          X=1: γ_1  (홈 퇴장 → 홈↓)
          X=2: γ_2  (어웨이 퇴장 → 홈↑)
          X=3: γ_1 + γ_2

        γ_A[X]: 어웨이팀 시점 (스왑!)
          X=0: 0
          X=1: γ_2  (홈 퇴장 → 어웨이↑)
          X=2: γ_1  (어웨이 퇴장 → 어웨이↓)
          X=3: γ_1 + γ_2
        """
        g1 = self.gamma_raw[0]  # γ_1
        g2 = self.gamma_raw[1]  # γ_2

        gamma_H = torch.stack([
            torch.tensor(0.0), g1, g2, g1 + g2
        ])
        gamma_A = torch.stack([
            torch.tensor(0.0), g2, g1, g1 + g2
        ])

        return gamma_H, gamma_A

    def _lookup_delta(self, delta_S_idx, delta_param):
        """
        δ 룩업: delta_S_idx가 -1이면 0 반환 (ΔS=0 고정).

        delta_S_idx: [N] tensor, 값 -1,0,1,2,3
        delta_param: [4] tensor (δ 파라미터)
        """
        # -1 → 0 (고정), 나머지 → delta_param[idx]
        # 먼저 -1을 0으로 치환한 후 룩업, 다시 마스크 적용
        safe_idx = delta_S_idx.clamp(min=0)
        values = delta_param[safe_idx]
        # ΔS=0 마스크 (delta_S_idx == -1인 곳은 0으로)
        mask = (delta_S_idx >= 0).float()
        return values * mask

    def forward(self):
        """
        NLL 전체 계산.

        Returns:
            total_loss: 스칼라 (NLL + 정규화)
            components: dict (디버깅용 세부 항목)
        """
        gamma_H, gamma_A = self._build_gamma()

        # ══════════════════════════════════════════════
        # 1. 구간 적분항: Σ_k μ_k  (홈 + 어웨이)
        # ══════════════════════════════════════════════

        # 각 구간의 a
        a_h = self.a_home[self.iv_match_idx]           # [N_iv]
        a_a = self.a_away[self.iv_match_idx]           # [N_iv]

        # b[basis_idx]
        b_val = self.b[self.iv_basis_idx]              # [N_iv]

        # γ[state_X] — 홈/어웨이 각각
        gamma_h = gamma_H[self.iv_state_X]             # [N_iv]
        gamma_a = gamma_A[self.iv_state_X]             # [N_iv]

        # δ[ΔS] — 홈/어웨이 각각
        delta_h = self._lookup_delta(self.iv_delta_S_idx, self.delta_H)
        delta_a = self._lookup_delta(self.iv_delta_S_idx, self.delta_A)

        # λ = exp(a + b + γ + δ)
        log_lambda_h = a_h + b_val + gamma_h + delta_h
        log_lambda_a = a_a + b_val + gamma_a + delta_a

        # μ_k = λ × duration
        mu_h = torch.exp(log_lambda_h) * self.iv_duration   # [N_iv]
        mu_a = torch.exp(log_lambda_a) * self.iv_duration

        # 구간 적분 합 (전체 기대 득점)
        integral_sum = mu_h.sum() + mu_a.sum()

        # ══════════════════════════════════════════════
        # 2. 점 이벤트항: Σ_g ln λ(t_g)
        # ══════════════════════════════════════════════

        # 각 골의 a
        ga_h = self.a_home[self.g_match_idx]           # [N_goals]
        ga_a = self.a_away[self.g_match_idx]

        # b[basis_idx]
        gb_val = self.b[self.g_basis_idx]

        # γ[state_X]
        g_gamma_h = gamma_H[self.g_state_X]
        g_gamma_a = gamma_A[self.g_state_X]

        # δ[delta_S_before]
        g_delta_h = self._lookup_delta(self.g_delta_S_idx, self.delta_H)
        g_delta_a = self._lookup_delta(self.g_delta_S_idx, self.delta_A)

        # ln λ = a + b + γ + δ
        g_log_lambda_h = ga_h + gb_val + g_gamma_h + g_delta_h
        g_log_lambda_a = ga_a + gb_val + g_gamma_a + g_delta_a

        # 홈 골은 홈 λ, 어웨이 골은 어웨이 λ
        g_log_lambda = (
            self.g_is_home * g_log_lambda_h +
            (1 - self.g_is_home) * g_log_lambda_a
        )
        point_sum = g_log_lambda.sum()

        # ══════════════════════════════════════════════
        # 3. NLL = -( point_sum - integral_sum )
        # ══════════════════════════════════════════════
        nll = -(point_sum - integral_sum)

        # ══════════════════════════════════════════════
        # 4. ML Prior 정규화: (1/2σ²) Σ_m (a_m - a_init)²
        # ══════════════════════════════════════════════
        reg_a = (
            (self.a_home - self.a_init_home).pow(2).sum() +
            (self.a_away - self.a_init_away).pow(2).sum()
        ) / (2 * SIGMA_A ** 2)

        # ══════════════════════════════════════════════
        # 5. L2 정규화: λ_reg (||b||² + ||γ||² + ||δ||²)
        # ══════════════════════════════════════════════
        reg_l2 = LAMBDA_REG * (
            self.b.pow(2).sum() +
            self.gamma_raw.pow(2).sum() +
            self.delta_H.pow(2).sum() +
            self.delta_A.pow(2).sum()
        )

        # ══════════════════════════════════════════════
        # 총 Loss
        # ══════════════════════════════════════════════
        total_loss = nll + reg_a + reg_l2

        components = {
            "total": total_loss.item(),
            "nll": nll.item(),
            "reg_a": reg_a.item(),
            "reg_l2": reg_l2.item(),
            "integral": integral_sum.item(),
            "point": point_sum.item(),
        }

        return total_loss, components

    def clamp_params(self):
        """파라미터 클램핑 (물리적 범위 제약)"""
        with torch.no_grad():
            self.b.clamp_(-0.5, 0.5)
            self.gamma_raw[0].clamp_(-1.5, 0.0)   # γ_1: 홈 퇴장 → 홈↓
            self.gamma_raw[1].clamp_(0.0, 1.5)     # γ_2: 어웨이 퇴장 → 홈↑
            # δ_H: 뒤지면(idx 0,1) 공격↑, 앞서면(idx 2,3) 수비↓
            self.delta_H[0].clamp_(-0.5, 1.0)      # ΔS ≤ -2
            self.delta_H[1].clamp_(-0.5, 1.0)      # ΔS = -1
            self.delta_H[2].clamp_(-1.0, 0.5)      # ΔS = +1
            self.delta_H[3].clamp_(-1.0, 0.5)      # ΔS ≥ +2
            # δ_A: 어웨이 시점 (ΔS는 홈 기준이므로 방향 반대)
            self.delta_A[0].clamp_(-1.0, 0.5)      # ΔS ≤ -2 → 어웨이 앞섬
            self.delta_A[1].clamp_(-1.0, 0.5)      # ΔS = -1 → 어웨이 앞섬
            self.delta_A[2].clamp_(-0.5, 1.0)      # ΔS = +1 → 어웨이 뒤짐
            self.delta_A[3].clamp_(-0.5, 1.0)      # ΔS ≥ +2 → 어웨이 뒤짐


# ─────────────────────────────────────────────────────────
# 최적화
# ─────────────────────────────────────────────────────────

def optimize_single(data: Dict, seed: int) -> Tuple[float, Dict]:
    """
    단일 시드로 2단계 최적화 실행.

    Stage 1: Adam (대략적 수렴)
    Stage 2: L-BFGS (정밀 수렴)
    """
    model = MMPPModel(data, seed=seed)

    # ── Stage 1: Adam ──
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    best_loss = float("inf")
    patience_counter = 0
    patience = 200

    for epoch in range(ADAM_EPOCHS):
        optimizer.zero_grad()
        loss, comp = model()
        loss.backward()
        optimizer.step()
        model.clamp_params()

        if (epoch + 1) % 200 == 0:
            logger.info(
                f"    [Adam {epoch+1:4d}] "
                f"total={comp['total']:.1f}  "
                f"nll={comp['nll']:.1f}  "
                f"reg_a={comp['reg_a']:.1f}  "
                f"reg_l2={comp['reg_l2']:.2f}"
            )

        if comp["total"] < best_loss - 0.1:
            best_loss = comp["total"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    [Adam] 조기 종료 at epoch {epoch+1}")
                break

    adam_loss = best_loss

    # ── Stage 2: L-BFGS ──
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for epoch in range(LBFGS_EPOCHS):
        def closure():
            lbfgs.zero_grad()
            loss, _ = model()
            loss.backward()
            return loss

        lbfgs.step(closure)
        model.clamp_params()

        if (epoch + 1) % 50 == 0:
            _, comp = model()
            logger.info(
                f"    [L-BFGS {epoch+1:3d}] "
                f"total={comp['total']:.1f}  "
                f"nll={comp['nll']:.1f}"
            )

    # 최종 loss
    _, final_comp = model()
    final_loss = final_comp["total"]

    logger.info(f"    최종 loss: {final_loss:.2f} (Adam: {adam_loss:.2f})")

    # 파라미터 추출
    params = extract_params(model)

    return final_loss, params


def extract_params(model: MMPPModel) -> Dict:
    """학습된 파라미터를 딕셔너리로 추출"""
    with torch.no_grad():
        b = model.b.detach().numpy().tolist()
        gamma_raw = model.gamma_raw.detach().numpy().tolist()
        delta_H = model.delta_H.detach().numpy().tolist()
        delta_A = model.delta_A.detach().numpy().tolist()
        a_home = model.a_home.detach().numpy()
        a_away = model.a_away.detach().numpy()

    return {
        "b": b,
        "gamma_1": gamma_raw[0],
        "gamma_2": gamma_raw[1],
        "gamma_3": gamma_raw[0] + gamma_raw[1],  # 가산 가정
        "delta_H": delta_H,
        "delta_A": delta_A,
        "a_home": a_home,
        "a_away": a_away,
    }


# ─────────────────────────────────────────────────────────
# 결과 출력 + 해석
# ─────────────────────────────────────────────────────────

def print_results(params: Dict, data: Dict):
    """학습된 파라미터를 해석 가능하게 출력"""
    print(f"\n{'='*60}")
    print(f"  Step 1.4 최적화 결과")
    print(f"{'='*60}")

    # b: 시간 프로파일
    bin_names = ["전반 초(0-15)", "전반 중(15-30)", "전반 말(30-HT)",
                 "후반 초(HT-+15)", "후반 중(+15-+30)", "후반 말(+30-FT)"]
    print(f"\n  📊 시간 프로파일 b (구간별 득점 강도 보정):")
    for i, (name, val) in enumerate(zip(bin_names, params["b"])):
        mult = np.exp(val)
        bar = "█" * int(abs(val) * 40) if val >= 0 else ""
        bar_neg = "▓" * int(abs(val) * 40) if val < 0 else ""
        print(f"    b_{i} ({name}): {val:+.4f}  (×{mult:.3f})  {bar_neg}{bar}")

    # γ: 레드카드 패널티
    g1 = params["gamma_1"]
    g2 = params["gamma_2"]
    print(f"\n  🟥 레드카드 패널티 γ:")
    print(f"    γ_1 (홈 퇴장 → 홈 득점): {g1:+.4f}  (×{np.exp(g1):.3f})")
    print(f"    γ_2 (어웨이 퇴장 → 홈 득점): {g2:+.4f}  (×{np.exp(g2):.3f})")
    print(f"    γ_3 = γ_1+γ_2 (양팀 퇴장): {g1+g2:+.4f}  (×{np.exp(g1+g2):.3f})")
    print(f"    해석: 홈 퇴장 시 홈 득점력 {(np.exp(g1)-1)*100:.1f}%, "
          f"어웨이 퇴장 시 홈 득점력 +{(np.exp(g2)-1)*100:.1f}%")

    # δ: 스코어차 효과
    ds_labels = ["ΔS≤-2(크게 뒤짐)", "ΔS=-1(약간 뒤짐)", "ΔS=+1(약간 앞섬)", "ΔS≥+2(크게 앞섬)"]
    print(f"\n  ⚽ 스코어차 효과 δ:")
    print(f"    {'':30} {'홈팀':>10} {'어웨이팀':>10}")
    print(f"    {'ΔS=0 (동점)':30} {'0 (고정)':>10} {'0 (고정)':>10}")
    for i, label in enumerate(ds_labels):
        dh = params["delta_H"][i]
        da = params["delta_A"][i]
        print(f"    {label:30} {dh:+10.4f} {da:+10.4f}")

    print(f"\n    홈팀 해석:")
    print(f"      뒤질 때 공격 부스트: {(np.exp(params['delta_H'][1])-1)*100:+.1f}% (ΔS=-1)")
    print(f"      앞설 때 수비 전환:   {(np.exp(params['delta_H'][2])-1)*100:+.1f}% (ΔS=+1)")

    # a 분포
    a_h = params["a_home"]
    a_a = params["a_away"]
    print(f"\n  📈 경기별 기본 강도 a:")
    print(f"    홈: mean={a_h.mean():.4f}, std={a_h.std():.4f}")
    print(f"    어웨이: mean={a_a.mean():.4f}, std={a_a.std():.4f}")
    print(f"    홈-어웨이 차이 평균: {(a_h - a_a).mean():.4f} "
          f"(홈이 ×{np.exp((a_h - a_a).mean()):.3f}배)")

    print()


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def run(num_starts: int = NUM_STARTS):
    """Multi-start 최적화 실행"""
    logger.info("🔧 Step 1.4: Joint NLL 최적화")
    logger.info("=" * 60)

    # 데이터 로드
    data = load_data(DB_PATH)

    # Multi-start
    best_loss = float("inf")
    best_params = None

    for start in range(num_starts):
        logger.info(f"\n🎯 Start {start+1}/{num_starts} (seed={start})")
        loss, params = optimize_single(data, seed=start)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            logger.info(f"  ⭐ 새로운 최적! loss={loss:.2f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  최적 loss: {best_loss:.2f}")

    # 결과 출력
    print_results(best_params, data)

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 공유 파라미터 저장 (b, γ, δ)
    save_params = {
        "b": best_params["b"],
        "gamma_1": best_params["gamma_1"],
        "gamma_2": best_params["gamma_2"],
        "gamma_3": best_params["gamma_3"],
        "delta_H": best_params["delta_H"],
        "delta_A": best_params["delta_A"],
        "optimization": {
            "best_loss": best_loss,
            "num_starts": num_starts,
            "sigma_a": SIGMA_A,
            "lambda_reg": LAMBDA_REG,
        }
    }
    param_path = OUTPUT_DIR / "nll_params.json"
    with open(param_path, "w") as f:
        json.dump(save_params, f, indent=2)
    logger.info(f"  파라미터 저장: {param_path}")

    # 경기별 a 저장
    a_df = pd.DataFrame({
        "fixture_id": data["fixture_ids"],
        "a_home": best_params["a_home"],
        "a_away": best_params["a_away"],
    })
    a_path = OUTPUT_DIR / "a_calibrated.csv"
    a_df.to_csv(a_path, index=False)
    logger.info(f"  보정된 a 저장: {a_path}")

    logger.info(f"\n✅ Step 1.4 완료!")


def main():
    parser = argparse.ArgumentParser(description="Step 1.4: Joint NLL Optimization")
    parser.add_argument("--starts", type=int, default=NUM_STARTS, help="Multi-start 횟수")
    parser.add_argument("--db", type=str, default=str(DB_PATH))

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    run(num_starts=args.starts)


if __name__ == "__main__":
    main()
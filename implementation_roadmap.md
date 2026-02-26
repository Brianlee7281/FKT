# Kalshi 축구 트레이딩 시스템 — 구현 로드맵

## 이 문서의 목적

Phase 1~4 설계 문서가 "무엇을 만들어야 하는가"를 정의했다면,
이 문서는 **"어떻게 만들고, 어떻게 테스트하고, 어떻게 돌리는가"**를 정의한다.

프로그래밍 환경 세팅부터 클라우드 배포까지,
한 번도 안 해본 사람이 따라갈 수 있도록 단계별로 설명한다.

---

## 전체 그림 (Big Picture)

```
[Stage 0] 개발 환경 세팅          ← 컴퓨터에 도구 설치
     ↓
[Stage 1] 데이터 수집 파이프라인    ← API에서 과거 데이터 다운로드
     ↓
[Stage 2] Phase 1 구현 + 백테스트  ← 오프라인 모델 학습 & 검증
     ↓
[Stage 3] Phase 2~3 구현          ← 라이브 엔진 개발
     ↓
[Stage 4] Phase 4 구현            ← 주문 실행 & 리스크 관리
     ↓
[Stage 5] 종이 거래 (Paper Trading) ← 실제 돈 없이 시뮬레이션
     ↓
[Stage 6] 클라우드 배포            ← 서버에 올려서 24/7 운영
     ↓
[Stage 7] 실전 투입 + 모니터링     ← 실제 돈으로 거래 시작
```

예상 소요 시간: **3~6개월** (풀타임 기준).
서두르지 마라. 각 Stage를 확실히 검증한 후 다음으로 넘어가야 한다.

---

## Stage 0: 개발 환경 세팅

### 0.1 필요한 것들

시작하기 전에 준비해야 할 것:

| 항목 | 설명 | 비용 |
|------|------|------|
| 컴퓨터 | Mac 또는 Windows (Linux면 더 좋음) | 기존 것 사용 |
| Python 3.11+ | 핵심 프로그래밍 언어 | 무료 |
| Git | 코드 버전 관리 도구 | 무료 |
| GitHub 계정 | 코드 저장소 | 무료 |
| API-Football 구독 | 과거 축구 데이터 | 월 $10~50 |
| Sportradar API 키 | 라이브 이벤트 스트림 | 이미 보유 |
| Kalshi 계좌 | 거래소 계좌 + API 키 | 무료 개설 |
| AWS 또는 GCP 계정 | 클라우드 서버 | 월 $20~50 |

### 0.2 Python 설치

**Mac:**
```bash
# Homebrew 설치 (Mac 패키지 관리자)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치
brew install python@3.11
```

**Windows:**
1. https://www.python.org/downloads/ 에서 Python 3.11 다운로드
2. 설치 시 **"Add Python to PATH"** 반드시 체크

**설치 확인:**
```bash
python3 --version   # Python 3.11.x가 출력되면 성공
```

### 0.3 Git 설치 및 GitHub 설정

Git은 코드의 "세이브 포인트"를 만드는 도구다.
게임에서 저장하듯이, 코드가 잘 돌아갈 때마다 저장해두면
실수해도 이전 상태로 돌아갈 수 있다.

```bash
# Git 설치 (Mac)
brew install git

# 기본 설정
git config --global user.name "네 이름"
git config --global user.email "네이메일@example.com"
```

**GitHub에서 저장소 만들기:**
1. https://github.com 에서 계정 생성
2. 우측 상단 "+" → "New repository"
3. 이름: `kalshi-football-trading`
4. **Private** 선택 (코드를 비공개로)
5. "Create repository" 클릭

```bash
# 로컬에 프로젝트 폴더 생성
mkdir kalshi-football-trading
cd kalshi-football-trading
git init
git remote add origin https://github.com/네아이디/kalshi-football-trading.git
```

### 0.4 프로젝트 구조 만들기

```bash
# 폴더 구조 생성
mkdir -p src/{data,phase1,phase2,phase3,phase4,utils}
mkdir -p data/{raw,processed,models}
mkdir -p tests
mkdir -p configs
mkdir -p notebooks
mkdir -p logs
```

최종 폴더 구조:

```
kalshi-football-trading/
│
├── src/                    # 소스 코드
│   ├── data/               # 데이터 수집 및 전처리
│   │   ├── api_football.py # API-Football 클라이언트
│   │   ├── sportradar.py   # Sportradar 클라이언트
│   │   └── preprocessor.py # Phase 1.1 구간 분할
│   │
│   ├── phase1/             # 오프라인 캘리브레이션
│   │   ├── markov_chain.py # Step 1.2 Q 행렬
│   │   ├── ml_prior.py     # Step 1.3 XGBoost
│   │   ├── joint_nll.py    # Step 1.4 NLL 최적화
│   │   └── validation.py   # Step 1.5 검증
│   │
│   ├── phase2/             # 프리매치 초기화
│   │   └── initializer.py
│   │
│   ├── phase3/             # 라이브 엔진
│   │   ├── state_machine.py
│   │   ├── expected_goals.py
│   │   ├── event_handler.py
│   │   ├── pricing.py
│   │   └── mc_core.py      # Numba MC 시뮬레이션
│   │
│   ├── phase4/             # 실행
│   │   ├── orderbook.py
│   │   ├── edge_detector.py
│   │   ├── position_sizer.py
│   │   ├── exit_manager.py
│   │   └── risk_manager.py
│   │
│   └── utils/              # 공용 유틸리티
│       ├── config.py       # 설정 로드
│       ├── logger.py       # 로그 기록
│       └── database.py     # DB 연결
│
├── data/                   # 데이터 파일 (Git에 올리지 않음)
│   ├── raw/                # API에서 받은 원본
│   ├── processed/          # 전처리된 구간 데이터
│   └── models/             # 학습된 모델 파일
│
├── configs/                # 설정 파일
│   ├── config.yaml         # API 키, 파라미터 등
│   └── risk_params.yaml    # 리스크 한도
│
├── notebooks/              # Jupyter 노트북 (탐색, 분석용)
│
├── tests/                  # 테스트 코드
│
├── logs/                   # 실행 로그
│
├── requirements.txt        # Python 패키지 목록
├── Dockerfile              # 클라우드 배포용
├── docker-compose.yml
└── README.md
```

### 0.5 가상 환경 및 패키지 설치

가상 환경은 이 프로젝트 전용 Python 공간을 만드는 것이다.
다른 프로젝트와 패키지 버전이 충돌하는 것을 방지한다.

```bash
# 가상 환경 생성
python3 -m venv venv

# 활성화
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# 프롬프트가 (venv)로 바뀌면 성공
```

**requirements.txt 작성:**

```
# 데이터 처리
pandas==2.2.0
numpy==1.26.4

# ML
xgboost==2.0.3
scikit-learn==1.4.0

# 최적화 (Phase 1.4)
torch==2.2.0
scipy==1.12.0

# MC 가속 (Phase 3.4)
numba==0.59.0

# 비동기 (Phase 3 라이브 엔진)
asyncio-mqtt==0.16.2
websockets==12.0
aiohttp==3.9.3

# 데이터베이스
sqlalchemy==2.0.25
psycopg2-binary==2.9.9    # PostgreSQL

# 시각화
matplotlib==3.8.3
seaborn==0.13.2

# Jupyter (탐색용)
jupyter==1.0.0
jupyterlab==4.1.0

# 설정
pyyaml==6.0.1
python-dotenv==1.0.1

# 로깅
loguru==0.7.2

# 테스트
pytest==8.0.0
pytest-asyncio==0.23.4
```

```bash
# 패키지 설치
pip install -r requirements.txt
```

### 0.6 설정 파일 작성

**configs/config.yaml:**

```yaml
# API 키 (실제 값은 .env 파일에, 여기는 구조만)
api_football:
  base_url: "https://v3.football.api-sports.io"
  # api_key는 .env에서 로드

sportradar:
  ws_url: "wss://api.sportradar.com/soccer/v4/stream/events"
  # api_key는 .env에서 로드

kalshi:
  rest_url: "https://trading-api.kalshi.com/trade-api/v2"
  ws_url: "wss://trading-api.kalshi.com/trade-api/ws/v2"
  # api_key는 .env에서 로드

# Phase 1 파라미터
phase1:
  leagues: ["39", "140", "135", "78", "61"]   # PL, LaLiga, SerieA, Buli, L1
  seasons: [2020, 2021, 2022, 2023, 2024]
  num_time_bins: 6
  regularization:
    sigma_a: 1.0
    lambda_reg: 0.01

# Phase 3 파라미터
phase3:
  mc_simulations: 50000
  cooldown_seconds: 15
  ob_anomaly_threshold: 0.05

# Phase 4 파라미터
phase4:
  entry_threshold: 0.02       # 2¢ 최소 엣지
  exit_threshold: 0.005       # 0.5¢
  kelly_fraction: 0.25        # Quarter-Kelly
  z_conservative: 1.645       # 90% 보수적 하한
  risk:
    order_cap: 0.03           # 단일 주문 3%
    match_cap: 0.05           # 경기별 5%
    total_cap: 0.20           # 전체 20%

# 데이터베이스
database:
  url: "postgresql://user:pass@localhost:5432/kalshi_trading"
```

**.env 파일 (Git에 절대 올리면 안 됨):**

```
API_FOOTBALL_KEY=여기에_실제_키
SPORTRADAR_KEY=여기에_실제_키
KALSHI_API_KEY=여기에_실제_키
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
DATABASE_URL=postgresql://user:pass@localhost:5432/kalshi_trading
```

**.gitignore 파일:**

```
# 절대 Git에 올리면 안 되는 것들
.env
*.pem
venv/
data/
logs/
__pycache__/
*.pyc
.DS_Store
```

```bash
# 여기까지 하고 첫 번째 저장
git add .
git commit -m "Initial project structure"
git push origin main
```

---

## Stage 1: 데이터 수집 파이프라인

### 1.1 API-Football 데이터 다운로드

이 단계의 목표: 과거 5시즌의 경기 데이터를 로컬 데이터베이스에 저장.

**src/data/api_football.py:**

```python
"""
API-Football 클라이언트.
과거 경기 데이터를 다운로드하여 로컬에 저장한다.

사용법:
    python -m src.data.api_football --league 39 --season 2023
"""
import requests
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class APIFootballClient:
    """API-Football REST API 클라이언트"""

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self):
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        self.headers = {
            "x-apisports-key": self.api_key
        }
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, params: dict) -> dict:
        """API 호출 + 속도 제한 준수"""
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        # API-Football 무료 플랜: 분당 10회 제한
        time.sleep(6)  # 안전하게 6초 대기
        return response.json()

    def download_fixtures(self, league_id: int, season: int):
        """한 시즌의 모든 경기 목록 다운로드"""
        data = self._request("fixtures", {
            "league": league_id,
            "season": season,
            "status": "FT"  # 종료된 경기만
        })

        filepath = self.data_dir / f"fixtures_{league_id}_{season}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"저장 완료: {filepath} ({len(data['response'])}경기)")
        return data["response"]

    def download_events(self, fixture_id: int):
        """경기별 이벤트 (골, 레드카드 등)"""
        data = self._request("fixtures/events", {"fixture": fixture_id})

        filepath = self.data_dir / f"events_{fixture_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return data["response"]

    def download_statistics(self, fixture_id: int):
        """경기별 통계 (점유율, xG 등)"""
        data = self._request("fixtures/statistics", {"fixture": fixture_id})

        filepath = self.data_dir / f"stats_{fixture_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return data["response"]

    def download_full_season(self, league_id: int, season: int):
        """한 시즌 전체 데이터 다운로드 (수 시간 소요)"""
        print(f"\n{'='*50}")
        print(f"리그 {league_id}, 시즌 {season} 다운로드 시작")
        print(f"{'='*50}\n")

        # 1. 경기 목록
        fixtures = self.download_fixtures(league_id, season)

        # 2. 각 경기의 이벤트 + 통계
        for i, fixture in enumerate(fixtures):
            fid = fixture["fixture"]["id"]
            print(f"  [{i+1}/{len(fixtures)}] 경기 {fid}...")

            try:
                self.download_events(fid)
                self.download_statistics(fid)
            except Exception as e:
                print(f"    ⚠️ 오류: {e}")
                continue

        print(f"\n✅ 리그 {league_id} 시즌 {season} 완료!")
```

**실행:**

```bash
# 프리미어리그 2023-24 시즌 다운로드
python -m src.data.api_football --league 39 --season 2023

# 5대 리그 × 5시즌 전체 (하루~이틀 소요)
# 스크립트를 돌려놓고 다른 일을 하면 됨
```

> **팁:** API-Football 무료 플랜은 일일 100회 호출 제한이 있다.
> 5시즌 × 5리그 × 380경기 × 3호출 ≈ 28,500회 호출이 필요하므로
> **Pro 플랜 ($50/월)**을 최소 1개월 구독하는 것을 권장한다.
> 데이터를 모두 다운로드한 후에는 해지해도 된다.

### 1.2 데이터베이스 설정

다운로드한 JSON 파일을 구조화된 데이터베이스에 넣으면 쿼리가 빨라진다.

**PostgreSQL 설치:**

```bash
# Mac
brew install postgresql@15
brew services start postgresql@15

# 데이터베이스 생성
createdb kalshi_trading
```

**Windows:** https://www.postgresql.org/download/windows/ 에서 설치.

> **왜 PostgreSQL인가?** SQLite는 간편하지만 동시 접근(라이브 엔진 + 로깅)에 약하다.
> 클라우드 배포 시에도 PostgreSQL이 표준이다.
> 단, 처음 학습 단계에서는 SQLite로 시작해도 괜찮다.

### 1.3 전처리 — Phase 1.1 구간 분할

다운로드한 원시 데이터를 Phase 1.1에서 정의한 구간(Interval) 형식으로 변환한다.

이것이 전체 시스템의 첫 번째 실제 코드이며,
설계 문서의 수학을 코드로 옮기는 첫 경험이 된다.

**src/data/preprocessor.py:**

```python
"""
Phase 1.1: 시계열 이벤트 분할 및 구간 데이터화.

원시 이벤트(골, 레드카드)를 λ가 일정한 구간(Interval)으로 변환한다.
설계 문서의 "데이터 변환 예시" 테이블을 코드로 구현한 것.
"""
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class Interval:
    """하나의 구간 레코드"""
    match_id: int
    t_start: float       # 구간 시작 (분)
    t_end: float         # 구간 종료 (분)
    state_X: int          # 마르코프 상태 (0~3)
    delta_S: int          # 홈팀 기준 득점차
    is_halftime: bool     # 하프타임 구간 여부
    T_m: float            # 경기 실제 종료 시간

@dataclass
class GoalEvent:
    """골 이벤트 (NLL 점 이벤트 기여)"""
    match_id: int
    time: float           # 골 시각 (분)
    team: str             # "home" or "away"
    delta_S_before: int   # 골 직전의 ΔS (인과관계 주의!)
    state_X: int          # 골 시점의 마르코프 상태

def process_match(match_id: int, events: list, fulltime_minutes: float):
    """
    한 경기의 이벤트를 구간 + 골 이벤트로 변환.

    설계 문서의 "구간 경계(Split Point) 규칙"을 구현:
    - 레드카드: X 변경 → 분할
    - 골: ΔS 변경 → 분할
    - 하프타임: 적분 제외
    """
    intervals = []
    goal_events = []

    # 현재 상태 추적
    current_X = 0        # 11v11
    current_S_home = 0
    current_S_away = 0
    current_delta_S = 0
    current_t = 0.0

    # 이벤트를 시간순으로 정렬
    sorted_events = sorted(events, key=lambda e: e["time"])

    for event in sorted_events:
        event_time = event["time"]
        event_type = event["type"]

        if event_type == "Goal":
            # 1. 현재 구간 닫기
            if event_time > current_t:
                intervals.append(Interval(
                    match_id=match_id,
                    t_start=current_t,
                    t_end=event_time,
                    state_X=current_X,
                    delta_S=current_delta_S,
                    is_halftime=False,
                    T_m=fulltime_minutes
                ))

            # 2. 골 이벤트 기록 (직전 ΔS 사용!)
            team = "home" if event["team_side"] == "home" else "away"
            goal_events.append(GoalEvent(
                match_id=match_id,
                time=event_time,
                team=team,
                delta_S_before=current_delta_S,  # ← 인과관계 핵심
                state_X=current_X
            ))

            # 3. 상태 업데이트
            if team == "home":
                current_S_home += 1
            else:
                current_S_away += 1
            current_delta_S = current_S_home - current_S_away

            # 4. 새 구간 시작
            current_t = event_time

        elif event_type == "Red Card":
            # 구간 닫기
            if event_time > current_t:
                intervals.append(Interval(
                    match_id=match_id,
                    t_start=current_t,
                    t_end=event_time,
                    state_X=current_X,
                    delta_S=current_delta_S,
                    is_halftime=False,
                    T_m=fulltime_minutes
                ))

            # 상태 전이
            if event["team_side"] == "home":
                if current_X == 0:
                    current_X = 1
                elif current_X == 2:
                    current_X = 3
            else:
                if current_X == 0:
                    current_X = 2
                elif current_X == 1:
                    current_X = 3

            current_t = event_time

        elif event_type == "HT":  # 하프타임
            # 전반 마지막 구간 닫기
            if event_time > current_t:
                intervals.append(Interval(
                    match_id=match_id,
                    t_start=current_t,
                    t_end=event_time,
                    state_X=current_X,
                    delta_S=current_delta_S,
                    is_halftime=False,
                    T_m=fulltime_minutes
                ))

            # 하프타임 구간 (적분에서 제외됨)
            ht_end = event_time + 15  # 대략 15분
            intervals.append(Interval(
                match_id=match_id,
                t_start=event_time,
                t_end=ht_end,
                state_X=current_X,
                delta_S=current_delta_S,
                is_halftime=True,  # ← 이 플래그로 NLL에서 제외
                T_m=fulltime_minutes
            ))
            current_t = ht_end

    # 마지막 구간 닫기
    if current_t < fulltime_minutes:
        intervals.append(Interval(
            match_id=match_id,
            t_start=current_t,
            t_end=fulltime_minutes,
            state_X=current_X,
            delta_S=current_delta_S,
            is_halftime=False,
            T_m=fulltime_minutes
        ))

    return intervals, goal_events
```

### 1.4 어디까지 하면 Stage 1 완료?

Stage 1 체크리스트:

- [ ] API-Football에서 5대 리그 × 5시즌 데이터 다운로드 완료
- [ ] PostgreSQL (또는 SQLite)에 경기, 이벤트, 통계 테이블 생성
- [ ] `preprocessor.py`로 전체 데이터를 구간(Interval) + 골 이벤트로 변환
- [ ] 변환 결과를 DB의 `intervals` 테이블과 `goal_events` 테이블에 저장
- [ ] 랜덤 10경기를 골라서 변환 결과를 수작업으로 검증 (설계 문서의 예시 테이블과 비교)

```bash
git add .
git commit -m "Stage 1: Data pipeline complete"
git push
```

---

## Stage 2: Phase 1 구현 + 백테스트

### 2.1 개발 순서

Phase 1은 5개의 Step이 있지만, 코드 작성 순서는 이렇다:

```
Step 1.1 (전처리) ← Stage 1에서 완료
     ↓
Step 1.2 (Q 행렬) ← 가장 간단, 여기서 시작
     ↓
Step 1.3 (XGBoost) ← ML 모델, Jupyter에서 탐색적으로
     ↓
Step 1.4 (Joint NLL) ← 가장 복잡, PyTorch
     ↓
Step 1.5 (검증) ← 가장 중요, 여기서 실패하면 되돌아감
```

### 2.2 Jupyter Notebook으로 탐색적 개발

처음부터 완성된 코드를 짜려고 하지 마라.
**Jupyter Notebook에서 한 셀씩 실행하며 결과를 눈으로 확인**하는 것이 훨씬 효율적이다.

```bash
# Jupyter 시작
jupyter lab
```

브라우저에서 `notebooks/` 폴더에 노트북을 만든다:

- `01_data_exploration.ipynb` — 데이터 구조 파악, 기초 통계
- `02_markov_chain.ipynb` — Q 행렬 추정
- `03_xgboost_prior.ipynb` — ML 모델 학습
- `04_joint_nll.ipynb` — NLL 최적화
- `05_validation.ipynb` — 검증

**탐색 → 확인 → 함수화 → 모듈화** 순서로 진행한다.
노트북에서 잘 돌아가는 코드를 `src/phase1/` 모듈로 옮기는 것이다.

### 2.3 Step 1.2: Q 행렬 추정

```python
# notebooks/02_markov_chain.ipynb

import numpy as np
import pandas as pd

# DB에서 구간 데이터 로드
intervals = pd.read_sql("SELECT * FROM intervals WHERE is_halftime = FALSE", engine)

# 상태별 총 체류 시간 (분)
time_in_state = {}
for state in [0, 1, 2, 3]:
    mask = intervals["state_X"] == state
    time_in_state[state] = (intervals[mask]["t_end"] - intervals[mask]["t_start"]).sum()

print("상태별 총 체류 시간 (분):")
for s, t in time_in_state.items():
    print(f"  상태 {s}: {t:.0f}분 ({t/60:.0f}시간)")

# 상태 전이 횟수 (레드카드 이벤트에서 집계)
# ... (전이 횟수 / 체류 시간 = 전이율)
```

이런 식으로 셀 하나씩 실행하면서 데이터의 모양을 확인한다.

### 2.4 Step 1.3: XGBoost

```python
# notebooks/03_xgboost_prior.ipynb

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# 피처 행렬 구성
# 각 경기별: [홈 최근5경기 롤링스탯, 어웨이 롤링스탯, 배당률]
# 타겟: 해당 경기의 총 득점

model = xgb.XGBRegressor(
    objective="count:poisson",  # 푸아송 회귀
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          early_stopping_rounds=50, verbose=True)

# 피처 중요도 → feature_mask.json
importance = model.feature_importances_
```

### 2.5 Step 1.4: Joint NLL (PyTorch)

이것이 전체 시스템에서 가장 복잡한 코드이다.
설계 문서의 Loss 함수를 PyTorch로 구현한다.

```python
# src/phase1/joint_nll.py (핵심 뼈대)

import torch
import torch.nn as nn

class MMPPCalibration(nn.Module):
    """
    Phase 1.4: MMPP 파라미터 동시 최적화.

    학습 가능 파라미터:
    - a_m: 경기별 기본 강도 (M개)
    - b: 시간 구간별 프로파일 (6개)
    - gamma_1, gamma_2: 퇴장 패널티 (2개)
    - delta_H, delta_A: 스코어차 효과 (각 4개)
    """

    def __init__(self, num_matches, a_init):
        super().__init__()

        # 경기별 기본 강도 (ML 예측치로 초기화)
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))

        # 시간 프로파일 (0으로 초기화)
        self.b = nn.Parameter(torch.zeros(6))

        # 퇴장 패널티
        self.gamma_1 = nn.Parameter(torch.tensor(0.0))  # 홈 퇴장
        self.gamma_2 = nn.Parameter(torch.tensor(0.0))  # 어웨이 퇴장

        # 스코어차 효과 (ΔS ∈ {≤-2, -1, +1, ≥+2}, 0은 기준점으로 고정)
        self.delta_H = nn.Parameter(torch.zeros(4))
        self.delta_A = nn.Parameter(torch.zeros(4))

    def forward(self, intervals_batch, goals_batch):
        """
        NLL 계산.

        intervals_batch: 구간 데이터 (t_start, t_end, X, ΔS, basis_index)
        goals_batch: 골 이벤트 (time, X, ΔS_before, basis_index, match_index)
        """
        # gamma 벡터 구성: [0, γ₁, γ₂, γ₁+γ₂]
        gamma = torch.stack([
            torch.tensor(0.0),
            self.gamma_1,
            self.gamma_2,
            self.gamma_1 + self.gamma_2
        ])

        # delta 벡터 구성: [δ(≤-2), δ(-1), 0, δ(+1), δ(≥+2)]
        delta_H = torch.cat([self.delta_H[:2], torch.tensor([0.0]), self.delta_H[2:]])
        delta_A = torch.cat([self.delta_A[:2], torch.tensor([0.0]), self.delta_A[2:]])

        # ── 구간 적분 (기대 득점) ──
        # μ_k = exp(a_m + b_i + γ_X + δ(ΔS)) × (t_end - t_start)
        integral_sum = 0.0
        for interval in intervals_batch:
            m = interval.match_idx
            log_rate = (self.a[m] + self.b[interval.basis_idx]
                       + gamma[interval.X] + delta_H[interval.delta_S_idx])
            mu_k = torch.exp(log_rate) * interval.duration
            integral_sum += mu_k

        # ── 점 이벤트 합 (골 시각의 로그 강도) ──
        point_sum = 0.0
        for goal in goals_batch:
            m = goal.match_idx
            log_rate = (self.a[m] + self.b[goal.basis_idx]
                       + gamma[goal.X] + delta_H[goal.delta_S_before_idx])
            point_sum += log_rate

        # ── NLL ──
        nll = integral_sum - point_sum

        # ── 정규화 ──
        a_init_tensor = torch.tensor(self.a_init)
        reg_a = 0.5 / (self.sigma_a ** 2) * ((self.a - a_init_tensor) ** 2).sum()
        reg_l2 = self.lambda_reg * (
            self.b.pow(2).sum() + self.gamma_1.pow(2) + self.gamma_2.pow(2)
            + self.delta_H.pow(2).sum() + self.delta_A.pow(2).sum()
        )

        loss = nll + reg_a + reg_l2
        return loss
```

> **중요:** 위 코드는 뼈대이다. 실제로는 배치 처리, 파라미터 클램핑,
> Multi-start, 2단계 옵티마이저(Adam → L-BFGS)를 추가해야 한다.
> 설계 문서의 Step 1.4 섹션을 참조하며 하나씩 구현한다.

### 2.6 Step 1.5: 백테스트

이 단계가 **가장 중요하다.** 여기서 실패하면 뒤로 돌아가야 한다.

```python
# notebooks/05_validation.ipynb

from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

# Walk-Forward CV
folds = [
    {"train": [2020, 2021, 2022], "val": [2023]},
    {"train": [2020, 2021, 2022, 2023], "val": [2024]},
]

for fold in folds:
    # 학습 기간 데이터로 Phase 1 전체 실행
    # 검증 기간에서 예측

    # Calibration Plot
    plt.figure(figsize=(6, 6))
    # ... 모델 예측 확률 vs 실제 빈도 시각화
    plt.plot([0, 1], [0, 1], 'k--')  # 대각선 = 완벽한 보정
    plt.title(f"Calibration Plot - Fold {fold}")
    plt.show()

    # Brier Score
    bs = brier_score_loss(y_true, y_pred_proba)
    print(f"Brier Score: {bs:.4f}")
```

### 2.7 어디까지 하면 Stage 2 완료?

Stage 2 체크리스트:

- [ ] Q 행렬 추정 완료, 값이 합리적인지 확인 (퇴장률 ≈ 0.03~0.05/경기)
- [ ] XGBoost Poisson 모델 학습 완료, feature_mask.json 생성
- [ ] Joint NLL 최적화 수렴 확인 (loss가 감소하다 안정화)
- [ ] 학습된 b, γ, δ의 부호가 축구 직관과 부합
- [ ] Walk-Forward CV 2개 Fold 모두에서 Calibration Plot이 대각선에 근접
- [ ] Brier Score가 시장 내재 확률 대비 개선
- [ ] 모든 프로덕션 파라미터를 `data/models/` 디렉토리에 저장

```bash
git add .
git commit -m "Stage 2: Phase 1 calibration + validation complete"
git push
```

---

## Stage 3: Phase 2~3 구현 (라이브 엔진)

### 3.1 Phase 2 — 비교적 간단

Phase 2는 킥오프 전 1회 실행되는 초기화 코드이므로
복잡한 비동기 처리가 필요 없다. 순차적으로 구현하면 된다.

```python
# src/phase2/initializer.py

import numpy as np
import xgboost as xgb
from scipy.linalg import expm

class PreMatchInitializer:
    """Phase 2: 경기 전 초기화"""

    def __init__(self, phase1_params):
        self.b = phase1_params["b"]           # (6,)
        self.gamma = phase1_params["gamma"]   # (4,)
        self.delta_H = phase1_params["delta_H"]
        self.delta_A = phase1_params["delta_A"]
        self.Q = phase1_params["Q"]           # (4, 4)
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(phase1_params["xgb_path"])
        self.E_alpha = phase1_params["expected_stoppage"]

    def compute_C_time(self):
        """시간 기저함수 가중합 상수"""
        delta_t = np.array([
            15, 15, 15 + self.E_alpha[0],   # 전반
            15, 15, 15 + self.E_alpha[1]     # 후반
        ])
        return np.sum(np.exp(self.b) * delta_t)

    def invert_a(self, mu_hat):
        """a = ln(μ̂) - ln(C_time)"""
        C_time = self.compute_C_time()
        return np.log(mu_hat) - np.log(C_time)

    def precompute_P_grid(self):
        """행렬 지수함수 사전 계산 (0~100분)"""
        P_grid = {}
        for dt in range(101):
            P_grid[dt] = expm(self.Q * dt)
        return P_grid

    def initialize(self, match_features):
        """전체 초기화 실행"""
        # Step 2.2~2.3: ML 추론 + a 역산
        mu_H = self.xgb_model.predict(match_features["home"])
        mu_A = self.xgb_model.predict(match_features["away"])
        a_H = self.invert_a(mu_H)
        a_A = self.invert_a(mu_A)

        # Step 2.5: P_grid 사전 계산
        P_grid = self.precompute_P_grid()

        return {
            "a_H": a_H, "a_A": a_A,
            "mu_H": mu_H, "mu_A": mu_A,
            "C_time": self.compute_C_time(),
            "P_grid": P_grid,
            "T_exp": 90 + sum(self.E_alpha)
        }
```

### 3.2 Phase 3 — 비동기 엔진 (가장 어려운 파트)

Phase 3은 **asyncio**를 사용한 비동기 프로그래밍이 필요하다.
처음 접하면 어렵게 느껴질 수 있지만, 핵심 개념은 간단하다:

**동기(Synchronous):** 한 번에 한 가지 일만 한다. 기다리는 동안 아무것도 못 한다.

**비동기(Asynchronous):** 여러 일을 동시에 한다.
WebSocket에서 데이터를 기다리는 동안 다른 계산을 할 수 있다.

```python
# 동기 (나쁜 예)
data = websocket.recv()      # 데이터 올 때까지 멈춤
process(data)                  # 그제야 처리

# 비동기 (좋은 예)
data = await websocket.recv()  # 기다리는 동안 다른 코루틴 실행 가능
process(data)
```

**비동기를 처음 배운다면:**
1. Python 공식 문서의 asyncio 튜토리얼을 먼저 읽어라
2. `async def`, `await`, `asyncio.gather()` 3개만 이해하면 된다
3. 간단한 WebSocket 예제를 먼저 만들어 봐라

### 3.3 Numba MC 코어

Phase 3.4의 Monte Carlo 시뮬레이션은 설계 문서에 전체 코드가 있다.
그대로 `src/phase3/mc_core.py`에 복사하고, 다음을 확인한다:

```bash
# Numba 워밍업 테스트
python -c "
from src.phase3.mc_core import mc_simulate_remaining
import numpy as np
import time

# 더미 호출 (JIT 컴파일 트리거)
_ = mc_simulate_remaining(
    0, 90, 0, 0, 0, 0,
    -0.5, -0.7, np.zeros(6), np.zeros(4),
    np.zeros(5), np.zeros(5),
    np.array([-0.03, -0.03, -0.03, 0.0]), np.zeros((4,4)),
    np.array([0, 15, 30, 47, 62, 77, 97], dtype=float), 100, 42
)
print('JIT 컴파일 완료')

# 벤치마크
start = time.time()
results = mc_simulate_remaining(
    30, 97, 1, 0, 0, 1,
    -0.5, -0.7, np.zeros(6), np.zeros(4),
    np.zeros(5), np.zeros(5),
    np.array([-0.03, -0.03, -0.03, 0.0]), np.zeros((4,4)),
    np.array([0, 15, 30, 47, 62, 77, 97], dtype=float), 50000, 42
)
elapsed = (time.time() - start) * 1000
print(f'50,000 시뮬레이션: {elapsed:.1f}ms')
print(f'평균 최종 스코어: Home={results[:,0].mean():.2f}, Away={results[:,1].mean():.2f}')
"
```

0.5~2ms 정도가 나오면 성공이다.

### 3.4 라이브 엔진 메인 루프

```python
# src/phase3/engine.py (뼈대)

import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor

class LiveTradingEngine:
    """Phase 3: 라이브 트레이딩 엔진 메인 루프"""

    def __init__(self, init_params):
        self.params = init_params
        self.mc_executor = ThreadPoolExecutor(max_workers=4)

        # 상태 변수
        self.t = 0.0
        self.score = [0, 0]
        self.state_X = 0
        self.delta_S = 0
        self.engine_phase = "WAITING"
        self.cooldown = False
        self.ob_freeze = False

    async def run(self):
        """메인 진입점 — 3개의 코루틴을 동시에 실행"""
        await asyncio.gather(
            self.tick_loop(),           # 1초 틱
            self.sportradar_listener(), # 이벤트 수신
            self.kalshi_listener()      # 호가 수신
        )

    async def tick_loop(self):
        """매 1초마다 실행: 시간 감쇠 + 프라이싱"""
        while self.engine_phase != "FINISHED":
            if self.engine_phase in ("FIRST_HALF", "SECOND_HALF"):
                self.t += 1/60

                # 호가 이상 감지
                self.check_ob_anomaly()

                # 잔여 기대 득점
                mu_H, mu_A = self.compute_remaining_mu()

                # 프라이싱 (MC는 executor에서)
                P_true, sigma = await self.price_async(mu_H, mu_A)

                # Phase 4로 전달
                order_ok = (not self.cooldown) and (not self.ob_freeze)
                await self.send_to_phase4(P_true, sigma, order_ok)

            await asyncio.sleep(1)

    async def sportradar_listener(self):
        """Sportradar WebSocket 이벤트 수신"""
        async with websockets.connect(self.params["sportradar_url"]) as ws:
            async for message in ws:
                event = json.loads(message)
                self.handle_event(event)

    async def kalshi_listener(self):
        """Kalshi 호가 수신 + 호가 이상 감지"""
        async with websockets.connect(self.params["kalshi_ws_url"]) as ws:
            async for message in ws:
                orderbook = json.loads(message)
                self.update_orderbook(orderbook)
```

### 3.5 어디까지 하면 Stage 3 완료?

Stage 3 체크리스트:

- [ ] Phase 2 초기화가 정상 동작 (a_H, a_A 값이 합리적)
- [ ] Sanity Check가 실제 경기 데이터에서 Go/Hold/Skip을 올바르게 판정
- [ ] Numba MC 코어가 50,000 시뮬레이션을 1ms 이내에 완료
- [ ] asyncio 메인 루프가 3개 코루틴을 동시에 실행
- [ ] **과거 경기 데이터로 시뮬레이션:** 실제 이벤트 시퀀스를 재생하면서 P_true가 합리적으로 변동하는지 확인

> 마지막 항목이 핵심이다. 라이브 API를 연결하기 전에,
> 과거 경기의 이벤트를 시간순으로 재생(replay)하면서
> 엔진이 올바르게 반응하는지 테스트한다.

---

## Stage 4: Phase 4 구현 (주문 실행)

### 4.1 Kalshi API 연동

Kalshi API 문서: https://trading-api.kalshi.com/trade-api/v2

```python
# src/phase4/kalshi_client.py

import requests
import json

class KalshiClient:
    """Kalshi REST API 클라이언트"""

    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(self, api_key, private_key_path):
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.token = None

    def login(self):
        """API 인증 토큰 발급"""
        # Kalshi의 인증 방식에 따라 구현
        # (RSA 서명 기반 — Kalshi 문서 참조)
        pass

    def get_balance(self):
        """계좌 잔고 조회"""
        resp = requests.get(
            f"{self.BASE_URL}/portfolio/balance",
            headers=self._auth_headers()
        )
        return resp.json()["balance"]

    def place_order(self, ticker, side, count, price):
        """주문 제출"""
        payload = {
            "ticker": ticker,
            "action": "buy",
            "side": side,        # "yes" or "no"
            "type": "limit",
            "count": count,
            "yes_price": price   # 센트 단위
        }
        resp = requests.post(
            f"{self.BASE_URL}/portfolio/orders",
            headers=self._auth_headers(),
            json=payload
        )
        return resp.json()

    def cancel_order(self, order_id):
        """주문 취소"""
        resp = requests.delete(
            f"{self.BASE_URL}/portfolio/orders/{order_id}",
            headers=self._auth_headers()
        )
        return resp.json()
```

### 4.2 Kelly Criterion + Risk Manager

설계 문서의 수식을 그대로 코드로 옮긴다.
여기서는 수식과 코드의 1:1 매핑이 중요하다.

```python
# src/phase4/position_sizer.py

class PositionSizer:
    """Phase 4.3: Fee-Adjusted Kelly + 3-Layer 리스크 한도"""

    def __init__(self, config):
        self.fee_rate = config["fee_rate"]
        self.kelly_frac = config["kelly_fraction"]     # 0.25
        self.order_cap = config["risk"]["order_cap"]    # 0.03
        self.match_cap = config["risk"]["match_cap"]    # 0.05
        self.total_cap = config["risk"]["total_cap"]    # 0.20

    def compute_kelly(self, P_true_cons, P_kalshi, direction):
        """Fee-Adjusted Kelly 비중 계산"""
        c = self.fee_rate

        if direction == "BUY_YES":
            EV = P_true_cons * (1-c) * (1 - P_kalshi) - (1 - P_true_cons) * P_kalshi
            denominator = (1-c) * (1 - P_kalshi) * P_kalshi
        else:  # BUY_NO
            EV = (1 - P_true_cons) * (1-c) * P_kalshi - P_true_cons * (1 - P_kalshi)
            denominator = (1-c) * P_kalshi * (1 - P_kalshi)

        if EV <= 0 or denominator <= 0:
            return 0.0

        f_star = EV / denominator
        return f_star * self.kelly_frac  # Fractional Kelly

    def apply_risk_limits(self, f_invest, bankroll,
                          current_match_exposure, current_total_exposure):
        """3-Layer 리스크 한도 적용"""
        amount = f_invest * bankroll

        # Layer 1: 단일 주문
        max_order = bankroll * self.order_cap
        amount = min(amount, max_order)

        # Layer 2: 경기별
        remaining_match = bankroll * self.match_cap - current_match_exposure
        if remaining_match <= 0:
            return 0.0
        amount = min(amount, remaining_match)

        # Layer 3: 전체 포트폴리오
        remaining_total = bankroll * self.total_cap - current_total_exposure
        if remaining_total <= 0:
            return 0.0
        amount = min(amount, remaining_total)

        return amount
```

---

## Stage 5: 종이 거래 (Paper Trading)

### 5.1 종이 거래란?

실제 돈을 쓰지 않고, 시스템이 "주문을 냈을 것"을 기록만 하는 것이다.

| 항목 | 종이 거래 | 실전 거래 |
|------|----------|----------|
| 시장 데이터 | 실시간 (진짜) | 실시간 (진짜) |
| 주문 실행 | 기록만 함 (가상) | 실제 Kalshi API 호출 |
| 돈 | 가상 잔고 | 실제 잔고 |
| 목적 | 시스템 검증 | 수익 창출 |

### 5.2 구현 방법

```python
# src/phase4/paper_trader.py

class PaperTrader:
    """종이 거래: 실제 주문 없이 시스템 검증"""

    def __init__(self, initial_bankroll=10000):
        self.bankroll = initial_bankroll
        self.positions = []
        self.trade_log = []

    def place_order(self, ticker, side, count, price, P_true, EV_adj):
        """가상 주문 — 시장가로 즉시 체결됐다고 가정"""
        self.trade_log.append({
            "timestamp": datetime.now(),
            "ticker": ticker,
            "side": side,
            "count": count,
            "price": price,
            "P_true": P_true,
            "EV_adj": EV_adj,
            "bankroll": self.bankroll
        })
        # 실제 API는 호출하지 않음!
        print(f"[PAPER] {side} {count}x {ticker} @ {price}¢")
```

### 5.3 종이 거래 기간

최소 **2주, 50경기 이상**을 종이 거래로 돌린다.
이 기간에 확인할 것:

- 시스템이 크래시 없이 안정적으로 돌아가는가?
- P_true와 실제 결과의 Brier Score가 Phase 1.5 수준을 유지하는가?
- Edge 실현율이 0.5 이상인가?
- 가상 P&L이 양수인가?

하나라도 실패하면 **Stage 2~3으로 돌아간다.**

---

## Stage 6: 클라우드 배포

### 6.1 왜 클라우드인가?

내 컴퓨터로 돌리면 안 되는 이유:

- 내 컴퓨터가 꺼지면(잠자기, 업데이트, 정전) 시스템이 죽는다
- 포지션이 열린 상태에서 시스템이 죽으면 청산 로직이 작동 안 한다
- 네트워크 불안정 (WiFi 끊김 등)
- 거래소 서버와의 물리적 거리 → 레이턴시

### 6.2 클라우드 선택

| 서비스 | 장점 | 월 비용 (예상) |
|--------|------|--------------|
| **AWS EC2** | 가장 범용적, 문서 풍부 | $20~40 |
| Google Cloud | 비슷한 수준 | $20~40 |
| DigitalOcean | 더 간단, 초보 친화적 | $12~24 |

> **추천:** 처음이라면 **DigitalOcean**이 가장 쉽다.
> AWS는 기능이 너무 많아서 길을 잃기 쉽다.

### 6.3 Docker로 패키징

Docker는 "내 컴퓨터에서 돌아가는 환경"을 통째로 포장하는 도구다.
서버에서도 동일한 환경이 보장된다.

**Docker 설치:** https://docs.docker.com/get-docker/

**Dockerfile:**

```dockerfile
# Python 3.11 기반 이미지
FROM python:3.11-slim

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 패키지 설치 (의존성만 먼저 → 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY configs/ ./configs/

# 실행
CMD ["python", "-m", "src.main"]
```

**docker-compose.yml:**

```yaml
version: "3.8"

services:
  # 트레이딩 엔진
  engine:
    build: .
    env_file: .env
    restart: always           # 크래시 시 자동 재시작
    depends_on:
      - db
    volumes:
      - ./logs:/app/logs
      - ./data/models:/app/data/models

  # 데이터베이스
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: kalshi_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

### 6.4 서버 세팅 (DigitalOcean 예시)

**1. Droplet(서버) 생성:**
- https://cloud.digitalocean.com 에서 계정 생성
- "Create Droplet" 클릭
- OS: Ubuntu 24.04
- Plan: Basic $12/월 (2 vCPU, 2GB RAM) — 시작하기에 충분
- Region: New York (Kalshi 서버와 가까운 곳)
- "Create Droplet"

**2. 서버 접속:**
```bash
ssh root@서버IP주소
```

**3. Docker 설치:**
```bash
# 서버에서 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
apt install docker-compose-plugin
```

**4. 코드 배포:**
```bash
# 서버에서 실행
git clone https://github.com/네아이디/kalshi-football-trading.git
cd kalshi-football-trading

# .env 파일 생성 (API 키 등)
nano .env
# ... 키 입력 ...

# 모델 파일 업로드 (로컬에서)
scp -r data/models/ root@서버IP:/root/kalshi-football-trading/data/models/

# 실행
docker compose up -d    # -d: 백그라운드 실행
```

**5. 로그 확인:**
```bash
docker compose logs -f engine     # 실시간 로그
```

**6. 상태 모니터링:**
```bash
docker compose ps                 # 서비스 상태
docker stats                      # CPU/메모리 사용량
```

### 6.5 자동 재시작 및 알림

```yaml
# docker-compose.yml에 추가
services:
  engine:
    restart: always    # 크래시 시 자동 재시작
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
```

**크래시 알림 (선택):**
Slack이나 Telegram으로 알림을 보내는 것을 추천한다:

```python
# src/utils/alerter.py
import requests

def send_alert(message):
    """텔레그램으로 알림 전송"""
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": message})
```

---

## Stage 7: 실전 투입 + 모니터링

### 7.1 실전 투입 체크리스트

**절대 서두르지 마라.** 다음을 모두 확인한 후에만 실전에 투입한다:

- [ ] Stage 2~5의 모든 체크리스트 통과
- [ ] 종이 거래 2주 이상 완료, 양의 가상 P&L
- [ ] 클라우드에서 24시간 이상 무크래시 운영
- [ ] Circuit breaker 테스트 (의도적으로 WebSocket 끊어서 복구 확인)
- [ ] Kalshi 실계좌에 소액 입금 ($100~$500)

### 7.2 점진적 투입

| 단계 | 기간 | 자본 | Kelly 비율 |
|------|------|------|-----------|
| 1단계 | 2주 | $100~$200 | 0.10 (10% Kelly) |
| 2단계 | 4주 | $500 | 0.25 (Quarter) |
| 3단계 | 8주+ | $1,000+ | 0.50 (Half) |

처음에는 의도적으로 아주 적은 돈으로 시작한다.
시스템이 돈을 벌든 잃든, **2주간은 절대 파라미터를 바꾸지 않는다.**
통계적으로 유의미한 샘플(50거래+)이 쌓인 후에 판단한다.

### 7.3 모니터링 대시보드

매일 확인해야 할 지표:

| 지표 | 건강 | 경고 | 위험 |
|------|------|------|------|
| 누적 Brier Score | Phase 1.5 수준 ± 0.02 | ± 0.05 | 벗어남 |
| Edge 실현율 | 0.7 ~ 1.3 | 0.5 ~ 0.7 | < 0.5 |
| 누적 P&L | 양의 기울기 | 횡보 | 음의 기울기 |
| Max Drawdown | < 10% | 10~20% | > 20% |
| 엔진 가동률 | 99%+ | 95~99% | < 95% |

**위험 신호 시 행동:**
1. 즉시 Kelly 비율을 0.10으로 하향
2. 원인 분석 (데이터? 모델? 실행?)
3. 필요 시 라이브 중단하고 Phase 1 재학습

### 7.4 재학습 사이클

100경기 이상 축적되면 Phase 1 재학습을 고려한다.
새 시즌 시작 시에는 **의무적으로** 재학습한다.

```
매 100경기: 사후 분석 → 재학습 판단
새 시즌: 의무 재학습 → 파라미터 교체 → 종이 거래 1주 → 실전 복귀
```

---

## 핵심 도구 요약

| 도구 | 용도 | 배우는 순서 |
|------|------|-----------|
| **Python** | 모든 코드 | 1순위 |
| **Git/GitHub** | 코드 버전 관리 | 1순위 |
| **Jupyter Notebook** | 탐색적 개발 | 1순위 |
| **PostgreSQL** | 데이터 저장 | 2순위 |
| **PyTorch** | Phase 1.4 NLL 최적화 | 3순위 |
| **XGBoost** | Phase 1.3 ML 모델 | 3순위 |
| **Numba** | Phase 3.4 MC 가속 | 4순위 |
| **asyncio/websockets** | Phase 3 라이브 엔진 | 4순위 |
| **Docker** | 클라우드 배포 패키징 | 5순위 |
| **DigitalOcean/AWS** | 클라우드 서버 | 5순위 |

---

## 학습 리소스 추천

| 주제 | 리소스 | 비고 |
|------|--------|------|
| Python 기초 | 점프 투 파이썬 (무료 온라인) | 한국어 |
| Git | Git 간편 안내서 (rogerdudler.github.io) | 한국어, 30분 |
| asyncio | Python 공식 문서 asyncio 튜토리얼 | 영어 |
| XGBoost | XGBoost 공식 튜토리얼 | 영어 |
| PyTorch | PyTorch 60분 블리츠 튜토리얼 | 영어, 한국어 번역 있음 |
| Docker | Docker 공식 Getting Started | 영어 |
| PostgreSQL | PostgreSQL Tutorial (postgresqltutorial.com) | 영어 |
| Kalshi API | docs.kalshi.com | 영어 |

---

## 마지막 조언

1. **한 번에 다 만들려고 하지 마라.** Stage 0~7을 순서대로, 각 단계를 확실히 검증한 후에 다음으로 넘어가라.

2. **작은 조각부터 테스트하라.** 전체 시스템을 한 번에 돌리기 전에, 각 함수가 올바른 결과를 내는지 단위 테스트(unit test)를 작성하라.

3. **로그를 남겨라.** 모든 주문, 모든 이벤트, 모든 오류를 기록하라. 문제가 생겼을 때 로그가 없으면 원인을 찾을 수 없다.

4. **돈을 잃을 준비를 하라.** 아무리 완벽한 시스템이라도 처음에는 돈을 잃을 수 있다. 잃어도 생활에 지장이 없는 금액만 투입하라.

5. **과적합을 경계하라.** 백테스트에서 완벽한 성과가 나왔다면, 그건 십중팔구 과적합이다. Phase 1.5의 Walk-Forward CV를 철저히 수행하라.

6. **시장이 바뀐다.** 축구 전술, 규칙, Kalshi 수수료 구조는 변한다. 시스템을 한 번 만들고 방치하면 안 된다. 정기적인 재학습과 모니터링이 필수다.

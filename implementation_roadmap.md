# Kalshi 축구 트레이딩 시스템 — 구현 로드맵 v3

## 변경 이력

| 버전 | 날짜 | 내용 |
|------|------|------|
| v1 | 2025-12 | 초기 작성 (설계 기반 청사진) |
| v2 | 2026-02 | Phase 2 데이터 소스 변경 (Sportradar → Goalserve) |
| **v3** | **2026-03-01** | **실제 코드베이스 대조 + 첫 라이브 테스트 반영** |

### v3 주요 변경

- Stage 0~4 완료 상태 반영 (코드 18,500줄, 테스트 84/84)
- 실제 폴더 구조·파일명·아키텍처로 전면 갱신
- Sportradar → Goalserve REST 폴링으로 전환 반영
- Orchestrator 레이어 추가 (설계에 없던 신규 모듈)
- Stage 5 첫 라이브 테스트 결과 + 발견된 버그·수정 반영
- 뼈대 코드 예시를 실제 구현체 참조로 교체

---

## 이 문서의 목적

Phase 1~4 설계 문서가 "무엇을 만들어야 하는가"를 정의했다면,
이 문서는 **"어떻게 만들고, 어떻게 테스트하고, 어떻게 돌리는가"**를 정의한다.

---

## 전체 그림 (Big Picture)

```
[Stage 0] 개발 환경 세팅               ✅ 완료
     ↓
[Stage 1] 데이터 수집 파이프라인         ✅ 완료
     ↓
[Stage 2] Phase 1 구현 + 백테스트       ✅ 완료 (2,465줄)
     ↓
[Stage 3] Phase 2~3 구현               ✅ 완료 (4,291줄)
     ↓
[Stage 4] Phase 4 + 오케스트레이터      ✅ 완료 (3,934줄)
     ↓
[Stage 5] Paper Trading                ◐ 진행 중 (첫 라이브 테스트 완료)
     ↓
[Stage 6] 클라우드 배포                 ⬚ 예정
     ↓
[Stage 7] 실전 투입 + 모니터링          ⬚ 예정
```

---

## Stage 0: 개발 환경 세팅 ✅

### 0.1 필요한 것들

| 항목 | 설명 | 상태 |
|------|------|------|
| Python 3.11+ | 핵심 프로그래밍 언어 | ✅ |
| Git / GitHub | 코드 버전 관리 | ✅ |
| API-Football 구독 | 과거 축구 데이터 (프리매치) | ✅ |
| **Goalserve API 키** | **라이브 이벤트 (REST 폴링)** | ✅ |
| Kalshi 계좌 + API 키 | 거래소 계좌 | ✅ |
| AWS 또는 GCP 계정 | 클라우드 서버 | ⬚ Stage 6 |

> **v1 대비 변경:** Sportradar 제거. 라이브 데이터 소스를 Goalserve Soccer Live Score로 교체.
> REST 폴링 3초 간격, IP 화이트리스트 인증. 월 $65 (Soccer Live Score 패키지).

### 0.4 프로젝트 구조 (실제)

```
kalshi-football-trading/
│
├── src/
│   ├── data/                  # 데이터 수집 및 전처리
│   │   ├── api_football.py    # API-Football 클라이언트 (951줄)
│   │   └── preprocessor.py    # Phase 1.1 구간 분할 (952줄)
│   │
│   ├── phase1/                # 오프라인 캘리브레이션
│   │   ├── markov_chain.py    # Step 1.2 Q 행렬 (384줄)
│   │   ├── ml_prior.py        # Step 1.3 XGBoost (612줄)
│   │   ├── joint_nll.py       # Step 1.4 NLL 최적화 (705줄)
│   │   └── validation.py      # Step 1.5 Walk-Forward CV (764줄)
│   │
│   ├── phase2/                # 프리매치 초기화
│   │   └── initializer.py     # Step 2.1~2.5 전체 (1,125줄)
│   │
│   ├── phase3/                # 라이브 엔진
│   │   ├── engine.py          # 메인 오케스트레이션 루프 (485줄)
│   │   ├── event_source.py    # EventSource ABC + NormalizedEvent (280줄)
│   │   ├── event_handler.py   # 골/퇴장/하프타임 처리 (245줄)
│   │   ├── goalserve.py       # Goalserve REST 폴링 소스 (585줄)
│   │   ├── replay.py          # 과거 데이터 리플레이 소스 (476줄)
│   │   ├── mu_calculator.py   # 잔여 μ 계산 (306줄)
│   │   ├── pricer.py          # 해석적 + MC 프라이싱 (277줄)
│   │   ├── mc_core.py         # Numba MC 시뮬레이션 (357줄)
│   │   └── stoppage.py        # 추가시간 관리 (154줄)
│   │
│   ├── phase4/                # 실행
│   │   ├── edge_detector.py   # Fee-adjusted EV 탐색 (337줄)
│   │   ├── position_sizer.py  # Fractional Kelly + 3-layer (235줄)
│   │   ├── exit_manager.py    # 3가지 청산 트리거 (411줄)
│   │   ├── execution_engine.py # 포트폴리오 + 주문 관리 (599줄)
│   │   ├── kalshi_client.py   # Kalshi REST API 클라이언트 (539줄)
│   │   └── post_match.py      # Brier Score + 대시보드 (408줄)
│   │
│   └── orchestrator/          # ★ 신규: 통합 오케스트레이션
│       ├── ticker_mapper.py   # Kalshi 마켓 탐색·매핑 (519줄)
│       ├── tick_router.py     # 매 틱 라우팅 (327줄)
│       ├── match_session.py   # 경기 라이프사이클 관리 (286줄)
│       └── orderbook_poller.py # Kalshi 호가 폴링 (273줄)
│
├── tools/                     # 운영 도구
│   ├── paper_runner.py        # 라이브 Paper Trading 러너 (655줄)
│   ├── goalserve_verify.py    # Goalserve 연결 검증 (448줄)
│   ├── kalshi_verify.py       # Kalshi API 검증 (411줄)
│   ├── ticker_mapper.py       # 마켓 탐색 도구
│   ├── discover_series.py     # Kalshi 시리즈 탐색
│   └── find_soccer_tickers.py # 축구 티커 조회
│
├── tests/                     # 테스트 (19파일, 6,043줄)
│   ├── test_engine.py
│   ├── test_mu_calculator.py
│   ├── test_mc_core.py
│   ├── test_pricer.py
│   ├── test_event_handler.py
│   ├── test_event_source.py
│   ├── test_goalserve.py
│   ├── test_replay.py
│   ├── test_stoppage.py
│   ├── test_edge_detector.py
│   ├── test_position_sizer.py
│   ├── test_exit_manager.py
│   ├── test_execution_engine.py
│   ├── test_post_match.py
│   ├── test_kalshi_client.py
│   ├── test_orchestrator.py
│   ├── test_orderbook_poller.py
│   ├── test_replay_paper.py     # 리플레이 Paper Trading
│   └── test_replay_sim.py       # 리플레이 시뮬레이션
│
├── data/
│   ├── kalshi_football.db     # SQLite 메인 DB
│   └── models/                # Phase 1 프로덕션 파라미터
│
├── logs/                      # 실행 로그
├── .env                       # API 키 (Git 미포함)
└── requirements.txt
```

> **v1 대비 변경:**
> - `src/utils/` 폴더 미사용 (로깅은 표준 `logging` 모듈 직접 사용)
> - `configs/` 폴더 미사용 (.env + 코드 내 상수로 관리)
> - `notebooks/` 미사용 (직접 모듈로 개발)
> - **`src/orchestrator/` 신규 추가** — Phase 3↔Phase 4 연결 계층
> - `tools/` 신규 추가 — 운영·검증 스크립트
> - Phase 3의 `state_machine.py`, `expected_goals.py`, `pricing.py` → `event_handler.py`, `mu_calculator.py`, `pricer.py`로 명칭 변경
> - Phase 4의 `orderbook.py`, `risk_manager.py` → `execution_engine.py`로 통합

### 0.5 패키지 의존성 (실제)

```
# 데이터 처리
pandas>=2.2
numpy>=1.26

# ML
xgboost>=2.0
scikit-learn>=1.4

# 최적화 (Phase 1.4)
torch>=2.2
scipy>=1.12

# MC 가속 (Phase 3.4)
numba>=0.59

# 비동기 + HTTP (Phase 3 라이브 엔진)
httpx>=0.27           # Goalserve REST 폴링 (asyncio 네이티브)
aiohttp>=3.9          # Kalshi REST

# 데이터베이스
# SQLite3 (표준 라이브러리, 별도 설치 불요)

# 설정
python-dotenv>=1.0

# 테스트
pytest>=8.0
pytest-asyncio>=0.23
```

> **v1 대비 변경:**
> - `websockets` 제거 → Goalserve는 REST 폴링이므로 WebSocket 불요
> - `asyncio-mqtt` 제거
> - `psycopg2-binary` 제거 → SQLite 사용 (PostgreSQL은 Stage 6에서 도입)
> - `httpx` 추가 → Goalserve 비동기 HTTP 클라이언트
> - `sqlalchemy` 제거 → 직접 `sqlite3` 사용
> - `loguru` 제거 → 표준 `logging` 모듈 사용

### 0.6 설정 파일

**.env:**

```
API_FOOTBALL_KEY=xxx
GOALSERVE_API_KEY=xxx
KALSHI_API_KEY_ID=xxx
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
```

> **v1 대비 변경:**
> - `SPORTRADAR_KEY` → `GOALSERVE_API_KEY`
> - `DATABASE_URL` 제거 (SQLite는 파일 경로로 관리)
> - `configs/config.yaml` 미사용 — Phase별 상수는 각 모듈 내 정의

---

## Stage 1: 데이터 수집 파이프라인 ✅

v1과 동일. `src/data/api_football.py` (951줄), `src/data/preprocessor.py` (952줄) 완성.
5대 리그 × 5시즌 데이터 → SQLite `data/kalshi_football.db`에 저장.

체크리스트:
- [x] API-Football에서 5대 리그 × 5시즌 데이터 다운로드
- [x] SQLite에 경기, 이벤트, 통계 테이블 생성
- [x] `preprocessor.py`로 구간(Interval) + 골 이벤트 변환
- [x] 변환 결과를 `intervals`, `goal_events`, `match_meta` 테이블에 저장

> **v1 대비 변경:** PostgreSQL 대신 **SQLite** 사용.
> 개발·테스트 단계에서 충분하며, 별도 서버 설치 불요.
> 클라우드 배포(Stage 6) 시 PostgreSQL/TimescaleDB로 마이그레이션 예정.

---

## Stage 2: Phase 1 구현 + 백테스트 ✅

### 구현 현황

| Step | 파일 | 줄 수 | 상태 |
|------|------|-------|------|
| 1.1 전처리 | `src/data/preprocessor.py` | 952 | ✅ |
| 1.2 Q 행렬 | `src/phase1/markov_chain.py` | 384 | ✅ |
| 1.3 XGBoost | `src/phase1/ml_prior.py` | 612 | ✅ |
| 1.4 Joint NLL | `src/phase1/joint_nll.py` | 705 | ✅ |
| 1.5 검증 | `src/phase1/validation.py` | 764 | ✅ |

v1의 뼈대 코드 대비 실제 구현에서 추가된 것:
- `joint_nll.py`: 배치 처리, 파라미터 클램핑, Multi-start, Adam→L-BFGS 2단계 옵티마이저 모두 구현
- `validation.py`: Walk-Forward CV + Calibration Plot + Brier Score + LR Test + Go/No-Go 판정 전체 구현
- `ml_prior.py`: Feature importance 기반 선택 + `feature_mask.json` 자동 생성

체크리스트:
- [x] Q 행렬 추정 완료
- [x] XGBoost Poisson 모델 학습, feature_mask.json 생성
- [x] Joint NLL 최적화 수렴 확인
- [x] b, γ, δ 부호 축구 직관과 부합
- [x] Walk-Forward CV Calibration Plot 대각선 근접
- [x] Brier Score 시장 내재 확률 대비 개선
- [x] 프로덕션 파라미터 `data/models/`에 저장

---

## Stage 3: Phase 2~3 구현 ✅

### Phase 2: Pre-Match Initialization

| Step | 함수/클래스 | 위치 | 상태 |
|------|-----------|------|------|
| 2.1 데이터 수집 | `fetch_prematch_api()` | initializer.py | ✅ |
| 2.2 피처 선택 | `assemble_features()` | initializer.py | ✅ |
| 2.3 a 역산 | `compute_a()`, `compute_C_time()` | initializer.py | ✅ |
| 2.4 Sanity Check | `run_sanity_check()` | initializer.py | ✅ |
| 2.5 시스템 초기화 | `initialize_match()` | initializer.py | ✅ |

> **참고:** Paper Trading에서는 Phase 1 모델 파일 없이 테스트하기 위해
> `tools/paper_runner.py`의 `make_default_engine_params()`가 기본값으로
> Phase 2를 우회한다. Phase 2 정식 연결은 Stage 5 안정화 후.

### Phase 3: Live Trading Engine

v1 로드맵의 뼈대(3개 코루틴 gather)와 실제 구현의 차이:

| v1 설계 | 실제 구현 | 이유 |
|---------|----------|------|
| `asyncio.gather(tick_loop, sportradar_listener, kalshi_listener)` | `asyncio.gather(tick_loop, event_loop)` | Kalshi 호가는 별도 폴러(orchestrator)에서 처리 |
| Sportradar WebSocket Push | Goalserve REST 폴링 3초 | 비용 + 접근성 |
| `self.t += 1/60` (엔진 자체 시간 전진) | 이벤트에서 전달되는 `event.minute` 사용 | **🔴 CRITICAL BUG — 시간이 진행되지 않음** |
| `state_machine.py` + `expected_goals.py` + `pricing.py` | `event_handler.py` + `mu_calculator.py` + `pricer.py` | 역할은 동일, 명칭 변경 |

**실제 Phase 3 아키텍처:**

```
                   GoalserveSource
                  (REST 폴링 3초)
                        │
                        ▼
              ┌── event_loop ──┐
              │  이벤트 수신     │
              │  → event_queue  │
              └────────┬───────┘
                       │
              ┌── tick_loop ───┐
              │  ① 이벤트 드레인│
              │  ② event_handler│
              │  ③ mu_calculator│
              │  ④ pricer       │
              │  ⑤ on_tick 콜백 │
              │  sleep(3초)     │
              └────────┬───────┘
                       │
              ┌── on_tick ─────┐
              │  MatchSession   │
              │  TickRouter     │
              │  Phase 4 전체   │
              └────────────────┘
```

| Step | 파일 | 줄 수 | 상태 |
|------|------|-------|------|
| 3.1 상태 머신 + 이벤트 수신 | engine.py, event_handler.py | 730 | ✅ |
| 3.2 μ 잔여 계산 | mu_calculator.py | 306 | ✅ |
| 3.3 이벤트 처리 (골/퇴장 점프) | event_handler.py | 245 | ✅ |
| 3.4 MC 시뮬레이션 / 해석적 | mc_core.py, pricer.py | 634 | ✅ |
| 3.5 추가시간 처리 | stoppage.py | 154 | ✅ |
| — 데이터 소스: Goalserve | goalserve.py | 585 | ✅ |
| — 데이터 소스: Replay | replay.py | 476 | ✅ |
| — 이벤트 인터페이스 | event_source.py | 280 | ✅ |

체크리스트:
- [x] Phase 2 초기화 정상 동작 (a_H, a_A 역산 검증)
- [x] Numba MC 코어 50,000 시뮬레이션 < 2ms
- [x] asyncio 메인 루프 2개 코루틴 동시 실행
- [x] 과거 경기 리플레이 시뮬레이션 (15경기 EPL, +$522)
- [x] Goalserve 라이브 연결 검증 (1.2~2.7s 레이턴시)
- [ ] **🔴 시간 매 틱 전진 — 미구현 (P0 수정 필요)**

---

## Stage 4: Phase 4 + 오케스트레이터 ✅

### Phase 4: Execution

v1 로드맵의 뼈대와 실제 구현의 차이:

| v1 설계 | 실제 구현 |
|---------|----------|
| `orderbook.py` | 오케스트레이터의 `orderbook_poller.py`에 통합 |
| `risk_manager.py` | `position_sizer.py`에 3-layer 리스크 통합 |
| `paper_trader.py` | `execution_engine.py`의 `paper=True` 모드 |
| — | `post_match.py` 신규 (Brier Score + 대시보드) |
| — | `kalshi_client.py` 신규 (RSA 인증 + REST) |

| Step | 파일 | 줄 수 | 상태 |
|------|------|-------|------|
| 4.2 Edge 판별 | edge_detector.py | 337 | ✅ |
| 4.3 포지션 사이징 | position_sizer.py | 235 | ✅ |
| 4.4 청산 로직 | exit_manager.py | 411 | ✅ |
| 4.5 주문 실행 | execution_engine.py | 599 | ✅ |
| 4.5 Kalshi API | kalshi_client.py | 539 | ✅ |
| 4.6 사후 분석 | post_match.py | 408 | ✅ |

### 오케스트레이터 (v1에 없던 신규 계층)

Phase 3 엔진과 Phase 4 실행을 연결하는 통합 계층.
설계 문서에 명시되지 않았으나, 실제 구현에서 필요했던 모듈들:

| 모듈 | 역할 | 줄 수 |
|------|------|-------|
| `ticker_mapper.py` | Kalshi 마켓 탐색 + 티커 매핑 (이벤트 API + fuzzy 매칭) | 519 |
| `tick_router.py` | 매 틱 Phase 4 파이프라인 라우팅 (entry/exit 평가) | 327 |
| `match_session.py` | 경기 라이프사이클 관리 (시작→진행→정산) | 286 |
| `orderbook_poller.py` | Kalshi 호가창 REST 폴링 | 273 |

체크리스트:
- [x] Fee-adjusted EV 양방향 탐색 (BUY_YES / BUY_NO)
- [x] Fractional Kelly (0.25) + 3-layer 리스크 한도
- [x] 3가지 청산 트리거 (edge 소멸, 역전, 만기 평가)
- [x] Paper 모드 주문 실행 + 포트폴리오 관리
- [x] Kalshi API RSA 인증 + 주문 제출
- [x] Brier Score 누적 + Edge 실현율 + 건강 대시보드
- [x] Kalshi 마켓 자동 탐색 (EPL, MLS, Bundesliga 등)
- [x] 호가창 폴링 + 합성 호가 생성
- [x] 전체 84/84 테스트 통과

---

## Stage 5: Paper Trading ◐ 진행 중

### 5.1 현재 상태

Paper Trading은 두 가지 모드로 검증 중:

| 모드 | 도구 | 데이터 | 호가 | 상태 |
|------|------|--------|------|------|
| **Replay** | `tests/test_replay_paper.py` | 과거 DB 이벤트 | 합성 호가 생성 | ✅ 완료 (15경기, +$522) |
| **Live** | `tools/paper_runner.py` | Goalserve 실시간 | Kalshi 실시간 호가 | ◐ 첫 테스트 완료 |

### 5.2 첫 라이브 테스트 결과 (2026-03-01)

**경기:** Manchester United vs Crystal Palace (EPL)
**결과:** MUN 2-1 CRY (27' CRY 선제, 56' CRY 퇴장, 57' MUN 동점, 65' MUN 역전)

**모델 반응 (정확했음):**

| 분 | 이벤트 | P_home |
|----|--------|--------|
| 27' | 접속 (0-1) | 0.164 |
| 56' | CRY 퇴장 | 0.099 → 0.355 |
| 57' | MUN 동점 | 0.355 |
| 65' | MUN 역전 | 0.825 |

**거래 결과 (참사):**

| 지표 | 값 |
|------|-----|
| 총 거래 | 1,994 (997 entry + 997 exit) |
| 총 손실 | -$11,655 |
| Win rate | 0% |
| Hold < 1초 | 98% |

### 5.3 발견된 버그 및 수정

#### 수정 완료 (2026-03-01)

| # | 버그 | 원인 | 수정 | 파일 |
|---|------|------|------|------|
| 0 | 로그에 contracts=0 | `TradeRecord`에 `contracts` 필드 없음 | `quantity_filled` 사용 | paper_runner.py |
| 1 | 진입 3초 후 즉시 청산 | ExitManager에 최소 보유 시간 없음 | `min_hold_ticks=50` (~150초) | exit_manager.py |
| 2 | 청산 후 즉시 재진입 | TickRouter에 재진입 쿨다운 없음 | `entry_cooldown_after_exit=100` (~5분) | tick_router.py |
| 3 | mu가 비정상 (121, 115) | `make_default_engine_params`에서 a_H를 선형 역산 | `a_H = ln(μ_H / C_time)` log-space 공식 | paper_runner.py |
| 4 | Mid-game join 시 0-0에서 시작 | GoalserveSource.initial_state 미적용 | `connect()` 시 initial_state 자동 생성 | goalserve.py, engine.py |
| 5 | 틱 간격 0.11초 (39,749 ticks/경기) | `asyncio.sleep(0)` + 0.1초 이벤트 대기 | `tick_interval=3.0` 추가 + `get_nowait()` | engine.py |

**예상 효과:** 997 trades → ~16 trades (98.4% 감소), $11,383 손실 회피

#### 미수정 (P0 — 다음 경기 전 반드시)

| # | 버그 | 심각도 | 설명 |
|---|------|--------|------|
| 6 | **시간이 진행되지 않음** | **🔴 CRITICAL** | `current_minute`이 이벤트 시에만 업데이트. Goalserve가 매 3초 timer를 보내지만, GoalserveSource._diff()가 TICK_UPDATE를 yield하지 않음. 90분 경기에서 unique minute = 8개뿐. **Theta decay = 0** |

**수정안:**
1. `EventType.TICK_UPDATE` 추가 (`event_source.py`)
2. GoalserveSource._diff()에서 매 폴링마다 timer 파싱 → minute 변경 시 TICK_UPDATE yield
3. EventHandler에 TICK_UPDATE 핸들러 추가 (`state.current_minute = event.minute`)

#### 미구현 (P1 — Paper Trading 안정화 후)

| # | 기능 | 심각도 | 설명 |
|---|------|--------|------|
| 7 | ob_freeze (호가 이상 감지) | 🟡 | 필드 존재하나 True 설정 코드 0줄 |
| 8 | Circuit breaker | 🟡 | Goalserve 연속 실패 시 주문 중단 미구현 |
| 9 | Phase 2 정식 연결 | 🟡 | paper_runner 하드코딩 탈피 |

#### 미구현 (P2 — Live 전환 전)

| # | 기능 | 심각도 | 설명 |
|---|------|--------|------|
| 10 | VWAP depth-aware 가격 | 🟢 | best bid/ask만 사용 중 |
| 11 | Stale MC 체크 | 🟢 | MC 0.5ms이므로 실질적 위험 낮음 |

### 5.4 Paper Trading 기간 목표

최소 **2주, 30경기 이상**을 Paper Trading으로 돌린다.
이 기간에 확인할 것:

- [ ] 시스템이 크래시 없이 경기 전체를 완주하는가?
- [ ] 시간 진행(theta decay)이 정상 작동하는가?
- [ ] churn 없이 합리적 거래 횟수 (경기당 5~20건)?
- [ ] P_true와 시장 가격의 괴리가 이벤트에 올바르게 반응하는가?
- [ ] 가상 P&L 추이가 합리적인가?

하나라도 실패하면 **Stage 3~4로 돌아간다.**

---

## Stage 6: 클라우드 배포 ⬚

v1 내용 유지. 추가 고려사항:

### 6.1 인프라 결정사항

| 항목 | v1 계획 | 현재 상태 |
|------|---------|----------|
| DB | PostgreSQL | SQLite → **PostgreSQL/TimescaleDB로 마이그레이션 필요** |
| 서버 | DigitalOcean $12/월 | 유지 (2 vCPU, 2GB RAM 충분) |
| 리전 | New York | 유지 (Kalshi + Goalserve 근접) |
| 모니터링 | — | Telegram 알림 추가 권장 |

### 6.2 추가 필요 작업

- [ ] SQLite → PostgreSQL 마이그레이션 스크립트
- [ ] Docker + docker-compose 설정
- [ ] Systemd 또는 Supervisor로 프로세스 관리
- [ ] Goalserve IP 화이트리스트에 서버 IP 등록
- [ ] 로그 로테이션 설정 (경기당 ~0.4MB)

---

## Stage 7: 실전 투입 + 모니터링 ⬚

v1 내용 대부분 유지. 실전 투입 체크리스트 업데이트:

### 7.1 실전 투입 체크리스트

- [ ] Stage 5 Paper Trading 30경기+ 완료
- [ ] 🔴 시간 진행 버그 수정 확인
- [ ] 🟡 ob_freeze, circuit breaker 구현 확인
- [ ] 🟡 Phase 2 정식 연결 (하드코딩 탈피)
- [ ] 클라우드에서 24시간 이상 무크래시 운영
- [ ] Circuit breaker 테스트 (의도적 Goalserve 차단으로 복구 확인)
- [ ] Kalshi 실계좌에 소액 입금 ($100~$500)

### 7.2 점진적 투입

| 단계 | 기간 | 자본 | Kelly |
|------|------|------|-------|
| 1단계 | 2주 | $100~$200 | 0.10 |
| 2단계 | 4주 | $500 | 0.25 |
| 3단계 | 8주+ | $1,000+ | 0.50 |

---

## 코드베이스 정량 요약

| 영역 | 파일 수 | 줄 수 |
|------|---------|-------|
| Phase 1 | 4 | 2,465 |
| Phase 2 | 1 | 1,125 |
| Phase 3 | 9 | 3,166 |
| Phase 4 | 6 | 2,529 |
| Orchestrator | 4 | 1,405 |
| Data | 2 | 1,903 |
| Tools | 6 | 1,791 |
| Tests | 19 | 6,043 |
| **합계** | **51** | **20,427** |

테스트: **84/84 (100%)**

---

## 핵심 도구 요약 (v3 갱신)

| 도구 | 용도 | 상태 |
|------|------|------|
| Python 3.11 | 모든 코드 | ✅ 사용 중 |
| Git/GitHub | 코드 버전 관리 | ✅ 사용 중 |
| SQLite | 개발 DB | ✅ 사용 중 |
| PyTorch | Phase 1.4 NLL | ✅ 사용 중 |
| XGBoost | Phase 1.3 ML | ✅ 사용 중 |
| Numba | Phase 3.4 MC | ✅ 사용 중 |
| asyncio + httpx | Phase 3 라이브 엔진 | ✅ 사용 중 |
| **Goalserve API** | **라이브 이벤트** | **✅ 사용 중 (v1: Sportradar)** |
| Docker | 클라우드 배포 | ⬚ Stage 6 |
| PostgreSQL | 프로덕션 DB | ⬚ Stage 6 |
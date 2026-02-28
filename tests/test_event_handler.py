"""
event_handler.py 단위 테스트

실행:
  python -m pytest tests/test_event_handler.py -v
  python tests/test_event_handler.py
"""

from src.phase3.event_source import EventType, NormalizedEvent
from src.phase3.event_handler import EventHandler, GameState


def make_event(etype, team="", minute=0.0, raw=None):
    """테스트용 이벤트 생성 헬퍼."""
    return NormalizedEvent(
        event_type=etype,
        team=team,
        minute=minute,
        raw=raw or {},
    )


def test_all():

    # ──────────────────────────────────────────────────
    # 테스트 1: 초기 GameState
    # ──────────────────────────────────────────────────
    state = GameState()
    assert state.S_H == 0 and state.S_A == 0
    assert state.delta_S == 0
    assert state.X == 0
    assert state.engine_phase == "PRE_MATCH"
    assert state.cooldown is False
    assert state.ob_freeze is False
    print("✅ T1: GameState 초기 상태 정상")

    # ──────────────────────────────────────────────────
    # 테스트 2: 홈 골 → S_H +1, ΔS +1, 쿨다운
    # ──────────────────────────────────────────────────
    handler = EventHandler(cooldown_seconds=15.0)
    state = GameState(engine_phase="FIRST_HALF")
    event = make_event(EventType.GOAL, team="home", minute=23.0)
    handler.handle(event, state)

    assert state.S_H == 1
    assert state.S_A == 0
    assert state.delta_S == 1
    assert state.cooldown is True
    assert state.current_minute == 23.0
    print("✅ T2: 홈 골 → 1-0, ΔS=1, 쿨다운 활성")

    # ──────────────────────────────────────────────────
    # 테스트 3: 어웨이 골 → S_A +1, ΔS 0
    # ──────────────────────────────────────────────────
    event2 = make_event(EventType.GOAL, team="away", minute=35.0)
    handler.handle(event2, state)

    assert state.S_H == 1
    assert state.S_A == 1
    assert state.delta_S == 0
    print("✅ T3: 어웨이 골 → 1-1, ΔS=0")

    # ──────────────────────────────────────────────────
    # 테스트 4: 레드카드 — 홈 퇴장 (X: 0→1)
    # ──────────────────────────────────────────────────
    state4 = GameState(engine_phase="FIRST_HALF")
    event4 = make_event(EventType.RED_CARD, team="home", minute=40.0)
    handler.handle(event4, state4)

    assert state4.X == 1   # 11v11 → 10v11
    assert state4.cooldown is True
    print("✅ T4: 홈 퇴장 → X=1 (10v11)")

    # ──────────────────────────────────────────────────
    # 테스트 5: 레드카드 — 어웨이 퇴장 (X: 0→2)
    # ──────────────────────────────────────────────────
    state5 = GameState(engine_phase="FIRST_HALF")
    event5 = make_event(EventType.RED_CARD, team="away", minute=55.0)
    handler.handle(event5, state5)

    assert state5.X == 2   # 11v11 → 11v10
    print("✅ T5: 어웨이 퇴장 → X=2 (11v10)")

    # ──────────────────────────────────────────────────
    # 테스트 6: 레드카드 연속 — 양쪽 퇴장 (X: 1→3)
    # ──────────────────────────────────────────────────
    state6 = GameState(X=1, engine_phase="SECOND_HALF")  # 이미 홈 퇴장
    event6 = make_event(EventType.RED_CARD, team="away", minute=70.0)
    handler.handle(event6, state6)

    assert state6.X == 3   # 10v11 → 10v10
    print("✅ T6: 양쪽 퇴장 → X=3 (10v10)")

    # ──────────────────────────────────────────────────
    # 테스트 7: 레드카드 — X=2에서 홈 퇴장 (2→3)
    # ──────────────────────────────────────────────────
    state7 = GameState(X=2, engine_phase="SECOND_HALF")
    event7 = make_event(EventType.RED_CARD, team="home", minute=75.0)
    handler.handle(event7, state7)

    assert state7.X == 3
    print("✅ T7: X=2에서 홈 퇴장 → X=3")

    # ──────────────────────────────────────────────────
    # 테스트 8: 하프타임 → engine_phase = "HALFTIME"
    # ──────────────────────────────────────────────────
    state8 = GameState(engine_phase="FIRST_HALF", S_H=1, S_A=0)
    event8 = make_event(EventType.HALFTIME, minute=47.0)
    handler.handle(event8, state8)

    assert state8.engine_phase == "HALFTIME"
    print("✅ T8: 하프타임 → HALFTIME")

    # ──────────────────────────────────────────────────
    # 테스트 9: 후반 시작 → engine_phase = "SECOND_HALF"
    # ──────────────────────────────────────────────────
    event9 = make_event(EventType.SECOND_HALF_START, minute=47.1)
    handler.handle(event9, state8)

    assert state8.engine_phase == "SECOND_HALF"
    print("✅ T9: 후반 시작 → SECOND_HALF")

    # ──────────────────────────────────────────────────
    # 테스트 10: 경기 종료 → FINISHED
    # ──────────────────────────────────────────────────
    state10 = GameState(engine_phase="SECOND_HALF", S_H=2, S_A=1)
    event10 = make_event(EventType.MATCH_END, minute=95.0,
                         raw={"final_score": "2-1"})
    handler.handle(event10, state10)

    assert state10.engine_phase == "FINISHED"
    print("✅ T10: 경기 종료 → FINISHED")

    # ──────────────────────────────────────────────────
    # 테스트 11: 추가시간 진입 — 상태 변경 없이 로그만
    # ──────────────────────────────────────────────────
    state11 = GameState(engine_phase="SECOND_HALF")
    event11 = make_event(EventType.STOPPAGE_ENTERED, minute=90.5,
                         raw={"half": "second"})
    handler.handle(event11, state11)

    # engine_phase 변경 없음
    assert state11.engine_phase == "SECOND_HALF"
    print("✅ T11: 추가시간 진입 — engine_phase 불변")

    # ──────────────────────────────────────────────────
    # 테스트 12: orders_allowed 로직
    # ──────────────────────────────────────────────────
    # 정상 상태 → 주문 가능
    state12 = GameState(engine_phase="FIRST_HALF")
    assert state12.orders_allowed is True

    # 쿨다운 → 주문 불가
    state12.cooldown = True
    state12.cooldown_until = float('inf')
    assert state12.orders_allowed is False

    # ob_freeze → 주문 불가
    state12.cooldown = False
    state12.ob_freeze = True
    assert state12.orders_allowed is False

    # HALFTIME → 주문 불가
    state12_ht = GameState(engine_phase="HALFTIME")
    assert state12_ht.orders_allowed is False

    # FINISHED → 주문 불가
    state12_fin = GameState(engine_phase="FINISHED")
    assert state12_fin.orders_allowed is False
    print("✅ T12: orders_allowed — 쿨다운/ob_freeze/HALFTIME 차단")

    # ──────────────────────────────────────────────────
    # 테스트 13: 이벤트 로그
    # ──────────────────────────────────────────────────
    h = EventHandler()
    s = GameState(engine_phase="FIRST_HALF")
    h.handle(make_event(EventType.GOAL, "home", 10.0), s)
    h.handle(make_event(EventType.RED_CARD, "away", 20.0), s)
    h.handle(make_event(EventType.HALFTIME, minute=47.0), s)
    assert h.event_count == 3
    log = h.get_event_log()
    assert log[0].event_type == EventType.GOAL
    assert log[1].event_type == EventType.RED_CARD
    assert log[2].event_type == EventType.HALFTIME
    print("✅ T13: 이벤트 로그 — 3개 기록")

    # ──────────────────────────────────────────────────
    # 테스트 14: 전체 경기 시뮬레이션
    # ──────────────────────────────────────────────────
    handler = EventHandler(cooldown_seconds=0.001)  # 테스트용 짧은 쿨다운
    state = GameState(engine_phase="FIRST_HALF")

    events = [
        make_event(EventType.GOAL, "home", 23.0),          # 1-0
        make_event(EventType.RED_CARD, "away", 40.0),      # X: 0→2
        make_event(EventType.STOPPAGE_ENTERED, "", 45.5,
                   raw={"half": "first"}),
        make_event(EventType.HALFTIME, "", 47.0),
        make_event(EventType.SECOND_HALF_START, "", 47.1),
        make_event(EventType.GOAL, "away", 55.0),          # 1-1
        make_event(EventType.GOAL, "home", 78.0),          # 2-1
        make_event(EventType.STOPPAGE_ENTERED, "", 90.5,
                   raw={"half": "second"}),
        make_event(EventType.GOAL, "home", 93.0),          # 3-1
        make_event(EventType.MATCH_END, "", 95.0),
    ]

    for e in events:
        handler.handle(e, state)

    assert state.S_H == 3
    assert state.S_A == 1
    assert state.delta_S == 2
    assert state.X == 2       # 어웨이만 퇴장
    assert state.engine_phase == "FINISHED"
    assert handler.event_count == 10
    print(f"✅ T14: 전체 경기 — 최종 {state.S_H}-{state.S_A}, X={state.X}, FINISHED")

    # ──────────────────────────────────────────────────
    print()
    print("═" * 50)
    print("  ALL 14 TESTS PASSED ✅")
    print("═" * 50)


if __name__ == "__main__":
    test_all()
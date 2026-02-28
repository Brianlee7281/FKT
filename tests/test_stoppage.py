"""
stoppage.py 단위 테스트

실행:
  python -m pytest tests/test_stoppage.py -v
  python tests/test_stoppage.py
"""

from src.phase3.stoppage import StoppageTimeManager


def test_all():

    # ──────────────────────────────────────────────────
    # 테스트 1: 초기 상태
    # ──────────────────────────────────────────────────
    mgr = StoppageTimeManager(T_exp=95.0)
    assert mgr.phase == "A"
    assert mgr.T == 95.0
    assert mgr.is_stoppage_time is False
    print("✅ T1: 초기 상태 — Phase A, T=95.0")

    # ──────────────────────────────────────────────────
    # 테스트 2: 전반 정규 시간 — T 불변
    # ──────────────────────────────────────────────────
    mgr = StoppageTimeManager(T_exp=95.0)
    for t in [0.0, 10.0, 30.0, 44.0]:
        T = mgr.update(t, "FIRST_HALF")
        assert T == 95.0
    assert mgr.phase == "A"
    print("✅ T2: 전반 정규 시간 — T=95.0 유지")

    # ──────────────────────────────────────────────────
    # 테스트 3: 전반 추가시간 감지 — T는 불변 (전반≠경기종료)
    # ──────────────────────────────────────────────────
    mgr = StoppageTimeManager(T_exp=95.0)
    T = mgr.update(46.0, "FIRST_HALF")
    assert T == 95.0  # 전반 추가시간이어도 T(경기종료)는 변하지 않음
    assert mgr.first_half_stoppage_entered is True
    print("✅ T3: 전반 추가시간 감지 — T 불변, 마킹만")

    # ──────────────────────────────────────────────────
    # 테스트 4: 하프타임 — T 불변
    # ──────────────────────────────────────────────────
    mgr = StoppageTimeManager(T_exp=95.0)
    T = mgr.update(47.0, "HALFTIME")
    assert T == 95.0
    print("✅ T4: 하프타임 — T 불변")

    # ──────────────────────────────────────────────────
    # 테스트 5: 후반 정규 시간 (Phase A) — T 불변
    # ──────────────────────────────────────────────────
    mgr = StoppageTimeManager(T_exp=95.0)
    mgr.reset_for_second_half()
    for t in [47.0, 60.0, 75.0, 89.0]:
        T = mgr.update(t, "SECOND_HALF")
        assert T == 95.0
    assert mgr.phase == "A"
    print("✅ T5: 후반 정규 시간 — Phase A, T=95.0")

    # ──────────────────────────────────────────────────
    # 테스트 6: 후반 추가시간 진입 → Phase C
    # ──────────────────────────────────────────────────
    mgr = StoppageTimeManager(T_exp=95.0, rolling_horizon=1.5)
    mgr.reset_for_second_half()
    T = mgr.update(91.0, "SECOND_HALF")
    assert mgr.phase == "C"
    assert mgr.second_half_stoppage_entered is True
    assert T == 92.5  # 91.0 + 1.5
    assert mgr.is_stoppage_time is True
    print("✅ T6: 후반 91분 → Phase C, T=92.5")

    # ──────────────────────────────────────────────────
    # 테스트 7: Phase C — 롤링 업데이트
    # ──────────────────────────────────────────────────
    T = mgr.update(92.0, "SECOND_HALF")
    assert T == 93.5   # 92.0 + 1.5
    T = mgr.update(93.0, "SECOND_HALF")
    assert T == 94.5   # 93.0 + 1.5
    T = mgr.update(95.0, "SECOND_HALF")
    assert T == 96.5   # 95.0 + 1.5
    print("✅ T7: Phase C 롤링 — T = t + 1.5")

    # ──────────────────────────────────────────────────
    # 테스트 8: Phase C — T 단조 증가 (시간이 되돌아가지 않음)
    # ──────────────────────────────────────────────────
    T_before = mgr.T
    T = mgr.update(93.0, "SECOND_HALF")  # 시간이 뒤로 감 (비정상)
    assert T >= T_before  # T는 줄어들지 않음
    print("✅ T8: T 단조 증가 보장")

    # ──────────────────────────────────────────────────
    # 테스트 9: on_stoppage_entered — 이벤트 기반 전환
    # ──────────────────────────────────────────────────
    mgr2 = StoppageTimeManager(T_exp=95.0)
    mgr2.on_stoppage_entered("first")
    assert mgr2.first_half_stoppage_entered is True
    assert mgr2.phase == "A"  # 전반 추가시간은 Phase 전환 안 함

    mgr2.on_stoppage_entered("second")
    assert mgr2.second_half_stoppage_entered is True
    assert mgr2.phase == "C"  # 후반 추가시간은 Phase C 전환
    print("✅ T9: on_stoppage_entered — 이벤트 기반 전환")

    # ──────────────────────────────────────────────────
    # 테스트 10: 중복 이벤트 무시
    # ──────────────────────────────────────────────────
    mgr3 = StoppageTimeManager(T_exp=95.0)
    mgr3.on_stoppage_entered("second")
    assert mgr3.phase == "C"
    mgr3.on_stoppage_entered("second")  # 두 번째 호출
    assert mgr3.phase == "C"  # 변화 없음
    print("✅ T10: 중복 on_stoppage_entered 무시")

    # ──────────────────────────────────────────────────
    # 테스트 11: reset_for_second_half
    # ──────────────────────────────────────────────────
    mgr4 = StoppageTimeManager(T_exp=95.0)
    mgr4.first_half_stoppage_entered = True
    mgr4.reset_for_second_half()
    assert mgr4.phase == "A"
    assert mgr4.T == 95.0
    print("✅ T11: reset_for_second_half → Phase A 복귀")

    # ──────────────────────────────────────────────────
    # 테스트 12: 커스텀 rolling_horizon
    # ──────────────────────────────────────────────────
    mgr5 = StoppageTimeManager(T_exp=95.0, rolling_horizon=2.0)
    T = mgr5.update(91.0, "SECOND_HALF")
    assert T == 93.0   # 91.0 + 2.0
    print("✅ T12: rolling_horizon=2.0 → T=93.0")

    # ──────────────────────────────────────────────────
    # 테스트 13: __str__
    # ──────────────────────────────────────────────────
    s = str(mgr5)
    assert "phase=C" in s
    assert "T=93.0" in s
    print("✅ T13: __str__ 정상 출력")

    # ──────────────────────────────────────────────────
    # 테스트 14: 전체 경기 시뮬레이션
    # ──────────────────────────────────────────────────
    mgr6 = StoppageTimeManager(T_exp=95.0, rolling_horizon=1.5)

    # 전반
    for t in range(0, 46):
        mgr6.update(float(t), "FIRST_HALF")
    assert mgr6.phase == "A"

    # 전반 추가시간
    mgr6.update(46.0, "FIRST_HALF")
    assert mgr6.first_half_stoppage_entered is True

    # 하프타임
    mgr6.update(47.0, "HALFTIME")
    mgr6.reset_for_second_half()

    # 후반
    for t in range(47, 91):
        mgr6.update(float(t), "SECOND_HALF")
    assert mgr6.phase == "A"

    # 후반 추가시간
    T = mgr6.update(91.0, "SECOND_HALF")
    assert mgr6.phase == "C"
    assert T == 92.5

    T = mgr6.update(93.0, "SECOND_HALF")
    assert T == 94.5

    T = mgr6.update(95.0, "SECOND_HALF")
    assert T == 96.5

    print("✅ T14: 전체 경기 시뮬레이션 통과")

    # ──────────────────────────────────────────────────
    print()
    print("═" * 50)
    print("  ALL 14 TESTS PASSED ✅")
    print("═" * 50)


if __name__ == "__main__":
    test_all()
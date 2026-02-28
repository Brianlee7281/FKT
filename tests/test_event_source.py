"""
event_source.py 단위 테스트

실행:
  pytest tests/test_event_source.py -v
"""

import time
import pytest
from src.phase3.event_source import (
    EventType,
    NormalizedEvent,
    parse_minute,
)


# ═════════════════════════════════════════════════════════
# EventType 열거형
# ═════════════════════════════════════════════════════════

class TestEventType:
    def test_string_values(self):
        """EventType의 값이 문자열이고 엔진이 기대하는 값과 일치"""
        assert EventType.GOAL == "goal"
        assert EventType.RED_CARD == "red_card"
        assert EventType.HALFTIME == "halftime"
        assert EventType.SECOND_HALF_START == "second_half_start"
        assert EventType.STOPPAGE_ENTERED == "stoppage_entered"
        assert EventType.MATCH_END == "match_end"

    def test_total_count(self):
        """Phase 3 설계 문서의 6개 이벤트 유형과 일치"""
        assert len(EventType) == 6


# ═════════════════════════════════════════════════════════
# NormalizedEvent 생성 + 유효성 검증
# ═════════════════════════════════════════════════════════

class TestNormalizedEvent:

    # ─── 정상 생성 ───

    def test_goal_event(self):
        """홈팀 골 이벤트 정상 생성"""
        evt = NormalizedEvent(
            event_type=EventType.GOAL,
            team="home",
            minute=72.0,
            raw={"scorer": "Haaland"},
        )
        assert evt.event_type == EventType.GOAL
        assert evt.team == "home"
        assert evt.minute == 72.0
        assert evt.raw["scorer"] == "Haaland"
        assert evt.timestamp > 0  # default_factory=time.time

    def test_red_card_event(self):
        """어웨이팀 퇴장 이벤트 정상 생성"""
        evt = NormalizedEvent(
            event_type=EventType.RED_CARD,
            team="away",
            minute=55.0,
            raw={"player": "Casemiro", "reason": "Second Yellow card"},
        )
        assert evt.event_type == EventType.RED_CARD
        assert evt.team == "away"

    def test_halftime_event(self):
        """하프타임 이벤트 (team 빈 문자열)"""
        evt = NormalizedEvent(
            event_type=EventType.HALFTIME,
            team="",
            minute=47.0,
        )
        assert evt.team == ""
        assert evt.minute == 47.0

    def test_stoppage_entered(self):
        """추가시간 진입 이벤트"""
        evt = NormalizedEvent(
            event_type=EventType.STOPPAGE_ENTERED,
            team="",
            minute=91.0,
            raw={"half": "second"},
        )
        assert evt.event_type == EventType.STOPPAGE_ENTERED

    def test_match_end(self):
        """경기 종료 이벤트"""
        evt = NormalizedEvent(
            event_type=EventType.MATCH_END,
            team="",
            minute=95.0,
        )
        assert evt.event_type == EventType.MATCH_END

    # ─── 유효성 검증 (실패 케이스) ───

    def test_goal_requires_team(self):
        """골 이벤트에 team이 없으면 ValueError"""
        with pytest.raises(ValueError, match="'home' 또는 'away'"):
            NormalizedEvent(
                event_type=EventType.GOAL,
                team="",
                minute=30.0,
            )

    def test_red_card_requires_team(self):
        """퇴장 이벤트에 team이 없으면 ValueError"""
        with pytest.raises(ValueError, match="'home' 또는 'away'"):
            NormalizedEvent(
                event_type=EventType.RED_CARD,
                team="neither",
                minute=30.0,
            )

    def test_negative_minute_rejected(self):
        """음수 minute 거부"""
        with pytest.raises(ValueError, match="minute 범위 초과"):
            NormalizedEvent(
                event_type=EventType.HALFTIME,
                team="",
                minute=-1.0,
            )

    def test_extreme_minute_rejected(self):
        """200분 초과 거부"""
        with pytest.raises(ValueError, match="minute 범위 초과"):
            NormalizedEvent(
                event_type=EventType.HALFTIME,
                team="",
                minute=201.0,
            )

    # ─── frozen 검증 ───

    def test_frozen(self):
        """이벤트는 불변 객체"""
        evt = NormalizedEvent(
            event_type=EventType.GOAL,
            team="home",
            minute=10.0,
        )
        with pytest.raises(AttributeError):
            evt.minute = 20.0

    # ─── 프로퍼티 ───

    def test_is_discrete_shock(self):
        """골/퇴장만 불연속 충격"""
        goal = NormalizedEvent(EventType.GOAL, "home", 10.0)
        red = NormalizedEvent(EventType.RED_CARD, "away", 20.0)
        ht = NormalizedEvent(EventType.HALFTIME, "", 45.0)
        end = NormalizedEvent(EventType.MATCH_END, "", 95.0)

        assert goal.is_discrete_shock is True
        assert red.is_discrete_shock is True
        assert ht.is_discrete_shock is False
        assert end.is_discrete_shock is False

    def test_triggers_cooldown(self):
        """쿨다운 활성화 대상 이벤트"""
        goal = NormalizedEvent(EventType.GOAL, "home", 10.0)
        red = NormalizedEvent(EventType.RED_CARD, "away", 20.0)
        stoppage = NormalizedEvent(EventType.STOPPAGE_ENTERED, "", 91.0)
        ht = NormalizedEvent(EventType.HALFTIME, "", 45.0)
        end = NormalizedEvent(EventType.MATCH_END, "", 95.0)

        assert goal.triggers_cooldown is True
        assert red.triggers_cooldown is True
        assert stoppage.triggers_cooldown is True
        assert ht.triggers_cooldown is False     # 이미 주문 동결
        assert end.triggers_cooldown is False    # 경기 종료

    # ─── __str__ ───

    def test_str_with_team(self):
        evt = NormalizedEvent(EventType.GOAL, "home", 72.0)
        assert "goal" in str(evt)
        assert "home" in str(evt)
        assert "72.0" in str(evt)

    def test_str_without_team(self):
        evt = NormalizedEvent(EventType.HALFTIME, "", 47.0)
        assert "halftime" in str(evt)
        assert "47.0" in str(evt)


# ═════════════════════════════════════════════════════════
# parse_minute 유틸리티
# ═════════════════════════════════════════════════════════

class TestParseMinute:
    def test_normal_minute(self):
        assert parse_minute("72") == 72.0

    def test_stoppage_time(self):
        assert parse_minute("90+3") == 93.0

    def test_first_half_stoppage(self):
        assert parse_minute("45+2") == 47.0

    def test_integer_input(self):
        """정수가 str()로 변환되어 처리"""
        assert parse_minute(72) == 72.0

    def test_float_string(self):
        assert parse_minute("72.5") == 72.5

    def test_invalid_returns_zero(self):
        assert parse_minute("abc") == 0.0

    def test_empty_string(self):
        assert parse_minute("") == 0.0

    def test_malformed_plus(self):
        """'90+' 같은 불완전한 문자열"""
        assert parse_minute("90+") == 90.0

    def test_whitespace(self):
        """양쪽 공백 제거"""
        assert parse_minute("  90+3  ") == 93.0
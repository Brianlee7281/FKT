"""
goalserve.py 단위 테스트

실제 Goalserve 라이브 JSON 스냅샷을 사용하여 파싱 + diff 로직을 검증한다.
네트워크 호출 없이 테스트 가능.

실행:
  python -m pytest tests/test_goalserve.py -v
  python tests/test_goalserve.py
"""

import json
import asyncio
from typing import Dict, List

from src.phase3.event_source import EventType, NormalizedEvent
from src.phase3.goalserve import GoalserveSource


# ═══════════════════════════════════════════════════
# 테스트 헬퍼: 실제 JSON 스냅샷 로드
# ═══════════════════════════════════════════════════

SAMPLE_MATCH = {
    "@status": "85",
    "@timer": "85",
    "@date": "Feb 28",
    "@formatted_date": "28.02.2026",
    "@time": "15:00",
    "@venue": "Turf Moor",
    "@id": "6678380",
    "localteam": {"@name": "Burnley", "@goals": "3", "@id": "9072"},
    "visitorteam": {"@name": "Brentford", "@goals": "3", "@id": "9059"},
    "events": {
        "event": [
            {
                "@type": "goal", "@minute": "9", "@extra_min": "",
                "@team": "visitorteam", "@player": "M. Damsgaard",
                "@result": "[0 - 1]", "@eventid": "66783801",
                "@playerId": "525402", "@assist": "D. Ouattara",
                "@assistid": "675906", "@ts": "1772291414",
            },
            {
                "@type": "goal", "@minute": "25", "@extra_min": "",
                "@team": "visitorteam", "@player": "I. Thiago",
                "@result": "[0 - 2]", "@eventid": "66783802",
                "@playerId": "649618", "@assist": "M. Damsgaard",
                "@assistid": "525402", "@ts": "1772292367",
            },
            {
                "@type": "goal", "@minute": "34", "@extra_min": "",
                "@team": "visitorteam", "@player": "K. Schade",
                "@result": "[0 - 3]", "@eventid": "66783803",
                "@playerId": "562536", "@assist": "",
                "@assistid": "", "@ts": "1772292901",
            },
            {
                "@type": "goal", "@minute": "45", "@extra_min": "3",
                "@team": "localteam", "@player": "M. Kayode (o.g.)",
                "@result": "[1 - 3]", "@eventid": "66783804",
                "@playerId": "796522", "@assist": "",
                "@assistid": "", "@ts": "1772293754",
            },
            {
                "@type": "yellowcard", "@minute": "77", "@extra_min": "",
                "@team": "visitorteam", "@player": "I. Thiago",
                "@result": "", "@eventid": "667838013",
                "@playerId": "649618", "@assist": "Roughing",
                "@assistid": "", "@ts": "1772296722",
            },
        ]
    },
    "ht": {"@score": "[1-3]"},
}

SAMPLE_RED_CARD_EVENT = {
    "@type": "redcard", "@minute": "14", "@extra_min": "",
    "@team": "visitorteam", "@player": "D. Love",
    "@result": "", "@eventid": "67427672",
    "@playerId": "417910", "@assist": "Unsportsmanlike conduct",
    "@assistid": "", "@ts": "1772291652",
}

SAMPLE_YELLOW_RED_EVENT = {
    "@type": "yellowred", "@minute": "60", "@extra_min": "",
    "@team": "localteam", "@player": "Test Player",
    "@result": "", "@eventid": "99999999",
    "@playerId": "999", "@assist": "",
    "@assistid": "", "@ts": "0",
}


def collect_events(source, match_data):
    """동기적으로 diff 이벤트를 수집."""
    events = []
    async def _run():
        async for evt in source._diff(match_data):
            events.append(evt)
    asyncio.run(_run())
    return events


def test_all():

    # ──────────────────────────────────────────────────
    # 테스트 1: _find_match — 경기 찾기
    # ──────────────────────────────────────────────────
    mock_response = {
        "scores": {
            "category": [
                {
                    "@name": "England: Premier League",
                    "matches": {"match": [SAMPLE_MATCH]},
                }
            ]
        }
    }
    found = GoalserveSource._find_match(mock_response, "6678380")
    assert found is not None
    assert found["localteam"]["@name"] == "Burnley"

    not_found = GoalserveSource._find_match(mock_response, "9999999")
    assert not_found is None
    print("✅ T1: _find_match — 경기 찾기/못찾기")

    # ──────────────────────────────────────────────────
    # 테스트 2: _parse_event — 골 파싱
    # ──────────────────────────────────────────────────
    source = GoalserveSource(api_key="test")
    evt = source._parse_event(SAMPLE_MATCH["events"]["event"][0])
    assert evt is not None
    assert evt.event_type == EventType.GOAL
    assert evt.team == "away"  # visitorteam → away
    assert evt.minute == 9.0
    assert evt.raw["player"] == "M. Damsgaard"
    print("✅ T2: 골 파싱 — away, 9분")

    # ──────────────────────────────────────────────────
    # 테스트 3: _parse_event — 추가시간 골 파싱
    # ──────────────────────────────────────────────────
    evt_stoppage = source._parse_event(SAMPLE_MATCH["events"]["event"][3])
    assert evt_stoppage is not None
    assert evt_stoppage.event_type == EventType.GOAL
    assert evt_stoppage.team == "home"  # localteam → home
    assert evt_stoppage.minute == 48.0   # 45 + 3 = 48 (실효 시간)
    assert evt_stoppage.raw["own_goal"] is True  # "(o.g.)"
    assert evt_stoppage.raw["display_minute"] == "45+3"
    print("✅ T3: 추가시간 골 파싱 — home, 45+3분 (실효 48분), 자책골")

    # ──────────────────────────────────────────────────
    # 테스트 4: _parse_event — 레드카드
    # ──────────────────────────────────────────────────
    evt_red = source._parse_event(SAMPLE_RED_CARD_EVENT)
    assert evt_red is not None
    assert evt_red.event_type == EventType.RED_CARD
    assert evt_red.team == "away"
    assert evt_red.minute == 14.0
    print("✅ T4: 레드카드 파싱 — away, 14분")

    # ──────────────────────────────────────────────────
    # 테스트 5: _parse_event — yellowred (2번째 옐로)
    # ──────────────────────────────────────────────────
    evt_yr = source._parse_event(SAMPLE_YELLOW_RED_EVENT)
    assert evt_yr is not None
    assert evt_yr.event_type == EventType.RED_CARD  # yellowred도 RED_CARD
    assert evt_yr.team == "home"
    print("✅ T5: yellowred → RED_CARD 변환")

    # ──────────────────────────────────────────────────
    # 테스트 6: _parse_event — 옐로카드/교체/VAR → None
    # ──────────────────────────────────────────────────
    evt_yc = source._parse_event(SAMPLE_MATCH["events"]["event"][4])  # yellowcard
    assert evt_yc is None
    print("✅ T6: 옐로카드 → None (무시)")

    # ──────────────────────────────────────────────────
    # 테스트 7: diff — 새 이벤트 감지
    # ──────────────────────────────────────────────────
    src7 = GoalserveSource(api_key="test")
    src7._prev_status = "80"  # 이전 폴링은 80분

    # 기존 이벤트 4개는 이미 본 것으로 마킹
    for evt_data in SAMPLE_MATCH["events"]["event"][:4]:
        src7._seen_event_ids.add(evt_data["@eventid"])
    src7._halftime_sent = True
    src7._second_half_sent = True
    src7._stoppage_entered["first"] = True

    # 5번째 이벤트(옐로카드)만 새로운 건데, 무시되는 타입
    events = collect_events(src7, SAMPLE_MATCH)
    goal_events = [e for e in events if e.event_type == EventType.GOAL]
    assert len(goal_events) == 0  # 새 골 없음
    print("✅ T7: diff — 기존 이벤트 무시, 새 옐로카드 무시")

    # ──────────────────────────────────────────────────
    # 테스트 8: diff — 새 골 감지
    # ──────────────────────────────────────────────────
    src8 = GoalserveSource(api_key="test")
    src8._prev_status = "80"
    src8._halftime_sent = True
    src8._second_half_sent = True
    src8._stoppage_entered["first"] = True

    # 처음 3개만 본 것으로 마킹 → 4,5번째가 새로운 이벤트
    for evt_data in SAMPLE_MATCH["events"]["event"][:3]:
        src8._seen_event_ids.add(evt_data["@eventid"])

    events = collect_events(src8, SAMPLE_MATCH)
    goal_events = [e for e in events if e.event_type == EventType.GOAL]
    assert len(goal_events) == 1  # 45+3 골만 새로 감지
    assert goal_events[0].team == "home"
    print("✅ T8: diff — 새 골 1개 감지 (45+3 자책골)")

    # ──────────────────────────────────────────────────
    # 테스트 9: diff — 하프타임 감지
    # ──────────────────────────────────────────────────
    src9 = GoalserveSource(api_key="test")
    src9._prev_status = "45"
    src9._stoppage_entered["first"] = True

    ht_match = {**SAMPLE_MATCH, "@status": "HT", "@timer": "45"}
    events = collect_events(src9, ht_match)
    ht_events = [e for e in events if e.event_type == EventType.HALFTIME]
    assert len(ht_events) == 1
    assert src9._halftime_sent is True
    print("✅ T9: HT 상태 → HALFTIME 이벤트 생성")

    # ──────────────────────────────────────────────────
    # 테스트 10: diff — 후반 시작 감지
    # ──────────────────────────────────────────────────
    src10 = GoalserveSource(api_key="test")
    src10._prev_status = "HT"
    src10._halftime_sent = True
    src10._stoppage_entered["first"] = True

    sh_match = {**SAMPLE_MATCH, "@status": "46", "@timer": "46"}
    events = collect_events(src10, sh_match)
    sh_events = [e for e in events if e.event_type == EventType.SECOND_HALF_START]
    assert len(sh_events) == 1
    assert src10._second_half_sent is True
    print("✅ T10: HT → 46분 → SECOND_HALF_START 이벤트")

    # ──────────────────────────────────────────────────
    # 테스트 11: diff — 경기 종료 감지
    # ──────────────────────────────────────────────────
    src11 = GoalserveSource(api_key="test")
    src11._prev_status = "90"
    src11._halftime_sent = True
    src11._second_half_sent = True
    src11._stoppage_entered["first"] = True
    src11._stoppage_entered["second"] = True

    ft_match = {**SAMPLE_MATCH, "@status": "FT", "@timer": "90"}
    events = collect_events(src11, ft_match)
    end_events = [e for e in events if e.event_type == EventType.MATCH_END]
    assert len(end_events) == 1
    assert end_events[0].raw["final_score"] == "3-3"
    assert src11._match_end_sent is True
    print("✅ T11: FT 상태 → MATCH_END (3-3)")

    # ──────────────────────────────────────────────────
    # 테스트 12: diff — 후반 추가시간 진입 감지
    # ──────────────────────────────────────────────────
    src12 = GoalserveSource(api_key="test")
    src12._prev_status = "89"
    src12._halftime_sent = True
    src12._second_half_sent = True
    src12._stoppage_entered["first"] = True

    st_match = {**SAMPLE_MATCH, "@status": "91", "@timer": "91"}
    events = collect_events(src12, st_match)
    st_events = [e for e in events if e.event_type == EventType.STOPPAGE_ENTERED]
    assert len(st_events) == 1
    assert st_events[0].raw["half"] == "second"
    print("✅ T12: 91분 → STOPPAGE_ENTERED (second)")

    # ──────────────────────────────────────────────────
    # 테스트 13: _seed_existing_events — 중복 방지
    # ──────────────────────────────────────────────────
    src13 = GoalserveSource(api_key="test")
    src13._seed_existing_events(SAMPLE_MATCH)
    assert len(src13._seen_event_ids) == 5  # 5개 이벤트 전부 마킹
    assert src13._prev_status == "85"
    assert src13._prev_home_goals == 3
    assert src13._halftime_sent is True  # timer > 45
    print(f"✅ T13: seed — {len(src13._seen_event_ids)}개 기존 이벤트 마킹")

    # ──────────────────────────────────────────────────
    # 테스트 14: find_all_live_matches
    # ──────────────────────────────────────────────────
    mock_full = {
        "scores": {
            "category": [
                {
                    "@name": "England: Premier League",
                    "matches": {
                        "match": [
                            {**SAMPLE_MATCH},
                            {**SAMPLE_MATCH, "@id": "6678723", "@status": "FT",
                             "localteam": {"@name": "Liverpool", "@goals": "5", "@id": "9249"},
                             "visitorteam": {"@name": "West Ham", "@goals": "2", "@id": "9427"}},
                        ]
                    },
                }
            ]
        }
    }
    live = GoalserveSource.find_all_live_matches(mock_full)
    # FT인 Liverpool 경기는 제외
    assert len(live) == 1
    assert live[0]["home"] == "Burnley"
    assert live[0]["status"] == "85"
    print(f"✅ T14: find_all_live_matches — {len(live)}개 라이브 (FT 제외)")

    # ──────────────────────────────────────────────────
    print()
    print("═" * 50)
    print("  ALL 14 TESTS PASSED ✅")
    print("═" * 50)


if __name__ == "__main__":
    test_all()
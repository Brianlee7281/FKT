"""
오케스트레이터 — 티커 매핑 (ticker_mapper.py)

Goalserve 경기 데이터 ↔ Kalshi 마켓 티커 매핑.

Kalshi 축구 단일 경기 마켓 구조:
  - 시리즈: KXEPLGAME, KXSERIEAGAME, KXUCLGAME, KXLALIGAGAME,
            KXBUNDESLIGAGAME, KXLIGUE1GAME, KXLIGAMXGAME,
            KXEREDIVISIEGAME, KXSUPERLIGGAME, KXARGENTINAGAME
  - 경기당 3개 마켓: 홈승 / 원정승 / 무승부
  - 티커 패턴: {SERIES}-{DATE}{HOME}{AWAY}-{OUTCOME}
    예: KXEPLGAME-26MAR15ARSEVE-ARS (Arsenal 승)

매핑 방식:
  Phase 1 (수동):  config 파일로 관리 (경기 전날 수동 작성)
  Phase 2 (자동):  Kalshi API에서 당일 마켓 조회 → 팀명 fuzzy match

사용법:
  mapper = TickerMapper()
  mapper.register_match(
      match_id="GS_12345",
      home_team="Arsenal", away_team="Everton",
      kalshi_tickers={
          "home_win": "KXEPLGAME-26MAR15ARSEVE-ARS",
          "away_win": "KXEPLGAME-26MAR15ARSEVE-EVE",
          "draw":     "KXEPLGAME-26MAR15ARSEVE-DRW",
      }
  )
  tickers = mapper.get_kalshi_tickers("GS_12345")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# Kalshi 시리즈 티커 매핑
# ═══════════════════════════════════════════════════

LEAGUE_SERIES = {
    "mls":         "KXMLSGAME",
    "epl":         "KXEPLGAME",
    "serie_a":     "KXSERIEAGAME",
    "ucl":         "KXUCLGAME",
    "la_liga":     "KXLALIGAGAME",
    "bundesliga":  "KXBUNDESLIGAGAME",
    "ligue_1":     "KXLIGUE1GAME",
    "liga_mx":     "KXLIGAMXGAME",
    "eredivisie":  "KXEREDIVISIEGAME",
    "super_lig":   "KXSUPERLIGGAME",
    "argentina":   "KXARGENTINAGAME",
}

# Goalserve 리그명 → league key 매핑
GOALSERVE_LEAGUE_MAP = {
    "USA: Mls": "mls",
    "England: Premier League": "epl",
    "Italy: Serie A": "serie_a",
    "Spain: La Liga": "la_liga",
    "Germany: Bundesliga": "bundesliga",
    "France: Ligue 1": "ligue_1",
    "Mexico: Liga Mx": "liga_mx",
    "Champions League": "ucl",
}

MARKET_TYPES = ["home_win", "away_win", "draw"]

# 팀명 약어 → 풀네임 매핑 (Kalshi ↔ Goalserve 연결용)
TEAM_ALIASES = {
    # MLS
    "van": ["vancouver", "whitecaps"],
    "tor": ["toronto", "toronto fc"],
    "dal": ["dallas", "fc dallas"],
    "nsh": ["nashville", "nashville sc"],
    "hou": ["houston", "dynamo"],
    "lafc": ["los angeles f", "lafc"],
    "lag": ["los angeles g", "la galaxy", "galaxy"],
    "skc": ["kansas city", "sporting"],
    "clb": ["columbus", "crew"],
    "clt": ["charlotte", "charlotte fc"],
    "phi": ["philadelphia", "union"],
    "nyc": ["new york city", "nycfc"],
    "nyrb": ["new york", "red bulls", "ny red bulls"],
    "orl": ["orlando", "orlando city"],
    "mia": ["miami", "inter miami"],
    "atl": ["atlanta", "atlanta united"],
    "sea": ["seattle", "sounders"],
    "por": ["portland", "timbers"],
    "col": ["colorado", "rapids"],
    "min": ["minnesota", "minnesota united"],
    "stl": ["saint louis", "st. louis", "st louis", "city sc"],
    "cin": ["cincinnati", "fc cincinnati"],
    "chi": ["chicago", "chicago fire"],
    "ne": ["new england", "revolution"],
    "dcu": ["dc united", "d.c. united"],
    "rsl": ["salt lake", "real salt lake"],
    "atx": ["austin", "austin fc"],
    "sd": ["san diego", "san diego fc"],
    "sj": ["san jose", "earthquakes"],
    "mtl": ["montreal", "cf montréal", "cf montreal"],
    # EPL
    "ars": ["arsenal"],
    "mun": ["manchester united", "man united", "man utd"],
    "mnc": ["manchester city", "man city"],
    "liv": ["liverpool"],
    "che": ["chelsea"],
    "tot": ["tottenham", "spurs"],
    "avl": ["aston villa"],
    "new": ["newcastle"],
    "bri": ["brighton"],
    "ful": ["fulham"],
    "whu": ["west ham"],
    "bou": ["bournemouth"],
    "cry": ["crystal palace"],
    "eve": ["everton"],
    "not": ["nottingham", "nottm forest"],
    "wol": ["wolves", "wolverhampton"],
    "ips": ["ipswich"],
    "lei": ["leicester"],
    "sou": ["southampton"],
}


def _fuzzy_team_match(team_name: str, text: str) -> bool:
    """
    팀명이 텍스트에 포함되는지 fuzzy 체크.

    "Vancouver Whitecaps" → "vancouver" in "vancouver vs toronto" ✓
    "FC Dallas" → "dallas" in "dallas vs nashville" ✓
    """
    team_lower = team_name.lower()

    # 직접 매칭
    if team_lower in text:
        return True

    # 단어 단위 매칭 (첫 단어만)
    words = team_lower.split()
    for w in words:
        if len(w) >= 4 and w in text:
            return True

    # 별명 매칭
    for alias_key, aliases in TEAM_ALIASES.items():
        if any(a in team_lower for a in aliases) or team_lower in aliases:
            # 이 별명의 다른 이름이 text에 있는지
            for a in aliases:
                if a in text:
                    return True
            if alias_key in text.split():
                return True

    return False


def _classify_markets(
    markets: list,
    home_lower: str,
    away_lower: str,
) -> Dict[str, str]:
    """
    이벤트의 마켓 3개를 home_win / away_win / draw로 분류.

    분류 로직 (우선순위):
      1. ticker 끝이 "TIE" → draw
      2. ticker 끝이 홈팀 약어 → home_win
      3. ticker 끝이 원정팀 약어 → away_win
      4. 남은 것 → 나머지 슬롯에 배정
    """
    result = {}
    unclassified = []

    # 홈/원정 별명 수집
    home_aliases = _get_team_aliases(home_lower)
    away_aliases = _get_team_aliases(away_lower)

    for m in markets:
        ticker = m.get("ticker", "")
        suffix = ticker.rsplit("-", 1)[-1].lower() if "-" in ticker else ""

        if suffix in ("tie", "drw", "draw"):
            result["draw"] = ticker
        elif suffix in home_aliases:
            result["home_win"] = ticker
        elif suffix in away_aliases:
            result["away_win"] = ticker
        else:
            unclassified.append(ticker)

    # 미분류 마켓 배정
    for ticker in unclassified:
        if "home_win" not in result:
            result["home_win"] = ticker
        elif "away_win" not in result:
            result["away_win"] = ticker
        elif "draw" not in result:
            result["draw"] = ticker

    return result


def _get_team_aliases(team_lower: str) -> set:
    """팀명에 해당하는 모든 약어 키를 반환."""
    aliases = set()
    for key, names in TEAM_ALIASES.items():
        if team_lower in names or any(n in team_lower for n in names):
            aliases.add(key)
        # 팀명의 단어가 names에 포함되는지
        for word in team_lower.split():
            if len(word) >= 4 and any(word in n for n in names):
                aliases.add(key)
    return aliases


# ═══════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════

@dataclass
class MatchMapping:
    """단일 경기의 티커 매핑."""
    match_id: str                          # Goalserve fixture ID
    home_team: str = ""
    away_team: str = ""
    league: str = ""
    kickoff: str = ""                      # ISO datetime
    kalshi_tickers: Dict[str, str] = field(default_factory=dict)
    # {"home_win": "KXEPLGAME-...", "away_win": "...", "draw": "..."}
    kalshi_event_ticker: str = ""          # 이벤트 티커 (있으면)


# ═══════════════════════════════════════════════════
# Ticker Mapper
# ═══════════════════════════════════════════════════

class TickerMapper:
    """
    Goalserve ↔ Kalshi 티커 매핑 관리자.

    수동 등록과 Kalshi API 기반 자동 탐색을 모두 지원.
    """

    def __init__(self):
        self._mappings: Dict[str, MatchMapping] = {}

    # ─── 수동 등록 ───────────────────────────────

    def register_match(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        kalshi_tickers: Dict[str, str],
        league: str = "",
        kickoff: str = "",
    ) -> MatchMapping:
        """
        수동으로 경기 매핑 등록.

        Args:
            match_id:       Goalserve fixture ID
            home_team:      홈팀명
            away_team:      원정팀명
            kalshi_tickers: {"home_win": "...", "away_win": "...", "draw": "..."}
            league:         리그 식별자 (epl, ucl, ...)
            kickoff:        킥오프 시간 (ISO)
        """
        mapping = MatchMapping(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            kickoff=kickoff,
            kalshi_tickers=kalshi_tickers,
        )
        self._mappings[match_id] = mapping
        logger.info(
            f"Registered match {match_id}: {home_team} vs {away_team} "
            f"→ {len(kalshi_tickers)} tickers"
        )
        return mapping

    # ─── 조회 ────────────────────────────────────

    def get_mapping(self, match_id: str) -> Optional[MatchMapping]:
        """match_id로 매핑 조회."""
        return self._mappings.get(match_id)

    def get_kalshi_tickers(self, match_id: str) -> Dict[str, str]:
        """match_id의 Kalshi 티커 dict 반환."""
        mapping = self._mappings.get(match_id)
        if mapping is None:
            return {}
        return mapping.kalshi_tickers

    def get_all_active(self) -> List[MatchMapping]:
        """등록된 모든 매핑."""
        return list(self._mappings.values())

    def has_match(self, match_id: str) -> bool:
        return match_id in self._mappings

    # ─── Kalshi API 기반 자동 탐색 ────────────────

    async def discover_from_kalshi(
        self,
        kalshi_client: Any,
        league: str,
        match_id: str = "",
        home_team: str = "",
        away_team: str = "",
    ) -> Optional[MatchMapping]:
        """
        Kalshi API에서 이벤트 기반으로 마켓을 자동 탐색.

        실제 Kalshi 구조:
          시리즈:  KXMLSGAME
          이벤트:  KXMLSGAME-26FEB28VANTOR  (경기 1개 = 이벤트 1개)
          마켓:    KXMLSGAME-26FEB28VANTOR-VAN  (홈승)
                   KXMLSGAME-26FEB28VANTOR-TOR  (원정승)
                   KXMLSGAME-26FEB28VANTOR-TIE  (무승부)

        탐색 흐름:
          ① 시리즈 → 이벤트 목록 조회
          ② 이벤트 title에서 팀명 fuzzy match
          ③ 이벤트 → 마켓 3개 조회
          ④ 마켓 ticker 끝 세그먼트로 home/away/draw 분류
        """
        series_ticker = LEAGUE_SERIES.get(league)
        if not series_ticker:
            logger.warning(f"Unknown league: {league}")
            return None

        # ① 이벤트 목록 조회
        try:
            data = await kalshi_client._get(
                "/events",
                params={
                    "series_ticker": series_ticker,
                    "status": "open",
                    "limit": "50",
                },
            )
            events = data.get("events", [])
        except Exception as e:
            logger.error(f"Kalshi events API error: {e}")
            return None

        if not events:
            logger.warning(f"No events for series {series_ticker}")
            return None

        # ② 이벤트에서 팀명 매칭
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        matched_event = None

        for ev in events:
            title = (ev.get("title", "") + " " + ev.get("sub_title", "")).lower()
            # "Vancouver vs Toronto" 같은 형태
            if _fuzzy_team_match(home_lower, title) and _fuzzy_team_match(away_lower, title):
                matched_event = ev
                break

        if not matched_event:
            logger.warning(
                f"No Kalshi event found for {home_team} vs {away_team} "
                f"in {series_ticker} ({len(events)} events)"
            )
            return None

        event_ticker = matched_event.get("event_ticker", "")
        logger.info(f"Matched event: {event_ticker}")

        # ③ 이벤트의 마켓 조회
        try:
            data = await kalshi_client._get(
                "/markets",
                params={
                    "event_ticker": event_ticker,
                    "limit": "10",
                },
            )
            markets = data.get("markets", [])
        except Exception as e:
            logger.error(f"Kalshi markets API error: {e}")
            return None

        if not markets:
            logger.warning(f"No markets for event {event_ticker}")
            return None

        # ④ 마켓 → home_win / away_win / draw 분류
        kalshi_tickers = _classify_markets(markets, home_lower, away_lower)

        if len(kalshi_tickers) < 2:
            logger.warning(
                f"Classification failed for {event_ticker}: "
                f"only {len(kalshi_tickers)} tickers classified"
            )
            return None

        mapping = self.register_match(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            kalshi_tickers=kalshi_tickers,
            league=league,
            kickoff=matched_event.get("close_time", ""),
        )
        mapping.kalshi_event_ticker = event_ticker
        return mapping

    async def discover_all_for_league(
        self,
        kalshi_client: Any,
        league: str,
    ) -> List[Dict]:
        """
        특정 리그의 모든 오픈 이벤트 정보를 반환.
        (매핑은 하지 않고 탐색만)

        Returns:
            [{event_ticker, title, home, away, markets: [{ticker, suffix}]}]
        """
        series_ticker = LEAGUE_SERIES.get(league)
        if not series_ticker:
            return []

        try:
            data = await kalshi_client._get(
                "/events",
                params={
                    "series_ticker": series_ticker,
                    "status": "open",
                    "limit": "50",
                },
            )
            events = data.get("events", [])
        except Exception as e:
            logger.error(f"Kalshi events API error: {e}")
            return []

        results = []
        for ev in events:
            title = ev.get("title", "")
            event_ticker = ev.get("event_ticker", "")

            # "Vancouver vs Toronto" → home, away 추출
            parts = title.split(" vs ")
            home = parts[0].strip() if len(parts) >= 2 else ""
            away = parts[1].strip() if len(parts) >= 2 else ""

            results.append({
                "event_ticker": event_ticker,
                "title": title,
                "home": home,
                "away": away,
            })

        return results

    # ─── 일괄 로드 (JSON config) ─────────────────

    def load_from_config(self, config: List[Dict]) -> int:
        """
        JSON 설정에서 일괄 로드.

        config 형식:
        [
            {
                "match_id": "GS_12345",
                "home_team": "Arsenal",
                "away_team": "Everton",
                "league": "epl",
                "kickoff": "2026-03-15T15:00:00Z",
                "kalshi_tickers": {
                    "home_win": "KXEPLGAME-26MAR15ARSEVE-ARS",
                    "away_win": "KXEPLGAME-26MAR15ARSEVE-EVE",
                    "draw":     "KXEPLGAME-26MAR15ARSEVE-DRW"
                }
            }
        ]
        """
        count = 0
        for entry in config:
            self.register_match(
                match_id=entry["match_id"],
                home_team=entry.get("home_team", ""),
                away_team=entry.get("away_team", ""),
                kalshi_tickers=entry.get("kalshi_tickers", {}),
                league=entry.get("league", ""),
                kickoff=entry.get("kickoff", ""),
            )
            count += 1
        return count

    # ─── 유틸리티 ────────────────────────────────

    def remove_match(self, match_id: str) -> bool:
        return self._mappings.pop(match_id, None) is not None

    def clear(self):
        self._mappings.clear()

    def summary(self) -> str:
        """현재 매핑 요약."""
        lines = [f"TickerMapper: {len(self._mappings)} matches"]
        for m in self._mappings.values():
            tickers = ", ".join(
                f"{k}={v}" for k, v in m.kalshi_tickers.items()
            )
            lines.append(f"  {m.match_id}: {m.home_team} vs {m.away_team} → {tickers}")
        return "\n".join(lines)
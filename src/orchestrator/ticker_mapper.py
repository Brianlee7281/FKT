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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# Kalshi 시리즈 티커 매핑
# ═══════════════════════════════════════════════════

LEAGUE_SERIES = {
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

MARKET_TYPES = ["home_win", "away_win", "draw"]


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
        Kalshi API에서 시리즈 기반으로 당일 마켓을 조회하고
        팀명 매칭으로 자동 매핑.

        Args:
            kalshi_client: KalshiClient 인스턴스
            league:        리그 식별자 (epl, ucl, ...)
            match_id:      Goalserve fixture ID
            home_team:     홈팀명 (매칭용)
            away_team:     원정팀명 (매칭용)

        Returns:
            MatchMapping or None (매칭 실패)
        """
        series_ticker = LEAGUE_SERIES.get(league)
        if not series_ticker:
            logger.warning(f"Unknown league: {league}")
            return None

        try:
            # Kalshi API로 해당 시리즈의 오픈 마켓 조회
            data = await kalshi_client._get(
                "/markets",
                params={
                    "series_ticker": series_ticker,
                    "status": "open",
                    "limit": "200",
                },
            )
            markets = data.get("markets", [])
        except Exception as e:
            logger.error(f"Kalshi API error: {e}")
            return None

        if not markets:
            return None

        # 팀명으로 필터 (title에서 검색)
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        matched = {}

        for m in markets:
            title = m.get("title", "").lower()
            subtitle = m.get("subtitle", "").lower()
            combined = title + " " + subtitle
            ticker = m.get("ticker", "")

            # 두 팀 모두 포함된 마켓 찾기
            if home_lower in combined and away_lower in combined:
                # outcome 분류 (ticker 끝부분으로)
                if ticker.endswith("-DRW") or "draw" in title:
                    matched["draw"] = ticker
                elif home_lower in title.split("to win")[0] if "to win" in title else "":
                    matched["home_win"] = ticker
                else:
                    # 팀 약어로 구분 (ticker 마지막 세그먼트)
                    last_seg = ticker.rsplit("-", 1)[-1].lower()
                    if any(w in last_seg for w in home_lower.split()[:1]):
                        matched["home_win"] = ticker
                    elif any(w in last_seg for w in away_lower.split()[:1]):
                        matched["away_win"] = ticker
                    elif "drw" in last_seg:
                        matched["draw"] = ticker

        if len(matched) < 2:  # 최소 2개는 매칭돼야
            logger.warning(
                f"Auto-match failed for {home_team} vs {away_team}: "
                f"only {len(matched)} tickers found"
            )
            return None

        mapping = self.register_match(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            kalshi_tickers=matched,
            league=league,
        )
        return mapping

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

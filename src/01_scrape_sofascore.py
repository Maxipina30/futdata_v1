import argparse
import asyncio
import json
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from playwright.async_api import async_playwright


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
API_URL = "https://api.sofascore.com/api/v1"
SITE_URL = "https://www.sofascore.com"

LEAGUES = {
    "premier": {
        "name": "Premier League",
        "competition": "Premier League",
        "unique_tournament_id": 17,
        "season_year": "25/26",
        "season_name": "Premier League 25/26",
    },
    "la_liga": {
        "name": "LaLiga",
        "competition": "La Liga",
        "unique_tournament_id": 8,
        "season_year": "25/26",
        "season_name": "LaLiga 25/26",
    },
    "serie_a": {
        "name": "Serie A",
        "competition": "Serie A",
        "unique_tournament_id": 23,
        "season_year": "25/26",
        "season_name": "Serie A 25/26",
    },
    "chile": {
        "name": "Liga de Primera",
        "competition": "Liga de Primera",
        "unique_tournament_id": 11653,
        "season_year": "2026",
        "season_name": "Primera Division 2026",
    },
}

TEAM_ALIASES = {
    "Brighton & Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Deportes Limache": "CD Limache",
    "Deportes Union La Calera": "Union La Calera",
    "Deportes Unión La Calera": "Union La Calera",
    "U. de Concepción": "Universidad de Concepcion",
    "Universidad de Concepción": "Universidad de Concepcion",
    "U. Católica": "Universidad Catolica",
    "Universidad Católica": "Universidad Catolica",
    "Ñublense": "Nublense",
    "Colo Colo": "Colo-Colo",
}

STAT_KEYS = {
    "ballPossession": "poss",
    "expectedGoals": "xg",
    "totalShotsOnGoal": "sh",
    "shotsOnGoal": "sot",
    "bigChanceCreated": "big_chances",
    "bigChanceScored": "big_chances_scored",
    "bigChanceMissed": "big_chances_missed",
    "cornerKicks": "corner_kicks",
    "fouls": "fouls",
    "passes": "passes",
    "accuratePasses": "accurate_passes",
    "totalTackle": "tackles",
    "freeKicks": "free_kicks",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "goalkeeperSaves": "goalkeeper_saves",
    "blockedScoringAttempt": "blocked_shots",
    "shotsOffGoal": "shots_off_target",
    "totalShotsInsideBox": "shots_inside_box",
    "totalShotsOutsideBox": "shots_outside_box",
    "accurateThroughBall": "through_balls",
    "touchesInOppBox": "touches_in_box",
    "accurateLongBalls": "accurate_long_balls",
    "accurateCross": "accurate_crosses",
    "duelWonPercent": "duel_win_pct",
    "dispossessed": "dispossessed",
    "groundDuelsPercentage": "ground_duels_won",
    "aerialDuelsPercentage": "aerial_duels_won",
    "dribblesPercentage": "successful_dribbles",
    "interceptionWon": "interceptions",
    "totalClearance": "clearances",
    "errorsLeadToShot": "errors_lead_to_shot",
    "highClaims": "high_claims",
    "goalKicks": "goal_kicks",
}

ZERO_FILL_STATS = {
    "big_chances",
    "big_chances_scored",
    "big_chances_missed",
    "corner_kicks",
    "fouls",
    "passes",
    "accurate_passes",
    "tackles",
    "free_kicks",
    "offsides",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
    "blocked_shots",
    "shots_off_target",
    "shots_inside_box",
    "shots_outside_box",
    "through_balls",
    "touches_in_box",
    "accurate_long_balls",
    "accurate_crosses",
    "dispossessed",
    "ground_duels_won",
    "aerial_duels_won",
    "successful_dribbles",
    "interceptions",
    "clearances",
    "errors_lead_to_shot",
    "high_claims",
    "goal_kicks",
}

SCHEDULE_STAT_COLUMNS = [
    value
    for value in dict.fromkeys(STAT_KEYS.values())
    if value not in {"poss", "xg", "sh", "sot"}
]

ODDS_MARKETS = {
    "Full time": {"1": "decimal_home_win", "X": "decimal_draw", "2": "decimal_away_win"},
    "Double chance": {
        "1X": "decimal_home_or_draw",
        "12": "decimal_home_or_away",
        "X2": "decimal_draw_or_away",
    },
    "Both teams to score": {"Yes": "decimal_btts_yes", "No": "decimal_btts_no"},
}


def clean_team_name(value):
    if not isinstance(value, str):
        return value
    return TEAM_ALIASES.get(value.strip(), value.strip())


def event_datetime(event, timezone):
    return datetime.fromtimestamp(event["startTimestamp"], ZoneInfo(timezone))


def result_for(gf, ga, finished):
    if not finished or gf is None or ga is None:
        return None
    if gf > ga:
        return "W"
    if gf < ga:
        return "L"
    return "D"


def pct(numerator, denominator):
    if denominator in (None, 0) or pd.isna(denominator):
        return None
    return round(float(numerator) / float(denominator) * 100, 1)


def ratio(numerator, denominator, digits=3):
    if numerator in (None, "") or denominator in (None, "", 0) or pd.isna(numerator) or pd.isna(denominator):
        return None
    return round(float(numerator) / float(denominator), digits)


def get_score(event, side):
    score = event.get(f"{side}Score") or {}
    return score.get("normaltime", score.get("current"))


def fraction_to_decimal(value):
    if value in (None, ""):
        return None
    try:
        return round(float(Fraction(str(value))) + 1, 4)
    except Exception:
        return None


def normalize_market_name(name):
    return " ".join(str(name or "").strip().split())


def parse_statistics(payload):
    by_side = {"home": {}, "away": {}}
    if not isinstance(payload, dict):
        return by_side
    periods = payload.get("statistics") or []
    all_period = next((period for period in periods if period.get("period") == "ALL"), None)
    if all_period is None and periods:
        all_period = periods[0]
    for group in (all_period or {}).get("groups", []):
        for item in group.get("statisticsItems", []):
            key = STAT_KEYS.get(item.get("key"))
            if not key:
                continue
            by_side["home"][key] = item.get("homeValue")
            by_side["away"][key] = item.get("awayValue")
    if all_period:
        for side in by_side:
            for key in ZERO_FILL_STATS:
                by_side[side].setdefault(key, 0)
    return by_side


def parse_odds(event, payload, timezone="America/Sao_Paulo"):
    dt = event_datetime(event, timezone)
    row = {
        "source_url": f"{SITE_URL}/{event.get('slug', '')}/{event.get('customId', '')}",
        "date": dt.date().isoformat(),
        "time": dt.strftime("%H:%M"),
        "local_team": clean_team_name(event.get("homeTeam", {}).get("name")),
        "away_team": clean_team_name(event.get("awayTeam", {}).get("name")),
    }
    if not isinstance(payload, dict):
        return row

    for market in payload.get("markets", []):
        market_name = normalize_market_name(market.get("marketName"))
        mapping = ODDS_MARKETS.get(market_name)
        if mapping:
            for choice in market.get("choices", []):
                col = mapping.get(choice.get("name"))
                if col:
                    row[col] = fraction_to_decimal(choice.get("fractionalValue"))
            continue

        if market_name == "Total goals":
            handicap = market.get("handicap")
            for choice in market.get("choices", []):
                name = choice.get("name")
                value = fraction_to_decimal(choice.get("fractionalValue"))
                if handicap in ("1.5", 1.5):
                    if name == "Over":
                        row["decimal_over_15"] = value
                    elif name == "Under":
                        row["decimal_under_15"] = value
                elif handicap in ("2.5", 2.5):
                    if name == "Over":
                        row["decimal_over_25"] = value
                    elif name == "Under":
                        row["decimal_under_25"] = value
    return row


def h2h_stats_from_events(events, current_event, prefix):
    current_home = clean_team_name(current_event.get("homeTeam", {}).get("name"))
    current_away = clean_team_name(current_event.get("awayTeam", {}).get("name"))
    local_goals = []
    away_goals = []
    local_points = []
    away_points = []

    for event in events:
        home_team = clean_team_name(event.get("homeTeam", {}).get("name"))
        away_team = clean_team_name(event.get("awayTeam", {}).get("name"))
        home_score = get_score(event, "home")
        away_score = get_score(event, "away")
        if home_score is None or away_score is None:
            continue
        if home_team == current_home and away_team == current_away:
            lg, ag = home_score, away_score
        elif home_team == current_away and away_team == current_home:
            lg, ag = away_score, home_score
        else:
            continue

        local_goals.append(float(lg))
        away_goals.append(float(ag))
        if lg > ag:
            local_points.append(3)
            away_points.append(0)
        elif lg < ag:
            local_points.append(0)
            away_points.append(3)
        else:
            local_points.append(1)
            away_points.append(1)

    if not local_goals:
        return {
            f"{prefix}_matches": 0,
            f"{prefix}_local_win_pct": 0,
            f"{prefix}_draw_pct": 0,
            f"{prefix}_away_win_pct": 0,
            f"{prefix}_local_goals_avg": 0,
            f"{prefix}_away_goals_avg": 0,
            f"{prefix}_total_goals_avg": 0,
            f"{prefix}_btts_pct": 0,
            f"{prefix}_over25_pct": 0,
            f"{prefix}_local_points_avg": 0,
            f"{prefix}_away_points_avg": 0,
        }

    local_goals = pd.Series(local_goals, dtype=float)
    away_goals = pd.Series(away_goals, dtype=float)
    total_goals = local_goals + away_goals
    return {
        f"{prefix}_matches": int(len(local_goals)),
        f"{prefix}_local_win_pct": float((local_goals > away_goals).mean()),
        f"{prefix}_draw_pct": float((local_goals == away_goals).mean()),
        f"{prefix}_away_win_pct": float((local_goals < away_goals).mean()),
        f"{prefix}_local_goals_avg": float(local_goals.mean()),
        f"{prefix}_away_goals_avg": float(away_goals.mean()),
        f"{prefix}_total_goals_avg": float(total_goals.mean()),
        f"{prefix}_btts_pct": float(((local_goals > 0) & (away_goals > 0)).mean()),
        f"{prefix}_over25_pct": float((total_goals > 2.5).mean()),
        f"{prefix}_local_points_avg": float(pd.Series(local_points, dtype=float).mean()),
        f"{prefix}_away_points_avg": float(pd.Series(away_points, dtype=float).mean()),
    }


def parse_event_h2h(event, payload, league_config, timezone):
    dt = event_datetime(event, timezone)
    row = {
        "date": dt.date().isoformat(),
        "time": dt.strftime("%H:%M"),
        "local_team": clean_team_name(event.get("homeTeam", {}).get("name")),
        "away_team": clean_team_name(event.get("awayTeam", {}).get("name")),
        "event_id": event.get("id"),
        "custom_id": event.get("customId"),
    }
    h2h_events = (payload or {}).get("events", []) if isinstance(payload, dict) else []
    filtered = []
    for h2h_event in h2h_events:
        unique_tournament = (h2h_event.get("tournament") or {}).get("uniqueTournament") or {}
        if unique_tournament.get("id") != league_config["unique_tournament_id"]:
            continue
        if h2h_event.get("startTimestamp", 0) >= event.get("startTimestamp", 0):
            continue
        if (h2h_event.get("status") or {}).get("type") != "finished":
            continue
        filtered.append(h2h_event)
    filtered = sorted(filtered, key=lambda item: item.get("startTimestamp", 0))
    for name, window in {"all": None, "last10": 10, "last5": 5}.items():
        scoped = filtered[-window:] if window else filtered
        row.update(h2h_stats_from_events(scoped, event, f"h2h_{name}"))
    return row


async def api_get_json(context, url, referer, required=False):
    response = await context.request.get(
        url,
        headers={"Accept": "application/json,text/plain,*/*", "Referer": referer},
        timeout=60_000,
    )
    text = await response.text()
    if response.status == 404 and not required:
        return None
    if response.status != 200:
        if required:
            raise RuntimeError(f"HTTP {response.status} en {url}: {text[:300]}")
        print(f"ADVERTENCIA HTTP {response.status}: {url}")
        return None
    return json.loads(text)


async def get_season_id(context, league_config, referer):
    tid = league_config["unique_tournament_id"]
    payload = await api_get_json(
        context,
        f"{API_URL}/unique-tournament/{tid}/seasons",
        referer,
        required=True,
    )
    seasons = payload.get("seasons", [])
    for season in seasons:
        if season.get("year") == league_config["season_year"]:
            return season["id"]
    for season in seasons:
        if season.get("name") == league_config["season_name"]:
            return season["id"]
    raise RuntimeError(f"No encontré temporada {league_config['season_year']} para {league_config['name']}")


async def get_seasons(context, league_config, referer):
    payload = await api_get_json(
        context,
        f"{API_URL}/unique-tournament/{league_config['unique_tournament_id']}/seasons",
        referer,
        required=True,
    )
    return payload.get("seasons", [])


def choose_seasons(seasons, current_season_id, seasons_back):
    season_ids = [season["id"] for season in seasons]
    if current_season_id not in season_ids:
        return [{"id": current_season_id, "name": str(current_season_id), "year": ""}]
    start = season_ids.index(current_season_id)
    selected = seasons[start : start + seasons_back + 1]
    return selected or [{"id": current_season_id, "name": str(current_season_id), "year": ""}]


async def get_season_events(context, tournament_id, season_id, referer, include_future):
    events = []
    paths = ["last", "next"] if include_future else ["last"]
    for path in paths:
        page = 0
        while True:
            payload = await api_get_json(
                context,
                f"{API_URL}/unique-tournament/{tournament_id}/season/{season_id}/events/{path}/{page}",
                referer,
            )
            if not payload:
                break
            page_events = payload.get("events", [])
            if path == "last":
                page_events = list(reversed(page_events))
            events.extend(page_events)
            if not payload.get("hasNextPage"):
                break
            page += 1
    deduped = {event["id"]: event for event in events if event.get("id")}
    return sorted(deduped.values(), key=lambda event: event.get("startTimestamp", 0))


async def enrich_event(context, event, referer, fetch_odds, fetch_h2h, league_config, timezone):
    event_id = event["id"]
    statistics = await api_get_json(context, f"{API_URL}/event/{event_id}/statistics", referer)
    odds = None
    if fetch_odds:
        odds = await api_get_json(context, f"{API_URL}/event/{event_id}/odds/1/all", referer)
    h2h = None
    if fetch_h2h and event.get("customId"):
        h2h_payload = await api_get_json(
            context,
            f"{API_URL}/event/{event['customId']}/h2h/events",
            referer,
        )
        h2h = parse_event_h2h(event, h2h_payload, league_config, timezone)
    return parse_statistics(statistics), parse_odds(event, odds, timezone) if odds else None, h2h


def schedule_rows_for_event(event, stats, league_config, timezone, season_id=None, season_name=None, season_year=None):
    home = clean_team_name(event.get("homeTeam", {}).get("name"))
    away = clean_team_name(event.get("awayTeam", {}).get("name"))
    dt = event_datetime(event, timezone)
    home_score = get_score(event, "home")
    away_score = get_score(event, "away")
    finished = event.get("status", {}).get("type") == "finished"
    round_num = (event.get("roundInfo") or {}).get("round")
    rows = []
    for side, team, opponent, venue, gf, ga in [
        ("home", home, away, "Home", home_score, away_score),
        ("away", away, home, "Away", away_score, home_score),
    ]:
        opponent_side = "away" if side == "home" else "home"
        row = {
            "date": dt.date().isoformat(),
            "time": dt.strftime("%H:%M"),
            "season_id": season_id,
            "season_name": season_name,
            "season_year": season_year,
            "comp": league_config["competition"],
            "round": f"Matchweek {round_num}" if round_num else None,
            "day": dt.strftime("%a"),
            "venue": venue,
            "result": result_for(gf, ga, finished),
            "gf": gf,
            "ga": ga,
            "opponent": opponent,
            "poss": stats[side].get("poss"),
            "attendance": None,
            "captain": None,
            "formation": None,
            "opp_formation": None,
            "referee": event.get("referee", {}).get("name") if event.get("referee") else None,
            "match_report": f"{SITE_URL}/{event.get('slug', '')}/{event.get('customId', '')}",
            "notes": event.get("status", {}).get("description"),
            "team": team,
            "table_side": "for",
            "xg": stats[side].get("xg"),
            "xga": stats[opponent_side].get("xg"),
        }
        for stat_col in SCHEDULE_STAT_COLUMNS:
            row[stat_col] = stats[side].get(stat_col)
            row[f"{stat_col}_allowed"] = stats[opponent_side].get(stat_col)

        row["pass_accuracy_pct"] = pct(stats[side].get("accurate_passes"), stats[side].get("passes"))
        row["pass_accuracy_pct_allowed"] = pct(
            stats[opponent_side].get("accurate_passes"),
            stats[opponent_side].get("passes"),
        )
        row["xg_per_sh"] = ratio(stats[side].get("xg"), stats[side].get("sh"))
        row["xg_per_sh_allowed"] = ratio(stats[opponent_side].get("xg"), stats[opponent_side].get("sh"))
        row["sot_per_sh"] = ratio(stats[side].get("sot"), stats[side].get("sh"))
        row["sot_per_sh_allowed"] = ratio(stats[opponent_side].get("sot"), stats[opponent_side].get("sh"))
        row["goals_minus_xg"] = None if gf is None or stats[side].get("xg") is None else round(float(gf) - float(stats[side]["xg"]), 3)
        row["goals_minus_xg_allowed"] = None if ga is None or stats[opponent_side].get("xg") is None else round(float(ga) - float(stats[opponent_side]["xg"]), 3)
        row["save_pct"] = pct(stats[side].get("goalkeeper_saves"), stats[opponent_side].get("sot"))
        row["save_pct_allowed"] = pct(stats[opponent_side].get("goalkeeper_saves"), stats[side].get("sot"))
        rows.append(row)
    return rows


def shooting_rows_for_event(event, stats, timezone, table_side, season_id=None, season_name=None, season_year=None):
    home = clean_team_name(event.get("homeTeam", {}).get("name"))
    away = clean_team_name(event.get("awayTeam", {}).get("name"))
    dt = event_datetime(event, timezone)
    home_score = get_score(event, "home")
    away_score = get_score(event, "away")
    finished = event.get("status", {}).get("type") == "finished"
    round_num = (event.get("roundInfo") or {}).get("round")
    rows = []
    for side, team, opponent, venue, gf, ga in [
        ("home", home, away, "Home", home_score, away_score),
        ("away", away, home, "Away", away_score, home_score),
    ]:
        source_side = side if table_side == "for" else ("away" if side == "home" else "home")
        sh = stats[source_side].get("sh")
        sot = stats[source_side].get("sot")
        rows.append(
            {
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "season_id": season_id,
                "season_name": season_name,
                "season_year": season_year,
                "round": f"Matchweek {round_num}" if round_num else None,
                "day": dt.strftime("%a"),
                "venue": venue,
                "result": result_for(gf, ga, finished),
                "gf": gf,
                "ga": ga,
                "opponent": opponent,
                "gls": gf,
                "sh": sh,
                "sot": sot,
                "sotpct": pct(sot, sh),
                "g_per_sh": round(gf / sh, 3) if sh else None,
                "g_per_sot": round(gf / sot, 3) if sot else None,
                "pk": None,
                "pkatt": None,
                "team": team,
                "table_side": table_side,
            }
        )
    return rows


async def scrape(args):
    league_config = LEAGUES[args.league]
    if args.pipeline_root:
        root = Path(args.pipeline_root)
        if not root.is_absolute():
            root = BASE_DIR / root
        raw_dir = root / "raw" / args.league / "team_matchlogs"
        odds_dir = root / "odds"
        cache_dir = root / "cache" / args.league
    else:
        raw_dir = BASE_DIR / "files" / "01_raw" / args.league / "team_matchlogs"
        odds_dir = BASE_DIR / "files" / "06_odds"
        cache_dir = BASE_DIR / "files" / "00_cache" / "sofascore" / args.league
    raw_dir.mkdir(parents=True, exist_ok=True)
    odds_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        context = await browser.new_context(
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        referer = f"{SITE_URL}/football/tournament/{args.league}/{league_config['unique_tournament_id']}"
        page = await context.new_page()
        await page.goto(referer, wait_until="domcontentloaded", timeout=60_000)

        seasons = await get_seasons(context, league_config, referer)
        season_id = args.season_id or await get_season_id(context, league_config, referer)
        selected_seasons = choose_seasons(seasons, season_id, args.seasons_back)
        events_by_season = []
        for selected_season in selected_seasons:
            selected_season_id = selected_season["id"]
            season_events = await get_season_events(
                context,
                league_config["unique_tournament_id"],
                selected_season_id,
                referer,
                args.include_future if selected_season_id == season_id else False,
            )
            events_by_season.append((selected_season, season_events))
        events = [
            event
            for _, season_events in events_by_season
            for event in season_events
        ]
        if args.from_date:
            events_by_season = [
                (
                    season,
                    [event for event in season_events if event_datetime(event, args.timezone).date().isoformat() >= args.from_date],
                )
                for season, season_events in events_by_season
            ]
        if args.to_date:
            events_by_season = [
                (
                    season,
                    [event for event in season_events if event_datetime(event, args.timezone).date().isoformat() <= args.to_date],
                )
                for season, season_events in events_by_season
            ]
        if args.limit:
            remaining = args.limit
            limited = []
            for season, season_events in events_by_season:
                limited_events = season_events[:remaining]
                limited.append((season, limited_events))
                remaining -= len(limited_events)
                if remaining <= 0:
                    break
            events_by_season = limited
        events = [
            event
            for _, season_events in events_by_season
            for event in season_events
        ]

        print(
            f"{league_config['name']} seasons="
            f"{', '.join(str(season['id']) for season, _ in events_by_season)}: "
            f"{len(events)} eventos"
        )
        (cache_dir / f"events_{season_id}.json").write_text(
            json.dumps({"seasons": selected_seasons, "events": events}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        schedule_rows = []
        shooting_for_rows = []
        shooting_against_rows = []
        odds_rows = []
        h2h_rows = []
        total_events = len(events)
        processed = 0
        for season, season_events in events_by_season:
            selected_season_id = season["id"]
            selected_season_name = season.get("name")
            selected_season_year = season.get("year")
            for event in season_events:
                processed += 1
                stats, odds, h2h = await enrich_event(
                    context,
                    event,
                    referer,
                    args.odds,
                    args.h2h,
                    league_config,
                    args.timezone,
                )
                schedule_rows.extend(
                    schedule_rows_for_event(
                        event,
                        stats,
                        league_config,
                        args.timezone,
                        selected_season_id,
                        selected_season_name,
                        selected_season_year,
                    )
                )
                shooting_for_rows.extend(
                    shooting_rows_for_event(
                        event,
                        stats,
                        args.timezone,
                        "for",
                        selected_season_id,
                        selected_season_name,
                        selected_season_year,
                    )
                )
                shooting_against_rows.extend(
                    shooting_rows_for_event(
                        event,
                        stats,
                        args.timezone,
                        "against",
                        selected_season_id,
                        selected_season_name,
                        selected_season_year,
                    )
                )
                if odds:
                    odds["season_id"] = selected_season_id
                    odds["season_name"] = selected_season_name
                    odds_rows.append(odds)
                if h2h:
                    h2h["season_id"] = selected_season_id
                    h2h["season_name"] = selected_season_name
                    h2h_rows.append(h2h)
                if processed % 20 == 0 or processed == total_events:
                    print(f"Procesados {processed}/{total_events} eventos")
                if args.delay:
                    await asyncio.sleep(args.delay)

        await browser.close()

    pd.DataFrame(schedule_rows).to_csv(raw_dir / f"{args.league}_team_schedule.csv", index=False)
    pd.DataFrame(shooting_for_rows).to_csv(raw_dir / f"{args.league}_team_shooting_for.csv", index=False)
    pd.DataFrame(shooting_against_rows).to_csv(raw_dir / f"{args.league}_team_shooting_against.csv", index=False)
    if odds_rows:
        pd.DataFrame(odds_rows).to_csv(odds_dir / f"sofascore_{args.league}_odds.csv", index=False)
    if h2h_rows:
        pd.DataFrame(h2h_rows).to_csv(raw_dir.parent / f"{args.league}_h2h_features.csv", index=False)

    print(f"Guardado matchlogs SofaScore en {raw_dir}")
    if odds_rows:
        print(f"Guardado cuotas SofaScore en {odds_dir / f'sofascore_{args.league}_odds.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier", choices=sorted(LEAGUES))
    parser.add_argument("--season-id", type=int, default=None)
    parser.add_argument("--seasons-back", type=int, default=0)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timezone", default="America/Sao_Paulo")
    parser.add_argument("--include-future", action="store_true")
    parser.add_argument("--odds", action="store_true")
    parser.add_argument("--h2h", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--pipeline-root", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    asyncio.run(scrape(args))


if __name__ == "__main__":
    main()

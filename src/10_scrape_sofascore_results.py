import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from playwright.async_api import async_playwright


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = BASE_DIR / "files" / "sofascore_pipeline"
API_URL = "https://api.sofascore.com/api/v1"
SITE_URL = "https://www.sofascore.com"


def event_datetime(event, timezone):
    timestamp = event.get("startTimestamp")
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))


def get_score(event, side):
    score = event.get(f"{side}Score") or {}
    return score.get("normaltime", score.get("current", score.get("display")))


def clean_team_name(value):
    return str(value or "").strip()


def target_from_goals(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return pd.NA
    if home_goals > away_goals:
        return 1
    if home_goals < away_goals:
        return -1
    return 0


def cached_events(root, from_date, to_date, timezone):
    events = []
    for path in (root / "cache").glob("*/events_*.json"):
        league = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload.get("events", []):
            dt = event_datetime(event, timezone)
            if dt is None:
                continue
            event_date = dt.date().isoformat()
            if from_date <= event_date <= to_date:
                event = dict(event)
                event["_league"] = league
                events.append(event)
    deduped = {event["id"]: event for event in events if event.get("id")}
    return sorted(deduped.values(), key=lambda item: item.get("startTimestamp", 0))


async def fetch_event(context, event_id):
    response = await context.request.get(
        f"{API_URL}/event/{event_id}",
        headers={"Accept": "application/json,text/plain,*/*", "Referer": SITE_URL},
        timeout=60_000,
    )
    if response.status != 200:
        text = await response.text()
        print(f"ADVERTENCIA HTTP {response.status} event={event_id}: {text[:160]}")
        return None
    payload = await response.json()
    return payload.get("event")


def result_row(event, fallback_event, timezone):
    dt = event_datetime(event, timezone) or event_datetime(fallback_event, timezone)
    status = event.get("status") or {}
    status_type = status.get("type")
    finished = status_type == "finished"
    home_goals = get_score(event, "home")
    away_goals = get_score(event, "away")
    if not finished:
        home_goals = pd.NA
        away_goals = pd.NA

    home_team = clean_team_name((event.get("homeTeam") or {}).get("name"))
    away_team = clean_team_name((event.get("awayTeam") or {}).get("name"))
    target = target_from_goals(home_goals, away_goals)
    total_goals = (
        pd.to_numeric(pd.Series([home_goals]), errors="coerce").iloc[0]
        + pd.to_numeric(pd.Series([away_goals]), errors="coerce").iloc[0]
    )
    target_over_15 = int(total_goals >= 2) if finished and pd.notna(total_goals) else pd.NA

    return {
        "league": fallback_event.get("_league"),
        "date": dt.date().isoformat() if dt else None,
        "time": dt.strftime("%H:%M") if dt else None,
        "round_num": (event.get("roundInfo") or fallback_event.get("roundInfo") or {}).get("round"),
        "local_team": home_team,
        "away_team": away_team,
        "status_type": status_type,
        "status_description": status.get("description"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "Target": target,
        "target_over_15": target_over_15,
        "event_id": event.get("id"),
        "source_url": f"{SITE_URL}/{event.get('slug', '')}/{event.get('customId', '')}",
    }


async def scrape_results(args):
    events = cached_events(PIPELINE_ROOT, args.from_date, args.to_date, args.timezone)
    if not events:
        raise RuntimeError("No hay eventos cacheados para ese rango. Ejecuta primero el scraping SofaScore con include_future.")

    rows = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.no_headless)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        for idx, fallback_event in enumerate(events, start=1):
            event = await fetch_event(context, fallback_event["id"])
            if event:
                rows.append(result_row(event, fallback_event, args.timezone))
            if idx % 10 == 0 or idx == len(events):
                print(f"Resultados revisados {idx}/{len(events)}")
            if args.delay:
                await asyncio.sleep(args.delay)
        await browser.close()

    output_dir = PIPELINE_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sofascore_weekend_results_{args.from_date}_to_{args.to_date}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Guardado resultados SofaScore en {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default="2026-05-01")
    parser.add_argument("--to-date", default="2026-05-04")
    parser.add_argument("--timezone", default="America/Sao_Paulo")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()
    asyncio.run(scrape_results(args))


if __name__ == "__main__":
    main()

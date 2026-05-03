import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = BASE_DIR / "files" / "sofascore_pipeline"
API_URL = "https://api.sofascore.com/api/v1"
SITE_URL = "https://www.sofascore.com"


def load_scraper_module():
    path = BASE_DIR / "src" / "01_scrape_sofascore.py"
    spec = importlib.util.spec_from_file_location("sofascore_scraper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cached_events(root, league, from_date=None, to_date=None):
    cache_dir = root / "cache" / league
    events = []
    for path in cache_dir.glob("events_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        events.extend(payload.get("events", []))
    deduped = {event["id"]: event for event in events if event.get("id")}
    events = sorted(deduped.values(), key=lambda item: item.get("startTimestamp", 0))
    if from_date or to_date:
        scraper = load_scraper_module()
        filtered = []
        for event in events:
            event_date = scraper.event_datetime(event, "America/Sao_Paulo").date().isoformat()
            if from_date and event_date < from_date:
                continue
            if to_date and event_date > to_date:
                continue
            filtered.append(event)
        events = filtered
    return events


async def fetch_odds(context, event_id):
    response = await context.request.get(
        f"{API_URL}/event/{event_id}/odds/1/all",
        headers={"Accept": "application/json,text/plain,*/*", "Referer": SITE_URL},
        timeout=60_000,
    )
    if response.status == 404:
        return None
    if response.status != 200:
        text = await response.text()
        print(f"ADVERTENCIA HTTP {response.status} event={event_id}: {text[:160]}")
        return None
    return await response.json()


async def refresh(args):
    scraper = load_scraper_module()
    events = cached_events(PIPELINE_ROOT, args.league, args.from_date, args.to_date)
    if args.limit:
        events = events[: args.limit]
    if not events:
        raise RuntimeError(f"No hay eventos cacheados para {args.league}.")

    rows = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.no_headless)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        for idx, event in enumerate(events, start=1):
            odds = await fetch_odds(context, event["id"])
            if odds:
                rows.append(scraper.parse_odds(event, odds, args.timezone))
            if idx % 20 == 0 or idx == len(events):
                print(f"Cuotas revisadas {idx}/{len(events)}")
            if args.delay:
                await asyncio.sleep(args.delay)
        await browser.close()

    output_dir = PIPELINE_ROOT / "odds"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sofascore_{args.league}_odds.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Guardado cuotas SofaScore en {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timezone", default="America/Sao_Paulo")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()
    asyncio.run(refresh(args))


if __name__ == "__main__":
    main()

"""
Scraper de SofaScore para TODOS los partidos de la Primera División de Chile 2026.

Uso:
    venv312/Scripts/python.exe apps\\liga_chilena\\src\01_scrape.py
"""

import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR  = BASE_DIR / "apps" / "liga_chilena" / "data" / "raw"

API_URL  = "https://api.sofascore.com/api/v1"
SITE_URL = "https://www.sofascore.com"

TOURNAMENT_ID = 11653
SEASON_YEAR   = "2026"


async def api_get(context, url, referer):
    response = await context.request.get(
        url,
        headers={"Accept": "application/json,text/plain,*/*", "Referer": referer},
        timeout=60_000,
    )
    text = await response.text()
    if response.status == 404:
        return None
    if response.status != 200:
        print(f"  HTTP {response.status}: {url}")
        return None
    return json.loads(text)


async def get_season_id(context, referer):
    payload = await api_get(
        context,
        f"{API_URL}/unique-tournament/{TOURNAMENT_ID}/seasons",
        referer,
    )
    if not payload:
        raise RuntimeError("No se pudo obtener la lista de temporadas")
    seasons = payload.get("seasons", [])
    for season in seasons:
        if season.get("year") == SEASON_YEAR:
            return season["id"], season.get("name", SEASON_YEAR)
    if seasons:
        s = seasons[0]
        print(f"  Temporada '{SEASON_YEAR}' no encontrada, usando '{s.get('year')}' (id={s['id']})")
        return s["id"], s.get("name", "")
    raise RuntimeError("No hay temporadas disponibles")


async def get_all_events(context, season_id, referer):
    events = []
    for path in ("last", "next"):
        page = 0
        while True:
            payload = await api_get(
                context,
                f"{API_URL}/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/events/{path}/{page}",
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
    deduped = {e["id"]: e for e in events if e.get("id")}
    return sorted(deduped.values(), key=lambda e: e.get("startTimestamp", 0))


def is_finished(event):
    return (event.get("status") or {}).get("type") == "finished"


async def fetch_match_data(context, event, referer):
    eid  = event["id"]
    home = (event.get("homeTeam") or {}).get("name", "?")
    away = (event.get("awayTeam") or {}).get("name", "?")
    print(f"  [{eid}] {home} vs {away}")

    lineups   = await api_get(context, f"{API_URL}/event/{eid}/lineups",   referer)
    incidents = await api_get(context, f"{API_URL}/event/{eid}/incidents", referer)

    match_dir = OUT_DIR / str(eid)
    match_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "event_id":        eid,
        "home_team":       home,
        "away_team":       away,
        "home_team_id":    (event.get("homeTeam") or {}).get("id"),
        "away_team_id":    (event.get("awayTeam") or {}).get("id"),
        "start_timestamp": event.get("startTimestamp"),
        "round":           (event.get("roundInfo") or {}).get("round"),
        "home_score": (
            (event.get("homeScore") or {}).get("normaltime")
            or (event.get("homeScore") or {}).get("current")
        ),
        "away_score": (
            (event.get("awayScore") or {}).get("normaltime")
            or (event.get("awayScore") or {}).get("current")
        ),
    }
    (match_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if lineups:
        (match_dir / "lineups.json").write_text(
            json.dumps(lineups, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"    Sin lineups")

    if incidents:
        (match_dir / "incidents.json").write_text(
            json.dumps(incidents, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"    Sin incidents")

    return lineups is not None and incidents is not None


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    referer = f"{SITE_URL}/football"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        await page.goto(referer, wait_until="domcontentloaded", timeout=60_000)

        print("Obteniendo temporada...")
        season_id, season_name = await get_season_id(context, referer)
        print(f"  Temporada: {season_name} (id={season_id})")

        print("Obteniendo partidos de la temporada...")
        all_events = await get_all_events(context, season_id, referer)
        finished   = [e for e in all_events if is_finished(e)]
        print(f"  Total eventos: {len(all_events)} | Terminados: {len(finished)}")

        ok = 0
        for event in finished:
            already = OUT_DIR / str(event["id"]) / "incidents.json"
            if already.exists():
                print(f"  [cache] {event['id']}")
                ok += 1
                continue
            success = await fetch_match_data(context, event, referer)
            if success:
                ok += 1

        await browser.close()

    print(f"\nListo: {ok}/{len(finished)} partidos con datos completos.")
    print(f"Datos en: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


BASE_URL = "https://www.sofascore.com"
API_URL = "https://api.sofascore.com/api/v1"
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "files" / "00_cache" / "sofascore_probe"


async def fetch_json(context, url, referer):
    response = await context.request.get(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Referer": referer,
        },
        timeout=60_000,
    )
    text = await response.text()
    return response.status, text


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-25")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page_url = f"{BASE_URL}/football/{args.date}"
    scheduled_url = f"{API_URL}/sport/football/scheduled-events/{args.date}/inverse"

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
        await page.goto(page_url, wait_until="domcontentloaded", timeout=60_000)

        status, text = await fetch_json(context, scheduled_url, page_url)
        print(f"scheduled status: {status}")
        print(text[:1000])
        if status != 200:
            await browser.close()
            return

        scheduled = json.loads(text)
        events = scheduled.get("events", [])
        print(f"events: {len(events)}")
        probe_events = events[: args.limit]
        (OUT_DIR / f"scheduled_{args.date}.json").write_text(
            json.dumps(scheduled, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for event in probe_events:
            event_id = event.get("id")
            slug = event.get("slug")
            if not event_id:
                continue
            print(
                "\nEVENT",
                event_id,
                event.get("homeTeam", {}).get("name"),
                "-",
                event.get("awayTeam", {}).get("name"),
                slug,
            )
            endpoints = {
                "statistics": f"{API_URL}/event/{event_id}/statistics",
                "odds": f"{API_URL}/event/{event_id}/odds/1/all",
                "lineups": f"{API_URL}/event/{event_id}/lineups",
                "incidents": f"{API_URL}/event/{event_id}/incidents",
            }
            for name, url in endpoints.items():
                endpoint_status, endpoint_text = await fetch_json(context, url, page_url)
                print(f"{name} status: {endpoint_status} {endpoint_text[:500]}")
                if endpoint_status == 200:
                    (OUT_DIR / f"{event_id}_{name}.json").write_text(
                        endpoint_text,
                        encoding="utf-8",
                    )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

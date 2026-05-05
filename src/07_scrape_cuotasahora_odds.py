import argparse
import base64
import gzip
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import unquote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


BASE_DIR = Path(__file__).resolve().parents[1]
ODDS_DIR = BASE_DIR / "files" / "06_odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.cuotasahora.com"
DEFAULT_SOURCE_URL = (
    "https://www.cuotasahora.com/football/h2h/arsenal-hA1Zm19f/"
    "newcastle-p6ahwuwJ/#OQsq6PYa:over-under;2;"
)
LEAGUE_SOURCE_URLS = {
    "chile": "https://www.cuotasahora.com/football/chile/liga-de-primera/",
    "la_liga": "https://www.cuotasahora.com/football/spain/laliga-ea-sports/",
    "serie_a": "https://www.cuotasahora.com/football/italy/serie-a/",
    "premier": "https://www.cuotasahora.com/football/england/premier-league/",
}

PASSWORD = ("J*8sQ!p" + chr(36) + "7aD_fR2yW@gHn*3bVp#sAdLd_k").encode()
SALT = b"5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d"

MARKET_IDS = {
    "h2h": 1,
    "over_under": 2,
    "double_chance": 4,
    "btts": 13,
}

TEAM_ALIASES = {
    "leeds utd": "Leeds United",
    "newcastle": "Newcastle United",
    "manchester utd": "Manchester United",
    "nottingham": "Nottingham Forest",
    "wolves": "Wolves",
    "coquimbo": "Coquimbo Unido",
    "u catolica": "Universidad Catolica",
    "u. catolica": "Universidad Catolica",
    "u de chile": "Universidad de Chile",
    "u. de chile": "Universidad de Chile",
    "u concepcion": "Universidad de Concepcion",
    "u. concepcion": "Universidad de Concepcion",
    "dep. concepcion": "Deportes Concepcion",
    "limache": "CD Limache",
    "dep. limache": "CD Limache",
    "nublense": "Nublense",
    "colo colo": "Colo-Colo",
    "union la calera": "Union La Calera",
}


def clean_number(value):
    if value is None:
        return None
    try:
        return float(Decimal(str(value)))
    except Exception:
        return None


def normalize_event_team(name):
    if not isinstance(name, str):
        return name
    return TEAM_ALIASES.get(name.strip().lower(), name.strip())


def get_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        }
    )
    return session


def derive_key():
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=1000,
    )
    return kdf.derive(PASSWORD)


def decode_payload(payload):
    decoded = base64.b64decode(payload.strip()).decode("utf-8")
    encrypted_b64, iv_hex = decoded.split(":")
    cipher = Cipher(algorithms.AES(derive_key()), modes.CBC(bytes.fromhex(iv_hex)))
    decryptor = cipher.decryptor()
    padded = decryptor.update(base64.b64decode(encrypted_b64)) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plain = unpadder.update(padded) + unpadder.finalize()
    if plain[:2] == b"\x1f\x8b":
        plain = gzip.decompress(plain)
    return json.loads(plain.decode("utf-8"))


def extract_event_data(html):
    soup = BeautifulSoup(html, "html.parser")
    event_node = soup.find(id="react-event-header")
    if not event_node or not event_node.get("data"):
        raise RuntimeError("No encontre react-event-header con metadata del evento")
    data = json.loads(event_node["data"])
    event_data = data["eventData"]
    return {
        "event_id": event_data["id"],
        "version_id": int(event_data.get("versionId", 13)),
        "sport_id": int(event_data.get("sportId", 1)),
        "scope_id": int(event_data.get("defaultScopeId", 2)),
        "xhashf": unquote(event_data["xhashf"]),
        "home": normalize_event_team(event_data["home"]),
        "away": normalize_event_team(event_data["away"]),
        "related": extract_related_events(soup, event_data),
    }


def extract_related_events(soup, event_data):
    rows = []
    nodes = [soup.find(id="react-leagues-events")]
    tournament = event_data.get("tournamentGamesComponent", {})
    if tournament.get("rows"):
        for row in tournament["rows"]:
            rows.append(row)

    for node in nodes:
        if node and node.get("data"):
            rows.extend(json.loads(node["data"]).get("rows", []))

    seen = set()
    events = []
    for row in rows:
        url = row.get("url")
        event = row.get("event")
        if not url or not event or url in seen:
            continue
        seen.add(url)
        home, away = [normalize_event_team(x) for x in re.split(r"\s+-\s+", event, maxsplit=1)]
        events.append(
            {
                "home": home,
                "away": away,
                "url": urljoin(BASE_URL, url),
                "formatted_date": row.get("formattedDate", "").replace("&nbsp;", " "),
            }
        )
    return events


def extract_events_from_listing(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for node in soup.find_all(id="react-leagues-events"):
        if node.get("data"):
            rows.extend(json.loads(node["data"]).get("rows", []))

    events = []
    seen = set()
    for row in rows:
        url = row.get("url")
        event = row.get("event")
        if not url or not event or url in seen:
            continue
        if " - " in event:
            home, away = [normalize_event_team(x) for x in re.split(r"\s+-\s+", event, maxsplit=1)]
        else:
            home, away = "", ""
        seen.add(url)
        events.append(
            {
                "home": home,
                "away": away,
                "url": urljoin(BASE_URL, url),
                "formatted_date": row.get("formattedDate", "").replace("&nbsp;", " "),
            }
        )

    if events:
        return events

    for script in soup.find_all("script", {"type": "application/ld+json"}):
        text = script.string or script.get_text()
        try:
            payload = json.loads(text)
        except Exception:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            is_sports_event = item_type == "SportsEvent" or (
                isinstance(item_type, list) and "SportsEvent" in item_type
            )
            url = item.get("url")
            event = item.get("name")
            if not is_sports_event or not url or url in seen:
                continue
            home, away = "", ""
            if isinstance(event, str) and " - " in event:
                home, away = [
                    normalize_event_team(part)
                    for part in re.split(r"\s+-\s+", event, maxsplit=1)
                ]
            seen.add(url)
            events.append(
                {
                    "home": home,
                    "away": away,
                    "url": urljoin(BASE_URL, url),
                    "formatted_date": item.get("startDate", ""),
                }
            )

    if events:
        return events

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/football/h2h/" not in href or href in seen:
            continue
        seen.add(href)
        events.append(
            {
                "home": "",
                "away": "",
                "url": urljoin(BASE_URL, href),
                "formatted_date": "",
            }
        )
    return events


def discover_events_from_url(url, session=None):
    session = session or get_session()
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return extract_events_from_listing(response.text)


def market_endpoint(meta, market):
    return (
        f"{BASE_URL}/match-event/{meta['version_id']}-{meta['sport_id']}-"
        f"{meta['event_id']}-{MARKET_IDS[market]}-{meta['scope_id']}-{meta['xhashf']}.dat?_="
        f"{int(time.time() * 1000)}"
    )


def fetch_market(session, meta, market, referer):
    response = session.get(
        market_endpoint(meta, market),
        headers={"Referer": referer, "X-Requested-With": "XMLHttpRequest"},
        timeout=30,
    )
    response.raise_for_status()
    return decode_payload(response.text)


def iter_market_odds(market_data):
    odds = market_data.get("odds", {}) if isinstance(market_data, dict) else {}
    if isinstance(odds, dict):
        return odds.values()
    if isinstance(odds, list):
        return odds
    return []


def iter_back_markets(decoded_by_market, market):
    back = decoded_by_market.get(market, {}).get("d", {}).get("oddsdata", {}).get("back", {})
    if isinstance(back, dict):
        return back.values()
    if isinstance(back, list):
        return back
    return []


def best_from_mapping(market_data, key):
    values = []
    for bookmaker_odds in iter_market_odds(market_data):
        if isinstance(bookmaker_odds, dict):
            values.append(clean_number(bookmaker_odds.get(key)))
    values = [value for value in values if value]
    return max(values) if values else None


def best_from_list(market_data, index):
    values = []
    for bookmaker_odds in iter_market_odds(market_data):
        if isinstance(bookmaker_odds, list) and len(bookmaker_odds) > index:
            values.append(clean_number(bookmaker_odds[index]))
    values = [value for value in values if value]
    return max(values) if values else None


def parse_odds(meta, decoded_by_market):
    row = {
        "source_url": meta["url"],
        "local_team": meta["home"],
        "away_team": meta["away"],
    }

    h2h = next(iter(iter_back_markets(decoded_by_market, "h2h")), {})
    row["decimal_home_win"] = best_from_mapping(h2h, "0")
    row["decimal_draw"] = best_from_mapping(h2h, "1")
    row["decimal_away_win"] = best_from_mapping(h2h, "2")

    double = next(iter(iter_back_markets(decoded_by_market, "double_chance")), {})
    row["decimal_home_or_draw"] = best_from_mapping(double, "0")
    row["decimal_home_or_away"] = best_from_mapping(double, "1")
    row["decimal_draw_or_away"] = best_from_mapping(double, "2")

    for market_data in iter_back_markets(decoded_by_market, "over_under"):
        if not isinstance(market_data, dict):
            continue
        handicap = clean_number(market_data.get("handicapValue"))
        if handicap == 1.5:
            row["decimal_over_15"] = best_from_list(market_data, 0)
            row["decimal_under_15"] = best_from_list(market_data, 1)
        elif handicap == 2.5:
            row["decimal_over_25"] = best_from_list(market_data, 0)
            row["decimal_under_25"] = best_from_list(market_data, 1)

    btts = next(iter(iter_back_markets(decoded_by_market, "btts")), {})
    row["decimal_btts_yes"] = best_from_list(btts, 0)
    row["decimal_btts_no"] = best_from_list(btts, 1)
    return row


def scrape_event(url, session=None):
    session = session or get_session()
    response = session.get(url, timeout=30)
    response.raise_for_status()
    html = response.text
    meta = extract_event_data(html)
    meta["url"] = url

    decoded = {}
    for market in MARKET_IDS:
        try:
            decoded[market] = fetch_market(session, meta, market, url)
        except Exception as exc:
            decoded[market] = {"error": str(exc)}
    return parse_odds(meta, decoded), meta["related"]


def scrape_urls(urls):
    session = get_session()
    rows = []
    related = []
    failures = []
    for url in urls:
        try:
            row, event_related = scrape_event(url, session=session)
            rows.append(row)
            related.extend(event_related)
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"No pude scrapear {url}: {exc}")
    related_df = pd.DataFrame(related).drop_duplicates() if related else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    return pd.DataFrame(rows), related_df, failures_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--league", choices=sorted(LEAGUE_SOURCE_URLS), default=None)
    parser.add_argument("--discover-from", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out", default=None)
    parser.add_argument("--related-out", default=None)
    args = parser.parse_args()

    discover_from = args.discover_from or (
        LEAGUE_SOURCE_URLS[args.league] if args.league else DEFAULT_SOURCE_URL
    )
    out_path = args.out or str(
        ODDS_DIR / (
            f"cuotasahora_{args.league}_consolidated.csv"
            if args.league
            else "cuotasahora_matchweek35_sample.csv"
        )
    )
    related_out = args.related_out or str(
        ODDS_DIR / (
            f"cuotasahora_{args.league}_related.csv"
            if args.league
            else "cuotasahora_related_events.csv"
        )
    )

    urls = args.url
    related = pd.DataFrame()
    failures = pd.DataFrame()
    if not urls:
        try:
            discovered = pd.DataFrame(discover_events_from_url(discover_from))
            failures = pd.DataFrame()
        except Exception:
            _, discovered, failures = scrape_urls([discover_from])
        related = discovered
        if discovered.empty or "url" not in discovered.columns:
            raise RuntimeError("No pude descubrir URLs relacionadas desde la pagina semilla")
        urls = discovered["url"].head(args.limit).tolist()

    odds, more_related, more_failures = scrape_urls(urls)
    failures = pd.concat([failures, more_failures], ignore_index=True)
    if related.empty:
        related = more_related

    odds.to_csv(out_path, index=False)
    if not related.empty:
        related.to_csv(related_out, index=False)
    if not failures.empty:
        failures.to_csv(ODDS_DIR / "cuotasahora_scrape_failures.csv", index=False)

    print(f"Cuotas guardadas en: {out_path}")
    print(odds.to_string(index=False) if not odds.empty else "Sin cuotas decodificadas")


if __name__ == "__main__":
    main()

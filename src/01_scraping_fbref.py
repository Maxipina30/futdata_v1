# %%
import os
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ---------------- Config ----------------
BASE_URL = "https://fbref.com"
COMP_ID = os.getenv("FUTDATA_FBREF_COMP_ID", "9")
SEASON = os.getenv("FUTDATA_SEASON", "2025-2026")
LEAGUE_NAME = os.getenv("FUTDATA_FBREF_LEAGUE_NAME", "Premier-League")
DEFAULT_LEAGUE_URL = f"{BASE_URL}/en/comps/{COMP_ID}/{SEASON}/{SEASON}-{LEAGUE_NAME}-Stats"
LEAGUE_URL = os.getenv("FUTDATA_FBREF_LEAGUE_URL", DEFAULT_LEAGUE_URL)
LEAGUE_KEY = os.getenv("FUTDATA_LEAGUE_KEY", "premier")
OUTPUT_DIR = os.path.join("files", "01_raw", LEAGUE_KEY)
CACHE_DIR = os.path.join("files", "00_cache", "fbref", LEAGUE_KEY, SEASON)
LOCAL_HTML_PATH = os.getenv("FUTDATA_FBREF_HTML_PATH", os.path.join("files", "Premier League Stats _ FBref.com.html"))
SCORES_HTML_PATH = os.getenv(
    "FUTDATA_FBREF_SCORES_HTML_PATH",
    os.path.join("files", "Premier League Scores & Fixtures _ FBref.com.html"),
)
FETCH_MATCHLOGS = os.getenv(
    "FUTDATA_FETCH_MATCHLOGS",
    "0" if os.path.exists(LOCAL_HTML_PATH) else "1",
) == "1"
COMP_NAME = os.getenv("FUTDATA_COMP_NAME", "Premier League")
CONTACT_EMAIL = os.getenv("FUTDATA_CONTACT_EMAIL", "contact@example.com")
REQUEST_DELAY_SECONDS = float(os.getenv("FUTDATA_REQUEST_DELAY_SECONDS", "7"))
USE_BROWSER_FALLBACK = os.getenv("FUTDATA_USE_BROWSER_FALLBACK", "1") == "1"
BROWSER_WAIT_SECONDS = float(os.getenv("FUTDATA_BROWSER_WAIT_SECONDS", "300"))
BROWSER_CHANNEL = os.getenv("FUTDATA_BROWSER_CHANNEL", "msedge")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "futdata_v1/0.1 "
        f"(research project; contact: {CONTACT_EMAIL}) "
        "requests"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_html(url):
    cache_path = os.path.join(CACHE_DIR, safe_filename(url) + ".html")
    local_html = Path(LOCAL_HTML_PATH)
    if local_html.exists() and url == LEAGUE_URL:
        print(f"Usando HTML local guardado: {local_html}")
        return local_html.read_text(encoding="utf-8", errors="ignore")

    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code == 403 and "Just a moment" in response.text:
        if USE_BROWSER_FALLBACK:
            html = get_html_with_browser(url)
            if html and "Just a moment" not in html and "<table" in html:
                with open(cache_path, "w", encoding="utf-8") as cache_file:
                    cache_file.write(html)
                return html
            if html:
                blocked_path = os.path.join(CACHE_DIR, safe_filename(url) + "_blocked.html")
                with open(blocked_path, "w", encoding="utf-8") as blocked_file:
                    blocked_file.write(html)
                print(f"HTML bloqueado guardado para diagnostico: {blocked_path}")

        if os.path.exists(cache_path):
            print(f"FBref bloqueo la descarga; usando cache local: {cache_path}")
            with open(cache_path, encoding="utf-8") as cache_file:
                return cache_file.read()
        raise RuntimeError(
            "FBref devolvio Cloudflare 403 ('Just a moment...'). "
            "No se intenta evadir Cloudflare. Usa una API/acceso formal, una sesion "
            "autorizada o una cache local creada desde una descarga permitida."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "FBref devolvio 429 rate limit. Baja FUTDATA_REQUEST_DELAY_SECONDS "
            "o espera antes de reintentar."
        )
    response.raise_for_status()
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        cache_file.write(response.text)
    return response.text


def get_html_with_browser(url):
    print("HTTP directo bloqueado. Probando navegador real con Playwright...")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright no esta instalado; no se puede usar fallback de navegador.")
        return None

    profile_dir = os.path.abspath(os.path.join(CACHE_DIR, "browser_profile"))
    os.makedirs(profile_dir, exist_ok=True)

    with sync_playwright() as p:
        launch_options = {
            "user_data_dir": profile_dir,
            "headless": False,
            "viewport": {"width": 1366, "height": 900},
            "locale": "en-US",
        }
        if BROWSER_CHANNEL:
            launch_options["channel"] = BROWSER_CHANNEL
        try:
            context = p.chromium.launch_persistent_context(**launch_options)
        except Exception as exc:
            if not BROWSER_CHANNEL:
                raise
            print(f"No se pudo abrir canal {BROWSER_CHANNEL} ({exc}); usando Chromium de Playwright.")
            launch_options.pop("channel", None)
            context = p.chromium.launch_persistent_context(**launch_options)

        page = context.pages[0] if context.pages else context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            print(
                "Si aparece Cloudflare, completa la validacion en la ventana. "
                f"Esperando hasta {BROWSER_WAIT_SECONDS:.0f}s por tablas..."
            )
            deadline = time.time() + BROWSER_WAIT_SECONDS
            table_count = 0
            while time.time() < deadline:
                table_count = page.locator("table").count()
                title = page.title()
                if table_count > 0 and "Just a moment" not in title:
                    break
                page.wait_for_timeout(2000)

            title = page.title()
            html = page.content()
            table_count = page.locator("table").count()
            print(f"Playwright title: {title}")
            print(f"Tablas visibles: {table_count}")
            return html
        except Exception as exc:
            print(f"Playwright no completo la validacion ({exc})")
            try:
                return page.content()
            except Exception:
                return None
        finally:
            try:
                context.close()
            except Exception:
                pass


def safe_filename(url):
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:180]


def preflight_access_check():
    robots_url = f"{BASE_URL}/robots.txt"
    print(f"User-Agent: {HEADERS['User-Agent']}")
    print(f"Delay entre requests: {REQUEST_DELAY_SECONDS:.1f}s")
    try:
        response = requests.get(robots_url, headers=HEADERS, timeout=20)
        print(f"robots.txt: HTTP {response.status_code}")
        if response.ok:
            disallow_lines = [
                line.strip()
                for line in response.text.splitlines()
                if line.lower().startswith(("user-agent:", "disallow:", "crawl-delay:"))
            ]
            for line in disallow_lines[:30]:
                print(f"  {line}")
    except requests.RequestException as exc:
        print(f"robots.txt: no accesible ({exc})")


def soup_with_commented_tables(html):
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            fragment = BeautifulSoup(comment, "html.parser")
            for table in fragment.find_all("table"):
                soup.append(table)
    return soup


def read_table(table):
    return pd.read_html(StringIO(str(table)))[0]


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part) and str(part) != "nan").strip("_")
            for col in df.columns
        ]
    return df


def table_by_id(soup, table_id):
    table = soup.find("table", id=table_id)
    if table is None:
        return None
    return flatten_columns(read_table(table))


def first_table_matching(soup, pattern):
    regex = re.compile(pattern)
    for table in soup.find_all("table"):
        table_id = table.get("id", "")
        if regex.search(table_id):
            return flatten_columns(read_table(table))
    return None


def save_csv(df, filename, compatible_name=None):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Guardado: {path} ({len(df)} filas)")

    if compatible_name:
        compatible_path = os.path.join("files", "01_raw", compatible_name)
        df.to_csv(compatible_path, index=False)
        print(f"Guardado compatibilidad: {compatible_path} ({len(df)} filas)")


def find_main_tables(soup):
    standings = first_table_matching(soup, r"results.*overall")
    home_away = first_table_matching(soup, r"results.*home_away")
    stats_for = table_by_id(soup, "stats_squads_standard_for")
    stats_against = table_by_id(soup, "stats_squads_standard_against")

    missing = [
        name
        for name, df in {
            "standings": standings,
            "home_away": home_away,
            "stats_for": stats_for,
            "stats_against": stats_against,
        }.items()
        if df is None
    ]
    if missing:
        raise RuntimeError(f"No se encontraron tablas principales de FBref: {', '.join(missing)}")

    return standings, home_away, stats_for, stats_against


def save_extra_squad_tables(soup):
    wanted = {
        "keeper_for": "stats_squads_keeper_for",
        "shooting_for": "stats_squads_shooting_for",
        "passing_for": "stats_squads_passing_for",
        "passing_types_for": "stats_squads_passing_types_for",
        "gca_for": "stats_squads_gca_for",
        "defense_for": "stats_squads_defense_for",
        "possession_for": "stats_squads_possession_for",
        "playing_time_for": "stats_squads_playing_time_for",
        "misc_for": "stats_squads_misc_for",
        "keeper_against": "stats_squads_keeper_against",
        "shooting_against": "stats_squads_shooting_against",
        "passing_against": "stats_squads_passing_against",
        "passing_types_against": "stats_squads_passing_types_against",
        "gca_against": "stats_squads_gca_against",
        "defense_against": "stats_squads_defense_against",
        "possession_against": "stats_squads_possession_against",
        "playing_time_against": "stats_squads_playing_time_against",
        "misc_against": "stats_squads_misc_against",
    }

    for name, table_id in wanted.items():
        df = table_by_id(soup, table_id)
        if df is not None:
            save_csv(df, f"{LEAGUE_KEY}_{name}.csv")


def read_local_scores_fixtures():
    scores_html = Path(SCORES_HTML_PATH)
    if not scores_html.exists():
        return None

    print(f"Usando HTML local de fixtures: {scores_html}")
    html = scores_html.read_text(encoding="utf-8", errors="ignore")
    soup = soup_with_commented_tables(html)
    table = soup.find("table", id=re.compile(r"^sched_"))
    if table is None:
        raise RuntimeError(f"No se encontro tabla sched_* en {scores_html}")

    return flatten_columns(read_table(table))


def build_matchlogs_from_schedule(schedule, stats_for):
    poss_by_team = build_possession_lookup(stats_for)
    rows = []
    schedule = schedule.dropna(subset=["Date", "Home", "Away"], how="any").copy()
    schedule["Date"] = pd.to_datetime(schedule["Date"], errors="coerce")
    schedule = schedule.dropna(subset=["Date"])
    schedule["Date"] = schedule["Date"].dt.date.astype(str)
    schedule = schedule[schedule["Home"].astype(str).str.lower() != "home"]
    schedule = schedule[schedule["Away"].astype(str).str.lower() != "away"]

    for _, match in schedule.iterrows():
        home = match["Home"]
        away = match["Away"]
        score = parse_score(match.get("Score"))
        home_gf, home_ga, away_gf, away_ga = (pd.NA, pd.NA, pd.NA, pd.NA)
        home_result = away_result = pd.NA

        if score:
            home_gf, home_ga = score
            away_gf, away_ga = home_ga, home_gf
            home_result = result_from_goals(home_gf, home_ga)
            away_result = result_from_goals(away_gf, away_ga)

        rows.append(
            schedule_row_to_matchlog(
                match, team=home, opponent=away, venue="Home",
                gf=home_gf, ga=home_ga, result=home_result,
                poss=poss_by_team.get(home, pd.NA),
            )
        )
        rows.append(
            schedule_row_to_matchlog(
                match, team=away, opponent=home, venue="Away",
                gf=away_gf, ga=away_ga, result=away_result,
                poss=poss_by_team.get(away, pd.NA),
            )
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["Date", "Time", "Equipo"]).reset_index(drop=True)


def build_possession_lookup(stats_for):
    team_col = find_column(stats_for, "squad")
    poss_col = find_column(stats_for, "poss")
    if not team_col or not poss_col:
        return {}

    poss = stats_for[[team_col, poss_col]].copy()
    poss[poss_col] = pd.to_numeric(poss[poss_col], errors="coerce")
    return dict(zip(poss[team_col], poss[poss_col]))


def find_column(df, needle):
    needle = needle.lower()
    for col in df.columns:
        if str(col).lower().endswith(needle) or needle in str(col).lower().split("_"):
            return col
    return None


def parse_score(score):
    if pd.isna(score):
        return None

    match = re.search(r"(\d+)\s*[–-]\s*(\d+)", str(score))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def result_from_goals(gf, ga):
    if gf > ga:
        return "W"
    if gf == ga:
        return "D"
    return "L"


def schedule_row_to_matchlog(match, team, opponent, venue, gf, ga, result, poss):
    date = match.get("Date")
    wk = match.get("Wk")
    return {
        "Date": date,
        "Time": clean_value(match.get("Time")),
        "Comp": COMP_NAME,
        "Round": format_round(wk),
        "Day": clean_value(match.get("Day")),
        "Venue": venue,
        "Result": result,
        "GF": gf,
        "GA": ga,
        "Opponent": opponent,
        "Poss": poss,
        "Attendance": clean_value(match.get("Attendance")),
        "Captain": "",
        "Formation": "",
        "Opp Formation": "",
        "Referee": clean_value(match.get("Referee")),
        "Match Report": clean_value(match.get("Match Report")),
        "Notes": clean_value(match.get("Notes")),
        "Equipo": team,
        "xG": "",
        "xGA": "",
    }


def clean_value(value):
    if pd.isna(value):
        return ""
    return value


def format_round(wk):
    if pd.isna(wk):
        return ""
    try:
        return f"Matchweek {int(float(wk))}"
    except (TypeError, ValueError):
        return str(wk)


def get_team_schedule_links(soup):
    links = {}
    standings_table = soup.find("table", id=re.compile(r"results.*overall"))
    search_root = standings_table if standings_table is not None else soup

    for link in search_root.select("a[href*='/squads/']"):
        team = link.get_text(strip=True)
        href = link.get("href", "")
        if not team or "/squads/" not in href:
            continue

        parts = href.strip("/").split("/")
        squad_id = parts[2] if len(parts) > 2 else None
        if not squad_id:
            continue

        schedule_url = f"{BASE_URL}/en/squads/{squad_id}/{SEASON}/matchlogs/c{COMP_ID}/schedule/"
        links[team] = schedule_url

    return links


def read_team_matches(team, url):
    html = get_html(url)
    soup = soup_with_commented_tables(html)
    table = first_table_matching(soup, r"matchlogs.*schedule|sched")
    if table is None:
        raise RuntimeError("no se encontro tabla schedule")

    table["Equipo"] = team
    return table


def read_all_matches(team_links):
    all_matches = []
    for team, url in team_links.items():
        try:
            matches = read_team_matches(team, url)
            all_matches.append(matches)
            print(f"{team}: {len(matches)} partidos")
        except Exception as exc:
            print(f"ADVERTENCIA {team}: {exc}")
        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_matches:
        raise RuntimeError("No se obtuvo ningun matchlog de equipos")

    return pd.concat(all_matches, ignore_index=True)


def main():
    print(f"Descargando Premier League desde FBref: {LEAGUE_URL}")
    preflight_access_check()
    html = get_html(LEAGUE_URL)
    soup = soup_with_commented_tables(html)

    standings, home_away, stats_for, stats_against = find_main_tables(soup)
    save_csv(standings, f"{LEAGUE_KEY}_standings.csv")
    save_csv(home_away, f"{LEAGUE_KEY}_home_away.csv")
    save_csv(stats_for, f"{LEAGUE_KEY}_stats_for.csv")
    save_csv(stats_against, f"{LEAGUE_KEY}_stats_against.csv")
    save_extra_squad_tables(soup)

    schedule = read_local_scores_fixtures()
    if schedule is not None:
        save_csv(schedule, f"{LEAGUE_KEY}_scores_fixtures.csv")
        matches = build_matchlogs_from_schedule(schedule, stats_for)
        save_csv(matches, f"{LEAGUE_KEY}_partidos.csv")
        print("01_scraping_fbref completado con stats y fixtures locales.")
        return

    if not FETCH_MATCHLOGS:
        print("Matchlogs omitidos: usando HTML local de la pagina principal.")
        print("Para intentar descargar matchlogs por equipo, define FUTDATA_FETCH_MATCHLOGS=1.")
        print("01_scraping_fbref completado con tablas de liga.")
        return

    team_links = get_team_schedule_links(soup)
    print(f"Equipos detectados: {len(team_links)}")
    if len(team_links) < 20:
        raise RuntimeError(f"Se esperaban al menos 20 equipos, se detectaron {len(team_links)}")

    matches = read_all_matches(team_links)
    save_csv(matches, f"{LEAGUE_KEY}_partidos.csv")
    print("01_scraping_fbref completado.")


if __name__ == "__main__":
    main()


# %%

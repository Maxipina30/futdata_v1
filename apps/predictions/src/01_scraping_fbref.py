import os
import re
import time
from io import StringIO
from pathlib import Path
 
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
 
# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
BASE_URL = "https://fbref.com"
COMP_ID = os.getenv("FUTDATA_FBREF_COMP_ID", "9")
SEASON = os.getenv("FUTDATA_SEASON", "2025-2026")
LEAGUE_NAME = os.getenv("FUTDATA_FBREF_LEAGUE_NAME", "Premier-League")
DEFAULT_LEAGUE_URL = f"{BASE_URL}/en/comps/{COMP_ID}/{SEASON}/{SEASON}-{LEAGUE_NAME}-Stats"
LEAGUE_URL = os.getenv("FUTDATA_FBREF_LEAGUE_URL", DEFAULT_LEAGUE_URL)
LEAGUE_KEY = os.getenv("FUTDATA_LEAGUE_KEY", "premier")
OUTPUT_DIR = BASE_DIR / "files" / "01_raw" / LEAGUE_KEY
CACHE_DIR = BASE_DIR / "files" / "00_cache" / "fbref" / LEAGUE_KEY / SEASON
CONTACT_EMAIL = os.getenv("FUTDATA_CONTACT_EMAIL", "contact@example.com")
REQUEST_DELAY_SECONDS = float(os.getenv("FUTDATA_REQUEST_DELAY_SECONDS", "7"))
USE_BROWSER_FALLBACK = os.getenv("FUTDATA_USE_BROWSER_FALLBACK", "1") == "1"
BROWSER_WAIT_SECONDS = float(os.getenv("FUTDATA_BROWSER_WAIT_SECONDS", "60"))
# Opciones: "msedge", "chrome", "" (Chromium de Playwright)
BROWSER_CHANNEL = os.getenv("FUTDATA_BROWSER_CHANNEL", "")
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
 
 
# ─────────────────────────────────────────────
# Detección de bloqueo Cloudflare (más robusta)
# ─────────────────────────────────────────────
def _is_cloudflare_blocked(status_code: int, text: str) -> bool:
    """Detecta bloqueo de Cloudflare independientemente del status code."""
    if status_code in (403, 429, 503):
        return True
    cloudflare_markers = [
        "Just a moment",
        "cf-browser-verification",
        "Checking your browser",
        "cf_clearance",
        "Ray ID",
        "cloudflare",
        "Enable JavaScript and cookies",
    ]
    text_lower = text.lower()
    return any(m.lower() in text_lower for m in cloudflare_markers)
 
 
def _html_has_useful_content(html: str) -> bool:
    return "<table" in html and "Just a moment" not in html
 
 
# ─────────────────────────────────────────────
# Descarga principal con fallback a Playwright
# ─────────────────────────────────────────────
def get_html(url: str) -> str:
    cache_path = os.path.join(CACHE_DIR, safe_filename(url) + ".html")
 
    # 1. Intentar requests directo
    html = _try_requests(url)
 
    if html is not None and _html_has_useful_content(html):
        _save_cache(cache_path, html)
        return html
 
    # 2. Cloudflare detectado → Playwright
    if USE_BROWSER_FALLBACK:
        print("Cloudflare detectado. Activando fallback con Playwright...")
        html_browser = get_html_with_browser(url)
 
        if html_browser and _html_has_useful_content(html_browser):
            _save_cache(cache_path, html_browser)
            return html_browser
 
        if html_browser:
            blocked_path = os.path.join(CACHE_DIR, safe_filename(url) + "_blocked.html")
            _save_cache(blocked_path, html_browser)
            print(f"HTML bloqueado guardado para diagnóstico: {blocked_path}")
 
    # 3. Usar caché local si existe
    if os.path.exists(cache_path):
        print(f"Usando caché local: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return f.read()
 
    raise RuntimeError(
        "FBref bloqueó la descarga con Cloudflare y no hay caché disponible.\n"
        "Opciones:\n"
        "  - Instala Playwright: pip install playwright && playwright install chromium\n"
        "  - Sube manualmente el HTML descargado al directorio de caché\n"
        "  - Usa una VPN o proxy residencial\n"
        f"  URL: {url}"
    )
 
 
def _try_requests(url: str):
    """Intenta descarga directa. Devuelve HTML o None si falla/es bloqueado."""
    try:
        session = requests.Session()
        # Primera visita a la home para obtener cookies (simula navegación humana)
        try:
            session.get(BASE_URL, headers=HEADERS, timeout=15)
            time.sleep(1.5)
        except Exception:
            pass
 
        response = session.get(url, headers=HEADERS, timeout=30)
 
        if response.status_code == 429:
            raise RuntimeError(
                "FBref devolvió 429 (rate limit). "
                "Sube FUTDATA_REQUEST_DELAY_SECONDS o espera antes de reintentar."
            )
 
        if _is_cloudflare_blocked(response.status_code, response.text):
            return None  # señal para activar fallback
 
        response.raise_for_status()
        return response.text
 
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"requests falló ({exc}), intentando fallback...")
        return None
 
 
def _save_cache(path: str, html: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    except OSError as exc:
        print(f"No se pudo guardar caché en {path}: {exc}")
 
 
# ─────────────────────────────────────────────
# Fallback con Playwright  (fix de los bugs)
# ─────────────────────────────────────────────
def get_html_with_browser(url: str):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright no está instalado.\n"
            "Instálalo con: pip install playwright && playwright install chromium"
        )
        return None
 
    # Aplicar stealth si está disponible (evita detección de bot)
    _has_stealth = _check_stealth()
 
    profile_dir = os.path.abspath(os.path.join(CACHE_DIR, "browser_profile"))
    os.makedirs(profile_dir, exist_ok=True)
 
    with sync_playwright() as p:
        # Opciones de lanzamiento
        launch_kwargs = {
            "user_data_dir": profile_dir,
            "headless": False,          # headless=True es más detectable
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--window-size=1366,900",
            ],
        }
 
        if BROWSER_CHANNEL:
            launch_kwargs["channel"] = BROWSER_CHANNEL
 
        try:
            context = p.chromium.launch_persistent_context(**launch_kwargs)
        except Exception as exc:
            print(f"No se pudo abrir canal '{BROWSER_CHANNEL}' ({exc}); usando Chromium de Playwright.")
            launch_kwargs.pop("channel", None)
            try:
                context = p.chromium.launch_persistent_context(**launch_kwargs)
            except Exception as exc2:
                # Último recurso: launch normal (sin perfil persistente)
                print(f"launch_persistent_context falló ({exc2}); intentando launch normal...")
                browser = p.chromium.launch(
                    headless=False,
                    args=launch_kwargs["args"],
                )
                context = browser.new_context(
                    viewport={"width": 1366, "height": 900},
                    locale="en-US",
                    user_agent=HEADERS["User-Agent"],
                )
 
        page = context.pages[0] if context.pages else context.new_page()
 
        # Aplicar stealth si está disponible
        if _has_stealth:
            try:
                from playwright_stealth import stealth_sync
                stealth_sync(page)
                print("playwright-stealth activado.")
            except Exception as exc:
                print(f"playwright-stealth no pudo aplicarse: {exc}")
 
        # Ocultar webdriver manualmente como fallback de stealth
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)
 
        try:
            print(f"Navegando a: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
 
            deadline = time.time() + BROWSER_WAIT_SECONDS
            while time.time() < deadline:
                title = page.title()
                table_count = page.locator("table").count()
 
                if table_count > 0 and "Just a moment" not in title:
                    print(f"✓ Tablas encontradas: {table_count} | Título: {title}")
                    break
 
                remaining = int(deadline - time.time())
                print(
                    f"  Esperando tablas... título='{title}' "
                    f"tablas={table_count} tiempo restante={remaining}s"
                )
 
                # Si Cloudflare muestra CAPTCHA, dar tiempo al usuario para resolverlo
                if "Just a moment" in title or "Verify" in title:
                    page.wait_for_timeout(3_000)
                else:
                    page.wait_for_timeout(2_000)
 
            html = page.content()
            return html
 
        except Exception as exc:
            print(f"Playwright encontró un error: {exc}")
            try:
                return page.content()
            except Exception:
                return None
        finally:
            try:
                context.close()
            except Exception:
                pass
 
 
def _check_stealth() -> bool:
    """Devuelve True si playwright-stealth está instalado."""
    try:
        import playwright_stealth  # noqa: F401
        return True
    except ImportError:
        print(
            "playwright-stealth no está instalado (opcional pero recomendado).\n"
            "Instálalo con: pip install playwright-stealth"
        )
        return False
 
 
# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────
def safe_filename(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:180]
 
 
def preflight_access_check():
    robots_url = f"{BASE_URL}/robots.txt"
    print(f"User-Agent: {HEADERS['User-Agent']}")
    print(f"Delay entre requests: {REQUEST_DELAY_SECONDS:.1f}s")
    try:
        response = requests.get(robots_url, headers=HEADERS, timeout=20)
        print(f"robots.txt: HTTP {response.status_code}")
        if response.ok:
            lines = [
                line.strip()
                for line in response.text.splitlines()
                if line.lower().startswith(("user-agent:", "disallow:", "crawl-delay:"))
            ]
            for line in lines[:30]:
                print(f"  {line}")
    except requests.RequestException as exc:
        print(f"robots.txt: no accesible ({exc})")
 
 
def soup_with_commented_tables(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            fragment = BeautifulSoup(comment, "html.parser")
            for table in fragment.find_all("table"):
                soup.append(table)
    return soup
 
 
def read_table(table) -> pd.DataFrame:
    return pd.read_html(StringIO(str(table)))[0]
 
 
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(p) for p in col if str(p) and str(p) != "nan").strip("_")
            for col in df.columns
        ]
    return df
 
 
def table_by_id(soup: BeautifulSoup, table_id: str):
    table = soup.find("table", id=table_id)
    return flatten_columns(read_table(table)) if table is not None else None
 
 
def first_table_matching(soup: BeautifulSoup, pattern: str):
    regex = re.compile(pattern)
    for table in soup.find_all("table"):
        if regex.search(table.get("id", "")):
            return flatten_columns(read_table(table))
    return None
 
 
def save_csv(df: pd.DataFrame, filename: str, compatible_name: str = None):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Guardado: {path} ({len(df)} filas)")
    if compatible_name:
        alt = BASE_DIR / "files" / "01_raw" / compatible_name
        df.to_csv(alt, index=False)
        print(f"Guardado compatibilidad: {alt} ({len(df)} filas)")
 
 
# ─────────────────────────────────────────────
# Tablas principales
# ─────────────────────────────────────────────
def find_main_tables(soup: BeautifulSoup):
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
        raise RuntimeError(f"No se encontraron tablas: {', '.join(missing)}")
 
    return standings, home_away, stats_for, stats_against
 
 
def save_extra_squad_tables(soup: BeautifulSoup):
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
 
 
# ─────────────────────────────────────────────
# Matchlogs por equipo
# ─────────────────────────────────────────────
def get_team_schedule_links(soup: BeautifulSoup) -> dict:
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
 
 
def read_team_matches(team: str, url: str) -> pd.DataFrame:
    html = get_html(url)
    soup = soup_with_commented_tables(html)
    table = first_table_matching(soup, r"matchlogs.*schedule|sched")
    if table is None:
        raise RuntimeError("no se encontró tabla schedule")
    table["Equipo"] = team
    return table
 
 
def read_all_matches(team_links: dict) -> pd.DataFrame:
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
        raise RuntimeError("No se obtuvo ningún matchlog de equipos")
 
    return pd.concat(all_matches, ignore_index=True)
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"Descargando {LEAGUE_NAME} desde FBref: {LEAGUE_URL}")
    preflight_access_check()
 
    html = get_html(LEAGUE_URL)
    soup = soup_with_commented_tables(html)
 
    standings, home_away, stats_for, stats_against = find_main_tables(soup)
    save_csv(standings, f"{LEAGUE_KEY}_standings.csv", "chile_standings.csv")
    save_csv(home_away, f"{LEAGUE_KEY}_home_away.csv", "chile_home_away.csv")
    save_csv(stats_for, f"{LEAGUE_KEY}_stats_for.csv", "chile_stats_for.csv")
    save_csv(stats_against, f"{LEAGUE_KEY}_stats_against.csv", "chile_stats_against.csv")
    save_extra_squad_tables(soup)
 
    team_links = get_team_schedule_links(soup)
    print(f"Equipos detectados: {len(team_links)}")
    if len(team_links) < 10:
        raise RuntimeError(f"Se esperaban al menos 10 equipos, se detectaron {len(team_links)}")
 
    matches = read_all_matches(team_links)
    save_csv(matches, f"{LEAGUE_KEY}_partidos.csv", "chile_partidos.csv")
    print("Scraping completado.")
 
 
if __name__ == "__main__":
    main()

# %%

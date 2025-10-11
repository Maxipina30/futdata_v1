# %%
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
from io import StringIO
import time
import os

# --- Configuración base ---
BASE_URL = "https://fbref.com"
LEAGUE_URL = f"{BASE_URL}/en/comps/35/Chilean-Primera-Division-Stats"
OUTPUT_DIR = "files/01_raw"

# Crear carpeta si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_soup(url):
    """Obtiene HTML evitando bloqueos de Cloudflare."""
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    html = scraper.get(url).text
    return BeautifulSoup(html, "html.parser")

def read_table(soup, keywords, filename):
    """Busca y guarda tabla que contiene las keywords en su ID."""
    for t in soup.find_all("table"):
        tid = t.get("id", "")
        if any(k.lower() in tid.lower() for k in keywords):
            df = pd.read_html(StringIO(str(t)))[0]
            path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"💾 Guardado: {path} ({df.shape[0]} filas)")
            return df
    print(f"⚠️ No se encontró tabla con palabras {keywords}")
    return pd.DataFrame()

def get_team_links(soup):
    """Obtiene URLs de match logs de cada equipo (versión 2025)."""
    links = []
    for a in soup.select("table#results2025351_overall a[href*='/squads/']"):
        team_name = a.text.strip()
        href = a["href"]
        team_id = href.split("/")[3]
        team_url = f"{BASE_URL}/en/squads/{team_id}/2025/matchlogs/all_comps/"
        links.append((team_name, team_url))
    return links

def get_matchlogs(team_name, url):
    """Scrapea los partidos de un equipo desde su página de Match Logs."""
    print(f"   ⚽ {team_name} ...", end="")
    soup = get_soup(url)
    table = None
    for t in soup.find_all("table"):
        tid = t.get("id", "")
        if "matchlogs" in tid.lower() or "schedule" in tid.lower():
            table = t
            break
    if table is None:
        print(" sin datos.")
        return pd.DataFrame()
    df = pd.read_html(StringIO(str(table)))[0]
    df["Equipo"] = team_name
    print(f" {df.shape[0]} partidos.")
    return df

def main():
    print("🇨🇱 Descargando datos del Campeonato Chileno desde FBref...\n")
    soup = get_soup(LEAGUE_URL)

    # --- Tablas principales ---
    standings = read_table(soup, ["overall"], "chile_standings.csv")
    home_away = read_table(soup, ["home_away"], "chile_home_away.csv")
    stats_for = read_table(soup, ["standard_for"], "chile_stats_for.csv")
    stats_against = read_table(soup, ["standard_against"], "chile_stats_against.csv")

    # --- Match logs (partidos) ---
    print("\n📅 Descargando partidos por equipo...")
    team_links = get_team_links(soup)
    print(f"Se encontraron {len(team_links)} equipos.")

    all_matches = []
    for name, url in team_links:
        df_team = get_matchlogs(name, url)
        if not df_team.empty:
            all_matches.append(df_team)
        time.sleep(1.5)  # pausa para evitar bloqueo

    if all_matches:
        matches = pd.concat(all_matches, ignore_index=True)
        path = os.path.join(OUTPUT_DIR, "chile_partidos.csv")
        matches.to_csv(path, index=False)
        print(f"💾 Guardado: {path} ({matches.shape[0]} filas)")
    else:
        print("⚠️ No se obtuvieron partidos.")

    print("\n🏁 Proceso finalizado correctamente 🇨🇱")
    print("Archivos generados en files/01_raw/:")
    for f in [
        "chile_standings.csv",
        "chile_home_away.csv",
        "chile_stats_for.csv",
        "chile_stats_against.csv",
        "chile_partidos.csv",
    ]:
        print(f" - {os.path.join(OUTPUT_DIR, f)}")

if __name__ == "__main__":
    main()


# %%




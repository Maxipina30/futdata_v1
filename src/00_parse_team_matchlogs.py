import argparse
import os
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment


BASE_DIR = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
FILES_DIR = BASE_DIR / "files"
RAW_DIR = FILES_DIR / "01_raw" / "premier"
OUT_DIR = RAW_DIR / "team_matchlogs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HTML_SEARCH_DIRS = [
    FILES_DIR,
    OUT_DIR,
]

COMPETITION = "Premier League"

LEAGUE_COMPETITIONS = {
    "premier": "Premier League",
    "la_liga": "La Liga",
    "serie_a": "Serie A",
    "chile": "Liga de Primera",
}


def soup_with_commented_tables(html):
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" in comment:
            fragment = BeautifulSoup(comment, "html.parser")
            for table in fragment.find_all("table"):
                soup.append(table)
    return soup


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part) and str(part) != "nan").strip("_")
            for col in df.columns
        ]
    return df


def normalize_column_name(col):
    col = str(col).strip()
    col = re.sub(r"^(For|Against)\s+[^_]+_", "", col)
    col = re.sub(r"^Standard_", "", col)
    col = re.sub(r"\s+", "_", col)
    col = col.replace("%", "pct")
    col = col.replace("/", "_per_")
    col = re.sub(r"[^\w]+", "_", col)
    return col.strip("_").lower()


def clean_team_name(value):
    if not isinstance(value, str):
        return value
    value = re.sub(r"\s+", " ", value.strip())
    aliases = {
        "Brighton & Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Wolverhampton Wanderers": "Wolves",
    }
    return aliases.get(value, value)


def infer_team_name(path, soup):
    title = soup.find("title")
    title_text = title.get_text(" ", strip=True) if title else path.stem

    patterns = [
        r"^(.+?)\s+Match Logs",
        r"^(.+?)\s+Scores and Fixtures",
        r"^(.+?)\s+Stats,",
    ]
    for pattern in patterns:
        match = re.search(pattern, title_text)
        if match:
            return clean_team_name(match.group(1))

    return clean_team_name(path.stem.split("_")[0])


def infer_page_kind(path, soup, df):
    text = " ".join(
        [
            path.name,
            soup.find("title").get_text(" ", strip=True) if soup.find("title") else "",
            " ".join(df.columns),
        ]
    ).lower()

    if "shooting" in text or {"sh", "sot"}.issubset({str(c).lower() for c in df.columns}):
        return "shooting"
    if "poss" in {str(c).lower() for c in df.columns}:
        return "schedule"
    return "unknown"


def read_matchlogs_table(soup, table_id):
    table = soup.find("table", id=table_id)
    if table is None:
        return None
    return flatten_columns(pd.read_html(StringIO(str(table)))[0])


def table_has_competition(df, competition):
    columns = {normalize_column_name(col): col for col in df.columns}
    comp_col = columns.get("comp")
    if comp_col is None:
        return False
    comp_values = df[comp_col].dropna().astype(str).str.strip().str.lower()
    return comp_values.eq(competition.lower()).any()


def clean_matchlog(df, team, table_side, competition):
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    if "date" not in df.columns:
        return pd.DataFrame()

    df = df[df["date"].notna()].copy()
    df = df[df["date"].astype(str).str.lower() != "date"]

    if "comp" in df.columns:
        df = df[df["comp"].astype(str).eq(competition)].copy()

    df["team"] = team
    df["table_side"] = table_side

    for col in ["team", "table_side", "date", "comp", "round", "venue", "result", "opponent"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_team_name)

    for col in df.columns:
        if col not in {
            "team",
            "table_side",
            "date",
            "time",
            "comp",
            "round",
            "day",
            "venue",
            "result",
            "opponent",
            "captain",
            "formation",
            "opp_formation",
            "referee",
            "match_report",
            "notes",
        }:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    return df


def discover_html_files(html_search_dirs):
    seen = set()
    files = []
    for directory in html_search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.html"):
            if path.name.lower().startswith("saved_resource"):
                continue
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            files.append(path)
    return sorted(files)


def parse_files(competition, html_search_dirs):
    schedule_rows = []
    schedule_fallback_rows = []
    shooting_for_rows = []
    shooting_against_rows = []

    for path in discover_html_files(html_search_dirs):
        html = path.read_text(encoding="utf-8", errors="ignore")
        if "matchlogs_for" not in html:
            continue

        soup = soup_with_commented_tables(html)
        title_text = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""
        team = infer_team_name(path, soup)

        matchlogs_for = read_matchlogs_table(soup, "matchlogs_for")
        if matchlogs_for is None:
            continue
        page_text = f"{path.name} {title_text}".lower()
        if competition.lower() not in page_text and not table_has_competition(matchlogs_for, competition):
            continue

        kind = infer_page_kind(path, soup, matchlogs_for)
        cleaned_for = clean_matchlog(matchlogs_for, team, "for", competition)
        if cleaned_for.empty:
            continue

        print(f"{path.name}: team={team}, kind={kind}, rows_for={len(cleaned_for)}")

        if kind == "schedule":
            schedule_rows.append(cleaned_for)
        elif kind == "shooting":
            shooting_cols = [
                "sh",
                "sot",
                "sot_pct",
                "sotpct",
                "g_per_sh",
                "g_per_sot",
                "pk",
                "pkatt",
                "dist",
                "fk",
            ]
            schedule_fallback_rows.append(
                cleaned_for.drop(columns=[col for col in shooting_cols if col in cleaned_for.columns])
            )
            shooting_for_rows.append(cleaned_for)
            matchlogs_against = read_matchlogs_table(soup, "matchlogs_against")
            if matchlogs_against is not None:
                cleaned_against = clean_matchlog(matchlogs_against, team, "against", competition)
                if not cleaned_against.empty:
                    shooting_against_rows.append(cleaned_against)

    scheduled_teams = set()
    if schedule_rows:
        scheduled_teams = set(pd.concat(schedule_rows, ignore_index=True)["team"].dropna())
    for fallback in schedule_fallback_rows:
        team = fallback["team"].dropna().iloc[0] if "team" in fallback.columns and fallback["team"].notna().any() else None
        if team and team not in scheduled_teams:
            schedule_rows.append(fallback)
            scheduled_teams.add(team)

    return schedule_rows, shooting_for_rows, shooting_against_rows


def write_output(out_dir, name, frames):
    path = out_dir / name
    if not frames:
        print(f"Sin datos para {name}")
        return

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=[c for c in ["team", "date", "opponent", "table_side"] if c in out.columns])
    out = out.sort_values([c for c in ["team", "date", "table_side"] if c in out.columns])
    out.to_csv(path, index=False)
    print(f"Guardado {path} ({len(out)} filas, {len(out.columns)} columnas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier", choices=sorted(LEAGUE_COMPETITIONS))
    parser.add_argument("--competition", default=None)
    args = parser.parse_args()

    competition = args.competition or LEAGUE_COMPETITIONS[args.league]
    raw_dir = FILES_DIR / "01_raw" / args.league
    out_dir = raw_dir / "team_matchlogs"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_search_dirs = [FILES_DIR, out_dir]

    print(f"Parseando matchlogs de {competition} -> {out_dir}")
    schedule_rows, shooting_for_rows, shooting_against_rows = parse_files(competition, html_search_dirs)
    write_output(out_dir, f"{args.league}_team_schedule.csv", schedule_rows)
    write_output(out_dir, f"{args.league}_team_shooting_for.csv", shooting_for_rows)
    write_output(out_dir, f"{args.league}_team_shooting_against.csv", shooting_against_rows)


if __name__ == "__main__":
    main()

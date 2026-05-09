"""
Dashboard interactivo — Liga Chilena · Primera División 2026

Uso:
    .runtime/python312/Scripts/streamlit.exe run apps/liga_chilena/dashboard.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from colo_xi import show_colo_xi_tab, show_pitch

BASE_DIR    = CURRENT_DIR
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="Liga Chilena · Análisis de jugadores",
    page_icon="⚽",
    layout="wide",
)

DARK  = "#0f1923"
PANEL = "#1a2635"
GREEN = "#2ecc71"
RED   = "#e74c3c"
PLOTLY_THEME = dict(template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL)

POS_LABEL = {"G": "Arquero", "D": "Defensa", "M": "Mediocampo", "F": "Delantera"}
FRANJAS_ORDER = ["0-15'", "16-30'", "31-45'", "46-60'", "61-75'", "76-90'+"]

TEAM_SHORT = {
    "Universidad de Chile":      "U. Chile",
    "Universidad Católica":      "U. Católica",
    "Universidad de Concepción": "U. Concep.",
    "Deportes Concepción":       "D. Concep.",
    "Deportes La Serena":        "La Serena",
    "Deportes Limache":          "Limache",
    "Everton de Viña del Mar":   "Everton",
    "Unión La Calera":           "La Calera",
    "Coquimbo Unido":            "Coquimbo",
    "Audax Italiano":            "Audax",
    "Cobresal":                  "Cobresal",
    "Huachipato":                "Huachipato",
    "Ñublense":                  "Ñublense",
    "O'Higgins":                 "O'Higgins",
    "Palestino":                 "Palestino",
    "Colo-Colo":                 "Colo-Colo",
}


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    idx    = pd.read_csv(REPORTS_DIR / "player_index.csv")
    pm     = pd.read_csv(REPORTS_DIR / "plus_minus_liga.csv").rename(columns={"minutos": "minutes"})
    stats  = pd.read_csv(REPORTS_DIR / "player_stats.csv")
    detail = pd.read_csv(REPORTS_DIR / ".." / "data" / "processed" / "player_match_pm.csv")
    goals  = pd.read_csv(REPORTS_DIR / "goals_by_minute.csv")
    subs   = pd.read_csv(REPORTS_DIR / "substitution_impact.csv")
    sub_agg= pd.read_csv(REPORTS_DIR / "sub_impact_by_player.csv")

    pm    = pm.merge(idx[["player_id","position","pos_label"]], on="player_id", how="left")
    stats = stats.merge(idx[["player_id","position","pos_label"]], on="player_id", how="left")

    stats["ga"]       = stats["goals"] + stats["assists"]
    stats["g_per90"]  = (stats["goals"]   / stats["minutes"] * 90).round(2)
    stats["a_per90"]  = (stats["assists"] / stats["minutes"] * 90).round(2)
    stats["ga_per90"] = (stats["ga"]      / stats["minutes"] * 90).round(2)
    stats["sh_per90"] = (stats["shots"]   / stats["minutes"] * 90).round(2)
    # conv_rate y sot_rate vienen del CSV como porcentaje (0-100)
    # columnas nuevas (pueden no existir en CSVs viejos — rellenar con 0)
    for col in ["kp_per90","int_per90","clr_per90","rec_per90","saves_per90",
                "pass_acc_pct","tackle_acc_pct","duel_win_pct",
                "bcc_per90","crosses_per90","long_ball_pct","cross_acc_pct",
                "dribble_win_pct","aerial_win_pct","fouls_per90","fouled_per90",
                "conv_rate","sot_rate"]:
        if col not in stats.columns:
            stats[col] = 0.0

    for df in (pm, stats):
        df["team_short"] = df["team"].map(TEAM_SHORT).fillna(df["team"])

    pm["hover"] = (
        pm["player"] + "<br>" + pm["team_short"] + " · " + pm["pos_label"].fillna("?") +
        "<br>GF: " + pm["gf_on"].astype(str) + "  GC: " + pm["ga_on"].astype(str) +
        "<br>+/-: " + pm["pm"].astype(str) + "  (" + pm["pm_per90"].astype(str) + "/90)"
    )
    stats["hover"] = (
        stats["player"] + "<br>" + stats["team_short"] + " · " + stats["pos_label"].fillna("?") +
        "<br>G:" + stats["goals"].astype(str) + " A:" + stats["assists"].astype(str) +
        " Tiros:" + stats["shots"].astype(str) +
        "<br>Conv:" + stats["conv_rate"].fillna(0).round(1).astype(str) + "%" +
        " Prec:" + stats["sot_rate"].fillna(0).round(1).astype(str) + "%"
    )

    detail["date"] = pd.to_datetime(detail["date"])
    goals["franja"] = pd.Categorical(goals["franja"], categories=FRANJAS_ORDER, ordered=True)
    sub_agg["team_short"] = sub_agg["team"].map(TEAM_SHORT).fillna(sub_agg["team"])

    teams = sorted(pm["team"].dropna().unique())
    return pm, stats, detail, goals, subs, sub_agg, teams


pm_all, stats_all, detail_all, goals_all, subs_all, sub_agg_all, all_teams = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Filtros")
    team_sel  = st.selectbox("Equipo", ["Todos los equipos"] + all_teams)
    pos_opts  = ["Todas las posiciones"] + [f"{k} — {v}" for k, v in POS_LABEL.items()]
    pos_sel   = st.selectbox("Posición", pos_opts)
    pos_code  = pos_sel.split(" — ")[0] if pos_sel != "Todas las posiciones" else None
    st.divider()
    min_min   = st.slider("Minutos mínimos", 90, 720, 270, step=45)
    min_shots = st.slider("Tiros mínimos (conversión/precisión)", 3, 20, 8)
    st.divider()
    st.caption("Liga Chilena · Primera División 2026")


def filt(df, min_m=None):
    d = df.copy()
    if min_m is not None:
        d = d[d["minutes"] >= min_m]
    if team_sel != "Todos los equipos":
        d = d[d["team"] == team_sel]
    if pos_code:
        d = d[d["position"] == pos_code]
    return d


# ── Helpers ────────────────────────────────────────────────────────────────────

def scatter_fig(df, x, y, color_col, size_col, hover, title, xlab, ylab,
                colorscale="RdYlGn", diag=False):
    fig = px.scatter(df, x=x, y=y, color=color_col, size=size_col, size_max=28,
                     color_continuous_scale=colorscale, hover_name=hover,
                     hover_data={x: True, y: True, color_col: ":.2f", size_col: True, "team": True},
                     title=title, labels={x: xlab, y: ylab})
    if diag and len(df):
        lim = max(df[x].max(), df[y].max()) + 1
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                      line=dict(color="white", width=1, dash="dot"))
    fig.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.88)
    fig.update_layout(**PLOTLY_THEME, height=500,
                      coloraxis_colorbar=dict(title=color_col, tickfont_color="white"))
    return fig


def diag_scatter_fig(df, x, y, color_col, hover, title, xlab, ylab, colorscale="RdYlGn"):
    fig = px.scatter(df, x=x, y=y, color=color_col, size="minutes", size_max=28,
                     color_continuous_scale=colorscale, hover_name=hover,
                     hover_data={x: True, y: True, color_col: ":.3f", "minutes": True, "team": True},
                     title=title, labels={x: xlab, y: ylab})
    if len(df) and df[x].sum() > 0:
        avg = df[y].sum() / df[x].sum()
        xm  = df[x].max() * 1.08
        fig.add_shape(type="line", x0=0, y0=0, x1=xm, y1=avg*xm,
                      line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dash"))
        fig.add_annotation(x=xm*0.88, y=avg*xm*0.88, text=f"prom. {avg:.1%}",
                           showarrow=False, font=dict(color="white", size=11), opacity=0.6)
    fig.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.88)
    fig.update_layout(**PLOTLY_THEME, height=500,
                      coloraxis_colorbar=dict(title=color_col, tickfont_color="white"))
    return fig


def bar_ranking(df, metric, label, n=15):
    top  = df.nlargest(n, metric)
    bot  = df.nsmallest(n, metric)
    sub  = pd.concat([bot, top]).drop_duplicates().sort_values(metric)
    sub["label"] = sub["player"].str.split().str[-1] + " · " + sub["team_short"]
    fig = go.Figure(go.Bar(
        x=sub[metric], y=sub["label"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in sub[metric]],
        text=sub[metric].map(lambda v: f"{v:+.2f}"), textposition="outside",
        hovertext=sub["player"] + "<br>" + sub["team"] + "<br>" + label + ": " +
                  sub[metric].map(lambda v: f"{v:+.2f}"),
        hoverinfo="text",
    ))
    fig.update_layout(**PLOTLY_THEME, height=max(500, len(sub)*28),
                      title=f"Top/Bottom {n} — {label}",
                      xaxis_title=label, yaxis_title=None,
                      xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1),
                      margin=dict(l=180))
    return fig


def prod_scatter(df, ycol, rate_col, ylabel, title):
    d = df[df[ycol] >= 1].copy()
    fig = px.scatter(d, x="minutes", y=ycol, color=rate_col, size="minutes", size_max=28,
                     color_continuous_scale="RdYlGn", hover_name="hover",
                     hover_data={"minutes": True, ycol: True, rate_col: ":.2f", "team": True},
                     title=title, labels={"minutes": "Minutos jugados", ycol: ylabel,
                                          rate_col: f"{ylabel}/90"})
    xm = d["minutes"].max() * 1.06 if len(d) else 1000
    for rate, dash in [(0.25, "dot"), (0.5, "dash"), (1.0, "dashdot")]:
        y_end = rate * xm / 90
        if len(d) and y_end > d[ycol].max() * 1.5:
            continue
        fig.add_shape(type="line", x0=0, y0=0, x1=xm, y1=y_end,
                      line=dict(color="rgba(255,255,255,0.22)", width=1.2, dash=dash))
        fig.add_annotation(x=xm*0.96, y=y_end*0.96, text=f"{rate}/90",
                           showarrow=False, font=dict(color="white", size=10), opacity=0.5)
    fig.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.88)
    fig.update_layout(**PLOTLY_THEME, height=460,
                      coloraxis_colorbar=dict(tickfont_color="white"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "⚖️ +/- Liga",
    "🎯 Ofensivo",
    "⏱️ Producción",
    "🕸️ Comparador",
    "Mejor XI campeonato",
    "⚡ Goles por minuto",
    "🔄 Sustituciones",
    "💎 Infravalorados",
    "🔍 Similares",
    "Recomendar XI",
])

# ╔══════════════════════════╗
# ║  Tab 1: +/- Liga        ║
# ╚══════════════════════════╝
with tab1:
    pm = filt(pm_all, min_m=min_min)
    st.subheader("GF vs GC — con el jugador en cancha")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(scatter_fig(pm, "gf_on", "ga_on", "pm_per90", "minutes", "hover",
                        "GF vs GC (color = +/-/90)", "Goles a favor", "Goles en contra",
                        diag=True), use_container_width=True)
    with c2:
        fig = px.scatter(pm, x="gf_on", y="ga_on", color="team_short", size="minutes",
                         size_max=25, hover_name="hover", title="GF vs GC por equipo",
                         labels={"gf_on": "Goles a favor", "ga_on": "Goles en contra",
                                 "team_short": "Equipo"})
        if len(pm):
            lim = max(pm["gf_on"].max(), pm["ga_on"].max()) + 1
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                          line=dict(color="white", width=1, dash="dot"))
        fig.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.88)
        fig.update_layout(**PLOTLY_THEME, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(bar_ranking(pm, "pm",       "+/- bruto"),     use_container_width=True)
    with c4:
        st.plotly_chart(bar_ranking(pm, "pm_per90", "+/- por 90 min"), use_container_width=True)


# ╔══════════════════════════╗
# ║  Tab 2: Ofensivo        ║
# ╚══════════════════════════╝
with tab2:
    st_df = filt(stats_all, min_m=min_min)
    st.subheader("Goles vs Asistencias")
    st.plotly_chart(scatter_fig(st_df, "goals", "assists", "ga_per90", "minutes", "hover",
                    "Goles vs Asistencias  (color = G+A/90)", "Goles", "Asistencias",
                    colorscale="RdYlGn", diag=True), use_container_width=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        d = st_df[st_df["shots"] >= min_shots]
        st.plotly_chart(diag_scatter_fig(d, "shots", "goals", "conv_rate", "hover",
                        f"Goles vs Tiros — conversión  (mín. {min_shots} tiros)",
                        "Tiros totales", "Goles"), use_container_width=True)
    with c2:
        d = st_df[st_df["shots"] >= min_shots]
        fig = diag_scatter_fig(d, "shots", "sot", "sot_rate", "hover",
                               f"Tiros al arco vs Tiros — precisión  (mín. {min_shots} tiros)",
                               "Tiros totales", "Tiros al arco")
        if len(d):
            xm = d["shots"].max() * 1.08
            fig.add_shape(type="line", x0=0, y0=0, x1=xm, y1=xm,
                          line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"))
        st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════╗
# ║  Tab 3: Producción      ║
# ╚══════════════════════════╝
with tab3:
    st_df = filt(stats_all, min_m=min_min)
    st.plotly_chart(prod_scatter(st_df, "goals",   "g_per90",  "Goles",
                    "Goles vs Minutos jugados"), use_container_width=True)
    st.plotly_chart(prod_scatter(st_df, "assists",  "a_per90",  "Asistencias",
                    "Asistencias vs Minutos jugados"), use_container_width=True)
    st.plotly_chart(prod_scatter(st_df, "ga",       "ga_per90", "G + A",
                    "G+A vs Minutos jugados"), use_container_width=True)


# ╔══════════════════════════════╗
# ║  Tab 4: Radar Comparador    ║
# ╚══════════════════════════════╝
with tab4:
    st.subheader("🕸️ Comparador de jugadores")

    RADAR_BY_POS = {
        "G": {
            "Atajadas/90":    "saves_per90",    # volumen — cuánto trabaja
            "Despejes/90":    "clr_per90",      # salidas activas
            "% Pases":        "pass_acc_pct",   # juego corto
            "% Pases largos": "long_ball_pct",  # distribución larga
            "Recuperac./90":  "rec_per90",      # pressing / salida al balón
            "+/-/90":         "pm_per90",
        },
        "D": {
            "% Tackles":      "tackle_acc_pct", # efectividad en entradas
            "% Duelos":       "duel_win_pct",   # dominio físico total
            "Interc./90":     "int_per90",      # lectura de juego
            "Despejes/90":    "clr_per90",      # trabajo de zona
            "% Pases":        "pass_acc_pct",   # calidad en salida
            "+/-/90":         "pm_per90",
        },
        "M": {
            "% Pases":        "pass_acc_pct",   # precisión general
            "Pases clave/90": "kp_per90",       # creatividad / último pase
            "Gr. ocasiones/90":"bcc_per90",     # generación de peligro real
            "% Centros":      "cross_acc_pct",  # calidad de centros
            "% Duelos":       "duel_win_pct",   # disputa del balón
            "+/-/90":         "pm_per90",
        },
        "F": {
            "Goles/90":       "g_per90",
            "% Conversión":   "conv_rate",      # efectividad goleadora
            "% Al arco":      "sot_rate",       # precisión de remate
            "Asist./90":      "a_per90",        # generación para compañeros
            "% Regates":      "dribble_win_pct",# desequilibrio individual
            "+/-/90":         "pm_per90",
        },
        "?": {
            "Goles/90":       "g_per90",
            "Asist./90":      "a_per90",
            "% Pases":        "pass_acc_pct",
            "% Duelos":       "duel_win_pct",
            "Recuperac./90":  "rec_per90",
            "+/-/90":         "pm_per90",
        },
    }

    # Base: TODOS los jugadores con al menos 90 min — independiente del sidebar
    radar_base = stats_all[stats_all["minutes"] >= 90].merge(
        pm_all[["player_id", "pm_per90"]], on="player_id", how="left"
    )
    radar_base["sel_label"] = (
        radar_base["player"].str.split().str[-1] + " — " +
        radar_base["team"].map(TEAM_SHORT).fillna(radar_base["team"]) + " (" +
        radar_base["pos_label"].fillna("?") + ")"
    )

    # Filtros propios del comparador (no dependen del sidebar)
    cf1, cf2, cf3 = st.columns([2, 2, 1])
    with cf1:
        radar_teams = ["Todos"] + sorted(radar_base["team"].dropna().unique())
        radar_team  = st.selectbox("Filtrar por equipo", radar_teams, key="radar_team")
    with cf2:
        radar_pos_opts = ["Todas"] + [f"{k} — {v}" for k, v in POS_LABEL.items()]
        radar_pos_sel  = st.selectbox("Filtrar por posición", radar_pos_opts, key="radar_pos")
        radar_pos_code = radar_pos_sel.split(" — ")[0] if radar_pos_sel != "Todas" else None
    with cf3:
        radar_min = st.number_input("Min. minutos", min_value=45, max_value=900,
                                    value=90, step=45, key="radar_min")

    rb_filt = radar_base[radar_base["minutes"] >= radar_min].copy()
    if radar_team != "Todos":
        rb_filt = rb_filt[rb_filt["team"] == radar_team]
    if radar_pos_code:
        rb_filt = rb_filt[rb_filt["position"] == radar_pos_code]

    all_labels   = sorted(rb_filt["sel_label"].unique())
    default_lbls = all_labels[:3] if len(all_labels) >= 3 else all_labels

    sel_labels = st.multiselect(
        f"Jugadores ({len(all_labels)} disponibles)",
        all_labels, default=default_lbls, max_selections=5,
    )

    # Pool completo de métricas disponibles (etiqueta → columna)
    ALL_METRICS = {
        # Ofensivo
        "Goles/90":              "g_per90",
        "Asist./90":             "a_per90",
        "Tiros/90":              "sh_per90",
        "% Conversión":          "conv_rate",
        "% Al arco":             "sot_rate",
        "Pases clave/90":        "kp_per90",
        "Gr. ocasiones/90":      "bcc_per90",
        # Pases
        "% Pases":               "pass_acc_pct",
        "% Pases largos":        "long_ball_pct",
        "Centros/90":            "crosses_per90",
        "% Centros":             "cross_acc_pct",
        # Regates
        "% Regates":             "dribble_win_pct",
        # Defensivo
        "% Duelos":              "duel_win_pct",
        "% Tackles":             "tackle_acc_pct",
        "Interc./90":            "int_per90",
        "Despejes/90":           "clr_per90",
        "Recuperac./90":         "rec_per90",
        "% Duelos aéreos":       "aerial_win_pct",
        # Disciplina
        "Faltas cometidas/90":   "fouls_per90",
        "Faltas recibidas/90":   "fouled_per90",
        # Portero
        "Atajadas/90":           "saves_per90",
        # Impacto
        "+/-/90":                "pm_per90",
    }

    if len(sel_labels) < 2:
        st.info("Selecciona al menos 2 jugadores.")
    else:
        sel_rows = radar_base[radar_base["sel_label"].isin(sel_labels)]

        # Posiciones de los jugadores seleccionados
        positions_sel = sel_rows["position"].dropna().unique().tolist()
        mixed = len(positions_sel) > 1

        # Métricas default: del primer jugador seleccionado
        first_pos = sel_rows.iloc[0]["position"] if not sel_rows.empty else "?"
        default_metrics = list(RADAR_BY_POS.get(first_pos, RADAR_BY_POS["?"]).keys())

        if mixed:
            st.info(
                f"Posiciones mezcladas ({', '.join(POS_LABEL.get(p, p) for p in positions_sel)}). "
                f"Elegí las métricas a comparar — el default es el perfil de **{POS_LABEL.get(first_pos, first_pos)}**."
            )
            chosen_labels = st.multiselect(
                "Métricas a comparar",
                options=list(ALL_METRICS.keys()),
                default=default_metrics,
                max_selections=7,
                key="radar_metrics",
            )
            if len(chosen_labels) < 3:
                st.warning("Selecciona al menos 3 métricas.")
                st.stop()
            metrics_labels = chosen_labels
            metrics_cols   = [ALL_METRICS[l] for l in chosen_labels]
            # Percentil sobre TODOS los jugadores (posición mixta)
            pos_pool = radar_base.copy()
            caption_txt = f"Percentil calculado sobre {len(pos_pool)} jugadores (≥{int(radar_min)} min, todas las posiciones)"
        else:
            dominant_pos   = positions_sel[0] if positions_sel else "?"
            metrics        = RADAR_BY_POS.get(dominant_pos, RADAR_BY_POS["?"])
            metrics_labels = list(metrics.keys())
            metrics_cols   = list(metrics.values())
            pos_pool       = radar_base[radar_base["position"] == dominant_pos].copy()
            caption_txt    = (
                f"Métricas para **{POS_LABEL.get(dominant_pos, dominant_pos)}** · "
                f"Percentil sobre {len(pos_pool)} jugadores de esa posición (≥{int(radar_min)} min)"
            )

        for col in metrics_cols:
            pos_pool[f"{col}_pct"] = pos_pool[col].fillna(0).rank(pct=True) * 100

        st.caption(caption_txt)

        fig = go.Figure()
        colors = px.colors.qualitative.Bold

        for i, lbl in enumerate(sel_labels):
            row_stats = radar_base[radar_base["sel_label"] == lbl]
            if row_stats.empty:
                continue
            pid   = row_stats.iloc[0]["player_id"]
            pname = row_stats.iloc[0]["player"]
            pool_row = pos_pool[pos_pool["player_id"] == pid]
            if pool_row.empty:
                continue
            pool_row = pool_row.iloc[0]

            values = [round(float(pool_row.get(f"{c}_pct", 0) or 0), 1) for c in metrics_cols]
            raw    = [round(float(pool_row.get(c, 0) or 0), 3) for c in metrics_cols]
            values_closed = values + [values[0]]
            theta_closed  = metrics_labels + [metrics_labels[0]]

            hover_txt = "<br>".join(
                f"{l}: {v}  (pct {p:.0f}°)"
                for l, v, p in zip(metrics_labels, raw, values)
            )
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatterpolar(
                r=values_closed, theta=theta_closed,
                fill="toself", name=pname,
                line=dict(color=color, width=2.5),
                fillcolor=color, opacity=0.18,
                hovertemplate=f"<b>{pname}</b><br>{hover_txt}<extra></extra>",
            ))
            fig.add_trace(go.Scatterpolar(
                r=values_closed, theta=theta_closed,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                name=pname, showlegend=False,
                hoverinfo="skip",
            ))

        fig.update_layout(
            **PLOTLY_THEME,
            polar=dict(
                bgcolor=PANEL,
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickfont=dict(color="white", size=11),
                    gridcolor="#3a5068", linecolor="#3a5068",
                    tickvals=[20, 40, 60, 80, 100],
                ),
                angularaxis=dict(
                    tickfont=dict(color="white", size=14),
                    gridcolor="#3a5068", linecolor="#3a5068",
                ),
            ),
            legend=dict(font=dict(color="white", size=13), bgcolor=PANEL,
                        bordercolor="#3a5068", borderwidth=1),
            height=600, showlegend=True,
            title=dict(text="Radar comparativo — percentil en la posición",
                       font=dict(color="white", size=14)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de valores absolutos
        rows_table = []
        for lbl in sel_labels:
            row_stats = radar_base[radar_base["sel_label"] == lbl]
            if row_stats.empty:
                continue
            pid = row_stats.iloc[0]["player_id"]
            pool_row = pos_pool[pos_pool["player_id"] == pid]
            if pool_row.empty:
                continue
            r = pool_row.iloc[0]
            rows_table.append({
                "Jugador": r["player"], "Equipo": r["team"], "Min": int(r["minutes"]),
                **{l: round(float(r.get(c, 0) or 0), 3) for l, c in zip(metrics_labels, metrics_cols)},
            })
        if rows_table:
            st.dataframe(pd.DataFrame(rows_table).set_index("Jugador"),
                         use_container_width=True)


# ╔══════════════════════════════╗
# ║  Tab 5: Mejor XI campeonato ║
# ╚══════════════════════════════╝
with tab5:
    st.subheader("Mejor XI del campeonato")
    min_xi_minutes = st.slider("Minutos minimos", 180, 900, 450, step=45, key="best_xi_min")

    xi_pool = stats_all.merge(
        pm_all[["player_id", "gf_on", "ga_on", "pm", "pm_per90"]],
        on="player_id",
        how="left",
        suffixes=("", "_pm"),
    ).copy()
    xi_pool = xi_pool[xi_pool["minutes"].fillna(0) >= min_xi_minutes]

    def championship_score(row):
        minutes_score = min(float(row.get("minutes") or 0), 900) / 900
        pm_score = np.clip(float(row.get("pm_per90") or 0), -1.5, 1.8)
        duel_score = np.clip(float(row.get("duel_win_pct") or 45), 0, 75) / 100
        rec_score = np.clip(float(row.get("rec_per90") or 0), 0, 9)
        ga_score = np.clip(float(row.get("ga_per90") or 0), 0, 2.5)
        kp_score = np.clip(float(row.get("kp_per90") or 0), 0, 4)
        saves_score = np.clip(float(row.get("saves_per90") or 0), 0, 6)
        defensive_score = np.clip(float(row.get("int_per90") or 0) + float(row.get("clr_per90") or 0), 0, 12)
        goals_score = np.clip(float(row.get("g_per90") or 0), 0, 1.2)
        position = row.get("position")
        if position == "G":
            return pm_score * 0.24 + saves_score * 0.14 + minutes_score * 0.34 + duel_score * 0.08 - ga_score * 0.06
        if position == "D":
            return pm_score * 0.26 + defensive_score * 0.08 + duel_score * 0.18 + rec_score * 0.05 + minutes_score * 0.26
        if position == "M":
            return pm_score * 0.24 + kp_score * 0.08 + rec_score * 0.07 + duel_score * 0.13 + minutes_score * 0.24 + ga_score * 0.08
        if position == "F":
            return pm_score * 0.20 + goals_score * 0.23 + ga_score * 0.12 + kp_score * 0.07 + minutes_score * 0.20
        return pm_score * 0.25 + minutes_score * 0.25

    xi_pool["score_xi"] = xi_pool.apply(championship_score, axis=1).round(3)
    formation = {"G": 1, "D": 4, "M": 4, "F": 2}
    best_parts = []
    for position, amount in formation.items():
        best_parts.append(
            xi_pool[xi_pool["position"].eq(position)].sort_values(
                ["score_xi", "minutes"],
                ascending=False,
            ).head(amount)
        )
    best_xi = pd.concat(best_parts, ignore_index=True) if best_parts else pd.DataFrame()

    if best_xi.empty:
        st.info("Sin jugadores suficientes con el filtro de minutos.")
    else:
        selected_pitch = {
            position: best_xi[best_xi["position"].eq(position)].copy()
            for position in formation
        }
        show_pitch(selected_pitch)
        pos_order = {"G": 0, "D": 1, "M": 2, "F": 3}
        best_xi["Linea"] = best_xi["position"].map(POS_LABEL)
        best_xi["Orden"] = best_xi["position"].map(pos_order)
        table = best_xi.sort_values(["Orden", "score_xi"], ascending=[True, False]).rename(columns={
            "player": "Jugador",
            "team_short": "Equipo",
            "minutes": "Min",
            "pm": "+/-",
            "pm_per90": "+/-90",
            "goals": "Goles",
            "assists": "Asist.",
            "ga_per90": "G+A90",
            "score_xi": "Score XI",
        })
        st.dataframe(
            table[["Linea", "Jugador", "Equipo", "Min", "+/-", "+/-90", "Goles", "Asist.", "G+A90", "Score XI"]],
            use_container_width=True,
            hide_index=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equipos representados", best_xi["team"].nunique())
        c2.metric("Prom. +/-90", f"{best_xi['pm_per90'].mean():+.2f}")
        c3.metric("Goles XI", int(best_xi["goals"].sum()))
        c4.metric("Asistencias XI", int(best_xi["assists"].sum()))

        st.markdown("**Lectura**")
        st.write(
            "El XI prioriza continuidad, impacto en +/- por 90 y produccion segun rol. "
            "Para delanteros pesa mas gol/G+A; para mediocampistas creacion y recuperacion; "
            "para defensas duelos, acciones defensivas y estabilidad; para arqueros minutos, atajadas e impacto."
        )

        fig = px.bar(
            table.sort_values("Score XI"),
            x="Score XI",
            y="Jugador",
            color="Linea",
            orientation="h",
            hover_data=["Equipo", "Min", "+/-90", "Goles", "Asist."],
            title="Ranking interno del XI ideal",
        )
        fig.update_layout(**PLOTLY_THEME, height=480, yaxis_title=None, xaxis_title="Score XI")
        st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════╗
# ║  Tab 6: Goles por minuto    ║
# ╚══════════════════════════════╝
with tab6:
    st.subheader("⚡ Distribución de goles por franja horaria")

    gdf = goals_all.copy()
    team_filter_goals = team_sel if team_sel != "Todos los equipos" else None

    if team_filter_goals:
        gf_team = gdf[gdf["scoring_team"]  == team_filter_goals].copy()
        ga_team = gdf[gdf["conceding_team"] == team_filter_goals].copy()

        gf_cnt = gf_team.groupby("franja").size().reindex(FRANJAS_ORDER, fill_value=0).reset_index()
        ga_cnt = ga_team.groupby("franja").size().reindex(FRANJAS_ORDER, fill_value=0).reset_index()
        gf_cnt.columns = ["franja", "count"]
        ga_cnt.columns = ["franja", "count"]

        fig = go.Figure([
            go.Bar(name="Goles a favor",  x=gf_cnt["franja"], y= gf_cnt["count"],
                   marker_color=GREEN, opacity=0.85),
            go.Bar(name="Goles en contra", x=ga_cnt["franja"], y=-ga_cnt["count"],
                   marker_color=RED,   opacity=0.85),
        ])
        title_str = f"Goles por franja — {TEAM_SHORT.get(team_filter_goals, team_filter_goals)}"
    else:
        cnt = gdf.groupby("franja").size().reindex(FRANJAS_ORDER, fill_value=0).reset_index()
        cnt.columns = ["franja", "count"]
        fig = go.Figure(go.Bar(x=cnt["franja"], y=cnt["count"],
                               marker_color=GREEN, opacity=0.85))
        title_str = "Goles por franja — toda la liga"

    fig.add_hline(y=0, line_color="white", line_width=0.8)
    fig.update_layout(**PLOTLY_THEME, height=420, title=title_str,
                      barmode="overlay",
                      xaxis_title="Franja", yaxis_title="Goles",
                      legend=dict(font_color="white"))
    st.plotly_chart(fig, use_container_width=True)

    # Segundo gráfico: minuto exacto (scatter de densidad)
    st.subheader("Minuto exacto de cada gol")
    gdf2 = goals_all.copy()
    if team_filter_goals:
        gdf2_gf = gdf2[gdf2["scoring_team"]  == team_filter_goals].assign(tipo="A favor")
        gdf2_ga = gdf2[gdf2["conceding_team"] == team_filter_goals].assign(tipo="En contra")
        gdf2 = pd.concat([gdf2_gf, gdf2_ga])
        color_map = {"A favor": GREEN, "En contra": RED}
    else:
        gdf2 = gdf2.assign(tipo="Gol")
        color_map = {"Gol": GREEN}

    fig2 = px.strip(gdf2, x="minute", color="tipo", color_discrete_map=color_map,
                    stripmode="overlay",
                    title="Distribución de goles por minuto exacto",
                    labels={"minute": "Minuto", "tipo": ""})
    fig2.update_traces(jitter=0.6, marker_size=6, opacity=0.7)
    fig2.update_layout(**PLOTLY_THEME, height=280, showlegend=True,
                       legend=dict(font_color="white"))
    st.plotly_chart(fig2, use_container_width=True)


# ╔══════════════════════════════╗
# ║  Tab 7: Sustituciones       ║
# ╚══════════════════════════════╝
with tab7:
    st.subheader("🔄 Impacto de sustituciones")
    st.caption("+/- del equipo desde que el jugador entra hasta el final del partido.")

    sub_df = sub_agg_all.copy()
    if team_sel != "Todos los equipos":
        sub_df = sub_df[sub_df["team"] == team_sel]

    min_entradas = st.slider("Mínimo de entradas como sustituto", 1, 6, 2)
    sub_df = sub_df[sub_df["entradas"] >= min_entradas].sort_values("pm_after_avg")

    if sub_df.empty:
        st.info("Sin datos con los filtros aplicados.")
    else:
        n_show = st.slider("Jugadores a mostrar (top + bottom)", 5, 20, 12)
        top  = sub_df.nlargest(n_show // 2,  "pm_after_avg")
        bot  = sub_df.nsmallest(n_show // 2, "pm_after_avg")
        show = pd.concat([bot, top]).drop_duplicates().sort_values("pm_after_avg")
        show["label"] = show["player_in"].str.split().str[-1] + " · " + show["team_short"]

        fig = go.Figure(go.Bar(
            x=show["pm_after_avg"], y=show["label"],
            orientation="h",
            marker_color=[GREEN if v >= 0 else RED for v in show["pm_after_avg"]],
            text=show["pm_after_avg"].map(lambda v: f"{v:+.2f}"),
            textposition="outside",
            customdata=np.stack([show["player_in"], show["team"], show["entradas"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                          "Entradas: %{customdata[2]}<br>+/- promedio: %{x:+.2f}<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_THEME, height=max(450, len(show)*30),
            title=f"Impacto promedio al ingresar  (mín. {min_entradas} entradas)",
            xaxis_title="+/- promedio mientras estuvo en cancha", yaxis_title=None,
            xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1),
            margin=dict(l=180),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: minuto de ingreso vs pm_after
        st.subheader("¿En qué minuto entra y cuánto impacta?")
        subs_det = subs_all.copy()
        if team_sel != "Todos los equipos":
            subs_det = subs_det[subs_det["team"] == team_sel]
        if not subs_det.empty:
            subs_det["team_short"] = subs_det["team"].map(TEAM_SHORT).fillna(subs_det["team"])
            subs_det["hover"] = (subs_det["player_in"] + "<br>" + subs_det["team_short"] +
                                 "<br>Minuto: " + subs_det["minute"].astype(str) +
                                 "<br>Marcador al entrar: " + subs_det["pm_at_entry"].astype(str) +
                                 "<br>+/- mientras estuvo: " + subs_det["pm_after"].astype(str))
            fig2 = px.scatter(subs_det, x="minute", y="pm_after",
                              color="pm_after",
                              color_continuous_scale=[[0,RED],[0.5,"#2e3f52"],[1,GREEN]],
                              hover_name="hover",
                              title="Minuto de ingreso vs +/- mientras estuvo en cancha",
                              labels={"minute": "Minuto de ingreso", "pm_after": "+/- posterior"})
            fig2.add_hline(y=0, line_color="white", line_width=0.7, line_dash="dot")
            fig2.update_traces(marker_line_color="white", marker_line_width=0.4,
                               marker_size=7, opacity=0.8)
            fig2.update_layout(**PLOTLY_THEME, height=420,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)


# ╔══════════════════════════════════╗
# ║  Tab 8: Jugadores infravalorados ║
# ╚══════════════════════════════════╝
with tab8:
    st.subheader("💎 Mapa de minutos vs rendimiento")
    st.caption(
        "Arriba-izquierda (pocos minutos, alto +/-/90): potencialmente infravalorados.  "
        "Abajo-derecha (muchos minutos, bajo +/-/90): posibles sobrecargados."
    )

    gem_df = filt(pm_all, min_m=90)   # mínimo solo 90 min para incluir suplentes con impacto
    gem_df = gem_df.merge(stats_all[["player_id","goals","assists","ga_per90"]],
                          on="player_id", how="left")
    gem_df["ga_label"] = gem_df["goals"].astype(str) + "G " + gem_df["assists"].astype(str) + "A"
    gem_df["hover_gem"] = (
        gem_df["player"] + "<br>" + gem_df["team_short"] + " · " + gem_df["pos_label"].fillna("?") +
        "<br>Minutos: " + gem_df["minutes"].astype(str) +
        "<br>+/-/90: " + gem_df["pm_per90"].astype(str) +
        "<br>" + gem_df["ga_label"]
    )

    if gem_df.empty:
        st.info("Sin datos.")
    else:
        avg_min = gem_df["minutes"].median()
        avg_pm  = gem_df["pm_per90"].median()

        fig = px.scatter(gem_df, x="minutes", y="pm_per90",
                         color="pm_per90",
                         color_continuous_scale=[[0,RED],[0.5,"#2e3f52"],[1,GREEN]],
                         size="minutes", size_max=22,
                         hover_name="hover_gem",
                         title="Minutos jugados vs +/-/90  (cuadrantes por mediana)",
                         labels={"minutes": "Minutos jugados", "pm_per90": "+/- por 90 min"})

        # Líneas de cuadrante en la mediana
        xmax = gem_df["minutes"].max() + 50
        ymin = gem_df["pm_per90"].min() - 0.3
        ymax = gem_df["pm_per90"].max() + 0.3

        fig.add_vline(x=avg_min, line_color="rgba(255,255,255,0.3)", line_dash="dash")
        fig.add_hline(y=avg_pm,  line_color="rgba(255,255,255,0.3)", line_dash="dash")

        # Etiquetas de cuadrante
        for txt, ax, ay in [
            ("💎 Infravalorados",   gem_df["minutes"].min(), ymax*0.88),
            ("⭐ Titulares sólidos", xmax*0.7,              ymax*0.88),
            ("📉 Bajo impacto",     xmax*0.7,              ymin*0.88),
            ("❓ Poco uso",          gem_df["minutes"].min(), ymin*0.88),
        ]:
            fig.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                               font=dict(color="rgba(255,255,255,0.4)", size=11))

        fig.update_traces(marker_line_color="white", marker_line_width=0.4, opacity=0.85)
        fig.update_layout(**PLOTLY_THEME, height=560, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de posibles infravalorados
        st.subheader("Candidatos a infravalorados")
        gems = gem_df[(gem_df["minutes"] < avg_min) & (gem_df["pm_per90"] > avg_pm)] \
               .sort_values("pm_per90", ascending=False)
        st.dataframe(
            gems[["player","team","pos_label","minutes","pm","pm_per90","goals","assists"]]
            .rename(columns={"pos_label":"pos","pm_per90":"+/-/90"})
            .reset_index(drop=True),
            use_container_width=True,
            height=min(400, len(gems)*38 + 40),
        )


# ╔══════════════════════════════╗
# ║  Tab Tabla (al final)       ║
# ╚══════════════════════════════╝
# Añadir pestaña de tabla directamente con un expander en la última tab
with tab8:
    st.divider()
    with st.expander("📋 Tabla completa de jugadores"):
        merged = filt(pm_all, min_m=min_min).merge(
            stats_all[["player_id","goals","assists","shots","sot",
                        "ga_per90","conv_rate","sot_rate"]],
            on="player_id", how="left",
        )
        cols = ["player","team","pos_label","partidos","minutes",
                "gf_on","ga_on","pm","pm_per90",
                "goals","assists","shots","sot","ga_per90","conv_rate"]
        cols = [c for c in cols if c in merged.columns]
        st.dataframe(merged[cols].sort_values("pm", ascending=False).reset_index(drop=True),
                     use_container_width=True, height=500)
        st.download_button("⬇️ Descargar CSV",
                           merged[cols].to_csv(index=False).encode("utf-8"),
                           "liga_chilena_2026.csv", "text/csv")


# ╔══════════════════════════════════╗
# ║  Tab 9: Jugadores similares      ║
# ╚══════════════════════════════════╝
with tab9:
    st.subheader("🔍 Jugadores similares a...")
    st.caption(
        "Distancia euclídea en espacio de percentiles. "
        "Las variables de comparación son las del perfil de la posición del jugador elegido, "
        "pero podés ajustarlas manualmente."
    )

    # Base: todos con ≥90 min — independiente del sidebar
    sim_base = stats_all[stats_all["minutes"] >= 90].merge(
        pm_all[["player_id", "pm_per90"]], on="player_id", how="left"
    ).copy()
    sim_base["sel_label"] = (
        sim_base["player"].str.split().str[-1] + " — " +
        sim_base["team"].map(TEAM_SHORT).fillna(sim_base["team"]) + " (" +
        sim_base["pos_label"].fillna("?") + ")"
    )

    sc1, sc2, sc3 = st.columns([2, 2, 1])
    with sc1:
        sim_min = st.number_input("Mín. minutos (candidatos)", 90, 900, 270, 45, key="sim_min")
    with sc2:
        sim_pool_opts = ["Todas las posiciones"] + [f"{k} — {v}" for k, v in POS_LABEL.items()]
        sim_pool_pos  = st.selectbox("Buscar similares en", sim_pool_opts, key="sim_pool_pos")
        sim_pool_code = sim_pool_pos.split(" — ")[0] if sim_pool_pos != "Todas las posiciones" else None
    with sc3:
        n_similar = st.number_input("Top N", 3, 25, 10, key="sim_n")

    ref_label = st.selectbox(
        "Jugador de referencia",
        sorted(sim_base["sel_label"].unique()),
        key="sim_ref",
    )
    ref_rows = sim_base[sim_base["sel_label"] == ref_label]

    if not ref_rows.empty:
        ref_row = ref_rows.iloc[0]
        ref_pos = ref_row.get("position") or "?"
        ref_pid = int(ref_row["player_id"])

        default_mlbls = list(RADAR_BY_POS.get(ref_pos, RADAR_BY_POS["?"]).keys())
        sim_mlbls = st.multiselect(
            f"Variables de comparación  [default: perfil **{POS_LABEL.get(ref_pos, ref_pos)}**]",
            list(ALL_METRICS.keys()),
            default=default_mlbls,
            key=f"sim_metrics_{ref_pid}",
        )

        if len(sim_mlbls) < 2:
            st.info("Selecciona al menos 2 variables.")
        else:
            sim_mcols = [ALL_METRICS[l] for l in sim_mlbls]

            # Pool de candidatos
            pool = sim_base[sim_base["minutes"] >= int(sim_min)].copy()
            if sim_pool_code:
                pool = pool[pool["position"] == sim_pool_code]

            if len(pool) < 2:
                st.info("Pool muy pequeño — ajusta los filtros.")
            else:
                # Percentile ranks dentro del pool
                pcols = [f"{c}__pct" for c in sim_mcols]
                for c, pc in zip(sim_mcols, pcols):
                    pool[pc] = pool[c].fillna(0).rank(pct=True) * 100

                # Vector percentil del jugador de referencia
                ref_in_pool = pool[pool["player_id"] == ref_pid]
                if not ref_in_pool.empty:
                    ref_vec  = ref_in_pool.iloc[0][pcols].fillna(0).values.astype(float)
                    pct_note = ""
                else:
                    # Interpolación: percentil del valor raw dentro del pool
                    ref_raw = np.array([float(ref_row.get(c, 0) or 0) for c in sim_mcols])
                    ref_vec = np.array([
                        (pool[c].fillna(0) < ref_raw[i]).sum() / len(pool) * 100
                        for i, c in enumerate(sim_mcols)
                    ])
                    pct_note = " _(percentil interpolado — no alcanza el mínimo del pool)_"

                # Distancia euclídea vectorizada
                M = pool[pcols].fillna(0).values.astype(float)
                pool["_dist"] = np.sqrt(((M - ref_vec) ** 2).sum(axis=1))
                max_dist      = 100.0 * np.sqrt(len(sim_mcols))
                pool["_sim"]  = (100.0 * (1.0 - pool["_dist"] / max_dist)).clip(0, 100).round(1)

                cands = (
                    pool[pool["player_id"] != ref_pid]
                    .sort_values("_dist")
                    .head(int(n_similar))
                )

                st.caption(
                    f"**{ref_row['player']}** ({POS_LABEL.get(ref_pos, ref_pos)}, "
                    f"{ref_row['team']}){pct_note}  ·  "
                    f"{len(sim_mlbls)} variables  ·  pool: {len(pool)} jugadores"
                )

                # ── Tabla ──────────────────────────────────────────────────────────
                def _trow(r, prefix=""):
                    d = {
                        "Jugador":   prefix + r["player"],
                        "Equipo":    r["team"],
                        "Pos.":      r.get("pos_label") or "?",
                        "Min.":      int(r["minutes"]),
                        "Similitud": f"{r['_sim']:.1f}%",
                    }
                    for lbl, col in zip(sim_mlbls, sim_mcols):
                        d[lbl] = round(float(r.get(col, 0) or 0), 2)
                    return d

                tbl = pd.DataFrame([_trow(r) for _, r in cands.iterrows()])
                st.dataframe(
                    tbl.set_index("Jugador"),
                    use_container_width=True,
                    height=min(600, (len(tbl) + 1) * 38 + 40),
                )

                # ── Radar: referencia vs top 3 ─────────────────────────────────────
                if not cands.empty:
                    st.divider()
                    top3        = cands.head(3)
                    top3_shorts = [r["player"].split()[-1] for _, r in top3.iterrows()]
                    st.subheader(
                        f"Radar: {ref_row['player'].split()[-1]} vs "
                        f"{', '.join(top3_shorts)}"
                    )

                    # Construir lista: (player_id, name, is_ref, pct_vector)
                    radar_list = [(ref_pid, ref_row["player"], True, ref_vec)]
                    for _, r in top3.iterrows():
                        pid_r   = int(r["player_id"])
                        pool_r  = pool[pool["player_id"] == pid_r]
                        if pool_r.empty:
                            continue
                        pv = pool_r.iloc[0][pcols].fillna(0).values.astype(float)
                        radar_list.append((pid_r, r["player"], False, pv))

                    fig_r  = go.Figure()
                    colors = px.colors.qualitative.Bold

                    for i, (pid_r, pname_r, is_ref_r, pct_vals) in enumerate(radar_list):
                        base_r   = sim_base[sim_base["player_id"] == pid_r]
                        raw_vals = [
                            round(float(base_r.iloc[0].get(c, 0) or 0), 2)
                            if not base_r.empty else 0.0
                            for c in sim_mcols
                        ]
                        vals_c = list(pct_vals) + [pct_vals[0]]
                        thet_c = sim_mlbls + [sim_mlbls[0]]
                        hover_txt = "<br>".join(
                            f"{l}: {v}  (pct {p:.0f}°)"
                            for l, v, p in zip(sim_mlbls, raw_vals, pct_vals)
                        )
                        color  = colors[i % len(colors)]
                        nlabel = f"★ {pname_r}" if is_ref_r else pname_r

                        fig_r.add_trace(go.Scatterpolar(
                            r=vals_c, theta=thet_c,
                            fill="toself", name=nlabel,
                            line=dict(color=color, width=3 if is_ref_r else 1.8),
                            fillcolor=color,
                            opacity=0.28 if is_ref_r else 0.12,
                            hovertemplate=f"<b>{pname_r}</b><br>{hover_txt}<extra></extra>",
                        ))

                    fig_r.update_layout(
                        **PLOTLY_THEME,
                        polar=dict(
                            bgcolor=PANEL,
                            radialaxis=dict(
                                visible=True, range=[0, 100],
                                tickfont=dict(color="white", size=11),
                                gridcolor="#3a5068", linecolor="#3a5068",
                                tickvals=[20, 40, 60, 80, 100],
                            ),
                            angularaxis=dict(
                                tickfont=dict(color="white", size=13),
                                gridcolor="#3a5068", linecolor="#3a5068",
                            ),
                        ),
                        legend=dict(font=dict(color="white", size=13), bgcolor=PANEL,
                                    bordercolor="#3a5068", borderwidth=1),
                        height=580, showlegend=True,
                        title=dict(
                            text=f"Perfil percentil — {ref_row['player']} vs similares",
                            font=dict(color="white", size=14),
                        ),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)


with tab10:
    show_colo_xi_tab()

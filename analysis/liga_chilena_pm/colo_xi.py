import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


CARPETA_BASE = Path(__file__).resolve().parent
CARPETA_PROYECTO = CARPETA_BASE.parents[1]
REPORTES_LIGA = CARPETA_BASE / "reports"
REPORTES_COLO = CARPETA_PROYECTO / "analysis" / "colo_colo_plus_minus" / "reports"
CRUDOS_COLO = CARPETA_PROYECTO / "analysis" / "colo_colo_plus_minus" / "data" / "raw"

ETIQUETA_POSICION = {
    "G": "Arquero",
    "D": "Defensa",
    "M": "Mediocampistas",
    "F": "Delanteros",
}
ORDEN_LINEAS = ["D", "M", "F", "G"]
ORDEN_RESUMEN = {"G": 0, "D": 1, "M": 2, "F": 3}
XI_SUGERIDO = {
    "Arquero": ["Gabriel Maureira"],
    # Orden visual tentativo de izquierda a derecha, no inferido automaticamente:
    # los datos base solo entregan G/D/M/F, no lateralidad ni coordenadas.
    "Defensa": ["Diego Ulloa", "Jonathan Villagra", "Joaquín Sosa", "Jeyson Rojas"],
    "Mediocampistas": ["Claudio Aquino", "Arturo Vidal", "Tomás Alarcón", "Víctor Felipe Méndez"],
    "Delanteros": ["Lautaro Pastrán", "Maximiliano Romero"],
}
TITULARES_XI = {jugador for jugadores in XI_SUGERIDO.values() for jugador in jugadores}


def corregir_texto(valor):
    if not isinstance(valor, str):
        return valor
    try:
        return valor.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return valor


def corregir_columnas_texto(tabla):
    salida = tabla.copy()
    for columna in salida.select_dtypes(include="object").columns:
        salida[columna] = salida[columna].map(corregir_texto)
    return salida


def numero(valor):
    if pd.isna(valor):
        return ""
    return f"{float(valor):.2f}"


def numero_con_signo(valor):
    if pd.isna(valor):
        return ""
    return f"{float(valor):+.2f}"


def apellido(nombre):
    return str(nombre).split()[-1]


def clave_grupo(apellidos):
    return " · ".join(sorted(apellidos))


def separar_apellidos_grupo(grupo):
    return [parte.strip() for parte in re.split(r"\s*[·•]\s*", str(grupo)) if parte.strip()]


def normalizar_grupo(grupo):
    return clave_grupo(separar_apellidos_grupo(grupo))


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def segundo_incidente(incidente):
    segundos = incidente.get("timeSeconds")
    if segundos is not None:
        return int(segundos)
    minuto = incidente.get("time")
    return int(minuto) * 60 if minuto is not None else 0


def etiqueta_tramo(inicio_seg, fin_seg):
    inicio = int(round(inicio_seg / 60))
    fin = int(round(fin_seg / 60))
    return f"{inicio}-{fin}"


@st.cache_data
def cargar_datos_colo_xi():
    mas_menos = corregir_columnas_texto(pd.read_csv(REPORTES_COLO / "plus_minus_colo_colo.csv"))
    grupos = corregir_columnas_texto(pd.read_csv(REPORTES_COLO / "lineup_combos.csv"))
    indice = corregir_columnas_texto(pd.read_csv(REPORTES_LIGA / "player_index.csv"))
    estadisticas = corregir_columnas_texto(pd.read_csv(REPORTES_LIGA / "player_stats.csv"))

    indice = indice[indice["team"].eq("Colo-Colo")].copy()
    estadisticas = estadisticas[estadisticas["team"].eq("Colo-Colo")].copy()
    estadisticas = estadisticas.merge(
        indice[["player_id", "position", "pos_label"]],
        on="player_id",
        how="left",
    )
    estadisticas = estadisticas.merge(
        mas_menos[["player_id", "partidos", "gf_on", "ga_on", "plus_minus", "pm_per90"]],
        on="player_id",
        how="left",
    )
    estadisticas["score_rol"] = (
        estadisticas["pm_per90"].fillna(0).clip(-1.5, 1.5) * 0.35
        + estadisticas["ga_per90"].fillna(0).clip(0, 2.5) * 0.18
        + estadisticas["kp_per90"].fillna(0).clip(0, 4) * 0.08
        + estadisticas["rec_per90"].fillna(0).clip(0, 8) * 0.04
        + estadisticas["duel_win_pct"].fillna(45).clip(0, 75) / 100 * 0.15
        + estadisticas["minutes"].fillna(0).clip(0, 900) / 900 * 0.20
    ).round(3)
    return mas_menos, grupos, estadisticas


@st.cache_data
def cargar_impacto_suplentes():
    ruta = REPORTES_LIGA / "sub_impact_by_player.csv"
    if not ruta.exists():
        return pd.DataFrame()
    suplentes = corregir_columnas_texto(pd.read_csv(ruta))
    suplentes = suplentes[suplentes["team"].eq("Colo-Colo")].copy()
    return suplentes


@st.cache_data
def construir_posiciones_canonicas():
    conteos = {}
    for carpeta_partido in sorted(ruta for ruta in CRUDOS_COLO.iterdir() if ruta.is_dir()):
        meta = leer_json(carpeta_partido / "meta.json")
        lineups = leer_json(carpeta_partido / "lineups.json")
        if not meta or not lineups:
            continue
        lado = "home" if meta["home_team"] == "Colo-Colo" else "away"
        for entrada in lineups.get(lado, {}).get("players", []):
            jugador = entrada.get("player", {})
            jugador_id = jugador.get("id")
            posicion = entrada.get("position") or jugador.get("position")
            if jugador_id is None or not posicion:
                continue
            conteos.setdefault(jugador_id, {})
            conteos[jugador_id][posicion] = conteos[jugador_id].get(posicion, 0) + 1
    return {
        jugador_id: max(posiciones.items(), key=lambda item: item[1])[0]
        for jugador_id, posiciones in conteos.items()
    }


def construir_intervalos_jugadores(lineups, incidentes, colo_es_local, posiciones_canonicas=None):
    posiciones_canonicas = posiciones_canonicas or {}
    lado = "home" if colo_es_local else "away"
    jugadores = {}
    activos = {}

    for entrada in lineups.get(lado, {}).get("players", []):
        jugador = entrada.get("player", {})
        jugador_id = jugador.get("id")
        if jugador_id is None:
            continue
        nombre = corregir_texto(jugador.get("name") or jugador.get("shortName") or f"id_{jugador_id}")
        jugadores[jugador_id] = {
            "nombre": nombre,
            "apellido": apellido(nombre),
            "posicion": posiciones_canonicas.get(jugador_id) or entrada.get("position") or jugador.get("position"),
            "intervalos": [],
            "estadisticas": entrada.get("statistics") or {},
        }
        if not entrada.get("substitute", True):
            activos[jugador_id] = 0

    cambios = []
    for incidente in incidentes.get("incidents") or []:
        if incidente.get("incidentType") != "substitution":
            continue
        if incidente.get("isHome", False) != colo_es_local:
            continue
        entra = incidente.get("playerIn") or {}
        sale = incidente.get("playerOut") or {}
        cambios.append({
            "segundo": min(segundo_incidente(incidente), 5400),
            "entra_id": entra.get("id"),
            "sale_id": sale.get("id"),
        })

    for cambio in sorted(cambios, key=lambda item: item["segundo"]):
        if cambio["sale_id"] in activos:
            jugadores[cambio["sale_id"]]["intervalos"].append(
                (activos.pop(cambio["sale_id"]), cambio["segundo"])
            )
        if cambio["entra_id"] in jugadores:
            activos[cambio["entra_id"]] = cambio["segundo"]

    for jugador_id, inicio in activos.items():
        jugadores[jugador_id]["intervalos"].append((inicio, 5400))
    return jugadores


def intersectar_dos_intervalos(intervalos_a, intervalos_b):
    compartidos = []
    for a0, a1 in intervalos_a:
        for b0, b1 in intervalos_b:
            inicio, fin = max(a0, b0), min(a1, b1)
            if inicio < fin:
                compartidos.append((inicio, fin))
    return compartidos


def intersectar_varios_intervalos(lista_intervalos):
    if not lista_intervalos:
        return []
    compartidos = lista_intervalos[0]
    for intervalos in lista_intervalos[1:]:
        compartidos = intersectar_dos_intervalos(compartidos, intervalos)
    return compartidos


def fusionar_tramos_contiguos(tramos):
    if not tramos:
        return []
    ordenados = sorted(tramos)
    fusionados = [ordenados[0]]
    for inicio, fin in ordenados[1:]:
        ultimo_inicio, ultimo_fin = fusionados[-1]
        if inicio <= ultimo_fin:
            fusionados[-1] = (ultimo_inicio, max(ultimo_fin, fin))
        else:
            fusionados.append((inicio, fin))
    return fusionados


def ocurre_en_tramos(segundo, intervalos):
    return any(inicio <= segundo < fin for inicio, fin in intervalos)


def resumen_estadistico(jugador):
    estadisticas = jugador.get("estadisticas") or {}
    return {
        "goles": estadisticas.get("goals") or 0,
        "asistencias": estadisticas.get("goalAssist") or 0,
        "tiros": estadisticas.get("totalShots") or 0,
        "pases_clave": estadisticas.get("keyPass") or 0,
        "centros": estadisticas.get("totalCross") or 0,
        "centros_precisos": estadisticas.get("accurateCross") or 0,
        "acciones_defensivas": (estadisticas.get("wonTackle") or 0)
        + (estadisticas.get("interceptionWon") or 0)
        + (estadisticas.get("ballRecovery") or 0),
    }


def buscar_jugadores_por_grupo(jugadores, grupo, posicion=None):
    apellidos_grupo = separar_apellidos_grupo(grupo)
    por_apellido = {}
    for datos in jugadores.values():
        por_apellido.setdefault(datos["apellido"], []).append(datos)

    seleccionados = []
    for apellido_grupo in apellidos_grupo:
        candidatos = por_apellido.get(apellido_grupo, [])
        if posicion:
            candidatos_posicion = [jugador for jugador in candidatos if jugador.get("posicion") == posicion]
            if candidatos_posicion:
                candidatos = candidatos_posicion
        if not candidatos:
            return []
        seleccionados.append(candidatos[0])
    return seleccionados


@st.cache_data
def desglose_grupo_por_partido(grupo, posicion=None):
    filas = []
    if not CRUDOS_COLO.exists():
        return pd.DataFrame()
    posiciones_canonicas = construir_posiciones_canonicas()
    grupo_normalizado = normalizar_grupo(grupo)

    for carpeta_partido in sorted(ruta for ruta in CRUDOS_COLO.iterdir() if ruta.is_dir()):
        meta = leer_json(carpeta_partido / "meta.json")
        lineups = leer_json(carpeta_partido / "lineups.json")
        incidentes = leer_json(carpeta_partido / "incidents.json")
        if not meta or not lineups or not incidentes:
            continue

        colo_es_local = meta["home_team"] == "Colo-Colo"
        jugadores = construir_intervalos_jugadores(lineups, incidentes, colo_es_local, posiciones_canonicas)
        goles = []
        for incidente in incidentes.get("incidents") or []:
            if incidente.get("incidentType") != "goal":
                continue
            goles.append({
                "segundo": min(segundo_incidente(incidente), 5400),
                "es_gol_colo": incidente.get("isHome", False) == colo_es_local,
            })

        cortes = {0, 5400}
        for datos in jugadores.values():
            for inicio, fin in datos["intervalos"]:
                if 0 < inicio < 5400:
                    cortes.add(inicio)
                if 0 < fin < 5400:
                    cortes.add(fin)
        cortes = sorted(cortes)

        tramos = []
        gf = 0
        gc = 0
        jugadores_grupo = {}
        for indice in range(len(cortes) - 1):
            inicio, fin = cortes[indice], cortes[indice + 1]
            if fin <= inicio:
                continue
            punto_medio = (inicio + fin) / 2
            activos_linea = [
                datos for datos in jugadores.values()
                if datos.get("posicion") == posicion and ocurre_en_tramos(punto_medio, datos["intervalos"])
            ]
            grupo_activo = clave_grupo([datos["apellido"] for datos in activos_linea])
            if grupo_activo != grupo_normalizado:
                continue
            tramos.append((inicio, fin))
            for datos in activos_linea:
                jugadores_grupo[datos["nombre"]] = datos
            gf += sum(1 for gol in goles if inicio <= gol["segundo"] < fin and gol["es_gol_colo"])
            gc += sum(1 for gol in goles if inicio <= gol["segundo"] < fin and not gol["es_gol_colo"])

        minutos_juntos = sum(fin - inicio for inicio, fin in tramos) / 60
        if minutos_juntos <= 0:
            continue
        tramos_mostrados = fusionar_tramos_contiguos(tramos)

        estadisticas_grupo = [resumen_estadistico(jugador) for jugador in jugadores_grupo.values()]
        home_score = meta.get("home_score")
        away_score = meta.get("away_score")
        colo_score = home_score if colo_es_local else away_score
        rival_score = away_score if colo_es_local else home_score
        estado = "Ganó" if colo_score > rival_score else "Empató" if colo_score == rival_score else "Perdió"

        filas.append({
            "Fecha": pd.to_datetime(meta.get("start_timestamp"), unit="s").date(),
            "Rival": corregir_texto(meta["away_team"] if colo_es_local else meta["home_team"]),
            "Sede": "Local" if colo_es_local else "Visita",
            "Resultado": f"{corregir_texto(meta['home_team'])} {home_score}-{away_score} {corregir_texto(meta['away_team'])}",
            "Estado": estado,
            "Tramos juntos": ", ".join(etiqueta_tramo(inicio, fin) for inicio, fin in tramos_mostrados),
            "Min juntos": round(minutos_juntos, 1),
            "GF juntos": gf,
            "GC juntos": gc,
            "+/- juntos": gf - gc,
            "Goles grupo": sum(item["goles"] for item in estadisticas_grupo),
            "Asistencias grupo": sum(item["asistencias"] for item in estadisticas_grupo),
            "Tiros grupo": sum(item["tiros"] for item in estadisticas_grupo),
            "Pases clave grupo": sum(item["pases_clave"] for item in estadisticas_grupo),
            "Acciones defensivas grupo": sum(item["acciones_defensivas"] for item in estadisticas_grupo),
            "Centros grupo": sum(item["centros"] for item in estadisticas_grupo),
            "Centros precisos grupo": sum(item["centros_precisos"] for item in estadisticas_grupo),
        })

    return pd.DataFrame(filas).sort_values("Fecha") if filas else pd.DataFrame()


@st.cache_data
def construir_grupos_exactos_desde_partidos():
    filas_por_grupo = {}
    posiciones_canonicas = construir_posiciones_canonicas()
    if not CRUDOS_COLO.exists():
        return pd.DataFrame()

    for carpeta_partido in sorted(ruta for ruta in CRUDOS_COLO.iterdir() if ruta.is_dir()):
        meta = leer_json(carpeta_partido / "meta.json")
        lineups = leer_json(carpeta_partido / "lineups.json")
        incidentes = leer_json(carpeta_partido / "incidents.json")
        if not meta or not lineups or not incidentes:
            continue
        colo_es_local = meta["home_team"] == "Colo-Colo"
        jugadores = construir_intervalos_jugadores(lineups, incidentes, colo_es_local, posiciones_canonicas)

        goles = []
        for incidente in incidentes.get("incidents") or []:
            if incidente.get("incidentType") == "goal":
                goles.append({
                    "segundo": min(segundo_incidente(incidente), 5400),
                    "es_gol_colo": incidente.get("isHome", False) == colo_es_local,
                })

        cortes = {0, 5400}
        for datos in jugadores.values():
            for inicio, fin in datos["intervalos"]:
                if 0 < inicio < 5400:
                    cortes.add(inicio)
                if 0 < fin < 5400:
                    cortes.add(fin)
        cortes = sorted(cortes)

        for indice in range(len(cortes) - 1):
            inicio, fin = cortes[indice], cortes[indice + 1]
            if fin <= inicio:
                continue
            punto_medio = (inicio + fin) / 2
            minutos = (fin - inicio) / 60
            gf = sum(1 for gol in goles if inicio <= gol["segundo"] < fin and gol["es_gol_colo"])
            gc = sum(1 for gol in goles if inicio <= gol["segundo"] < fin and not gol["es_gol_colo"])
            for posicion in ETIQUETA_POSICION:
                activos = [
                    datos for datos in jugadores.values()
                    if datos.get("posicion") == posicion and ocurre_en_tramos(punto_medio, datos["intervalos"])
                ]
                if not activos:
                    continue
                grupo = clave_grupo([datos["apellido"] for datos in activos])
                clave = (posicion, grupo)
                if clave not in filas_por_grupo:
                    filas_por_grupo[clave] = {"pos": posicion, "combo": grupo, "min": 0.0, "gf": 0, "ga": 0}
                filas_por_grupo[clave]["min"] += minutos
                filas_por_grupo[clave]["gf"] += gf
                filas_por_grupo[clave]["ga"] += gc

    filas = []
    for datos in filas_por_grupo.values():
        pm = datos["gf"] - datos["ga"]
        minutos = datos["min"]
        filas.append({
            "pos": datos["pos"],
            "combo": datos["combo"],
            "min": round(minutos, 1),
            "gf": int(datos["gf"]),
            "ga": int(datos["ga"]),
            "pm": int(pm),
            "pm_per90": round(pm / minutos * 90, 2) if minutos else 0,
        })
    return pd.DataFrame(filas)


def ficha_jugador(nombre):
    return f'<div class="jugador-cancha">{nombre}</div>'


def fila_cancha(jugadores):
    return '<div class="fila-cancha">' + "".join(ficha_jugador(jugador) for jugador in jugadores) + "</div>"


def mostrar_xi_en_cancha():
    html = f"""
    <style>
    .cancha-codex {{
        background: linear-gradient(180deg, #1f7a45 0%, #16643a 100%);
        border: 2px solid rgba(255,255,255,0.78);
        border-radius: 12px;
        padding: 20px 18px;
        margin: 8px 0 18px 0;
        min-height: 520px;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.22);
    }}
    .cancha-codex::before {{
        content: "";
        position: absolute;
        inset: 18px;
        border: 1px solid rgba(255,255,255,0.45);
        border-radius: 8px;
        pointer-events: none;
    }}
    .cancha-codex::after {{
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 150px;
        height: 150px;
        transform: translate(-50%, -50%);
        border: 1px solid rgba(255,255,255,0.40);
        border-radius: 50%;
        pointer-events: none;
    }}
    .linea-media-cancha {{
        position: absolute;
        left: 18px;
        right: 18px;
        top: 50%;
        border-top: 1px solid rgba(255,255,255,0.35);
    }}
    .area-arquero {{
        position: absolute;
        left: 34%;
        right: 34%;
        bottom: 18px;
        height: 78px;
        border: 1px solid rgba(255,255,255,0.35);
        border-bottom: 0;
        border-radius: 8px 8px 0 0;
    }}
    .linea-cancha {{
        position: relative;
        z-index: 1;
        margin: 0 0 26px 0;
    }}
    .titulo-linea {{
        color: rgba(255,255,255,0.82);
        font-size: 13px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .fila-cancha {{
        display: flex;
        gap: 12px;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
    }}
    .jugador-cancha {{
        background: rgba(15, 25, 35, 0.88);
        border: 1px solid rgba(255,255,255,0.60);
        color: #ffffff;
        border-radius: 999px;
        padding: 8px 14px;
        min-width: 145px;
        text-align: center;
        font-weight: 650;
        box-shadow: 0 6px 16px rgba(0,0,0,0.22);
    }}
    </style>
    <div class="cancha-codex">
        <div class="linea-media-cancha"></div>
        <div class="area-arquero"></div>
        <div class="linea-cancha">
            <div class="titulo-linea">Delanteros</div>
            {fila_cancha(XI_SUGERIDO["Delanteros"])}
        </div>
        <div class="linea-cancha">
            <div class="titulo-linea">Mediocampistas</div>
            {fila_cancha(XI_SUGERIDO["Mediocampistas"])}
        </div>
        <div class="linea-cancha">
            <div class="titulo-linea">Defensa</div>
            {fila_cancha(XI_SUGERIDO["Defensa"])}
        </div>
        <div class="linea-cancha">
            <div class="titulo-linea">Arquero</div>
            {fila_cancha(XI_SUGERIDO["Arquero"])}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.caption(
        "Nota: la cancha muestra un orden visual tentativo. El scraping actual identifica líneas "
        "(arquero/defensa/mediocampo/delantera), pero no roles finos como LD, LI o central."
    )


def mostrar_grupos_clave(grupos):
    grupos_mostrados = grupos[grupos["combo"].isin([
        "Maureira",
        "Rojas · Sosa · Ulloa · Villagra",
        "Alarcón · Aquino · Méndez · Vidal",
        "Pastrán · Romero",
    ])].copy()
    grupos_mostrados["Línea"] = grupos_mostrados["pos"].map(ETIQUETA_POSICION)
    grupos_mostrados["Orden"] = grupos_mostrados["pos"].map(ORDEN_RESUMEN)
    grupos_mostrados = grupos_mostrados.rename(columns={
        "combo": "Grupo",
        "min": "Min",
        "gf": "GF",
        "ga": "GC",
        "pm": "+/-",
        "pm_per90": "+/-90",
    })
    grupos_mostrados = grupos_mostrados.sort_values("Orden")
    st.dataframe(
        grupos_mostrados[["Línea", "Grupo", "Min", "GF", "GC", "+/-", "+/-90"]],
        use_container_width=True,
        hide_index=True,
    )
    if not grupos_mostrados.empty:
        defensa = grupos_mostrados[grupos_mostrados["Línea"].eq("Defensa")]
        medio = grupos_mostrados[grupos_mostrados["Línea"].eq("Mediocampistas")]
        ataque = grupos_mostrados[grupos_mostrados["Línea"].eq("Delanteros")]
        arquero = grupos_mostrados[grupos_mostrados["Línea"].eq("Arquero")]

        conclusiones = []
        if not defensa.empty:
            fila = defensa.iloc[0]
            conclusiones.append(
                f"Defensa: {fila['Grupo']} es la base mas estable por volumen "
                f"({numero(fila['Min'])} min). No es la de mejor +/-90, pero si la muestra mas confiable."
            )
        if not medio.empty:
            fila = medio.iloc[0]
            conclusiones.append(
                f"Mediocampo: {fila['Grupo']} deja buen balance ({numero_con_signo(fila['+/-90'])} +/-90) "
                f"sin goles recibidos en sus tramos medidos."
            )
        if not ataque.empty:
            fila = ataque.iloc[0]
            conclusiones.append(
                f"Ataque: {fila['Grupo']} es la senal mas clara del XI: "
                f"{int(fila['GF'])}-{int(fila['GC'])} en goles y {numero_con_signo(fila['+/-90'])} +/-90."
            )
        if not arquero.empty:
            fila = arquero.iloc[0]
            conclusiones.append(
                f"Arquero: {fila['Grupo']} aparece mejor por impacto por 90, pero con menos minutos que De Paul; "
                "conviene leerlo como apuesta de rendimiento, no como certeza definitiva."
            )

        st.markdown("**Lectura de la recomendación**")
        for conclusion in conclusiones:
            st.write(f"- {conclusion}")
        st.info(
            "Conclusión: el 4-4-2 no sale por gusto táctico, sino porque conserva una defensa de 4 ya usada, "
            "un mediocampo de 4 con balance positivo y una dupla ofensiva que produjo la mejor señal compartida. "
            "La duda principal no es la estructura, sino los nombres finos por banda/rol, porque el dato actual no trae lateralidad."
        )


def mostrar_comparacion_puesto_a_puesto(estadisticas):
    st.markdown("**Comparación puesto a puesto**")
    posicion = st.selectbox(
        "Puesto",
        ["G", "D", "M", "F"],
        index=1,
        format_func=lambda valor: ETIQUETA_POSICION.get(valor, valor),
        key="colo_puesto",
    )
    minimo_minutos = st.slider("Minutos mínimos", 0, 900, 120, step=30, key="colo_min_puesto")
    candidatos = estadisticas[
        estadisticas["position"].eq(posicion) & estadisticas["minutes"].fillna(0).ge(minimo_minutos)
    ].sort_values(["score_rol", "minutes"], ascending=False)

    if candidatos.empty:
        st.info("No hay jugadores para ese filtro.")
        return

    mejor = candidatos.iloc[0]
    st.success(
        f"Recomendado en {ETIQUETA_POSICION.get(posicion, posicion).lower()}: {mejor['player']} "
        f"({numero_con_signo(mejor['pm_per90'])} +/-90, {numero(mejor['minutes'])} min)."
    )
    columnas = [
        "player", "minutes", "plus_minus", "pm_per90", "goals", "assists",
        "ga_per90", "kp_per90", "crosses_per90", "pass_acc_pct", "int_per90",
        "clr_per90", "rec_per90", "tackle_acc_pct", "duel_win_pct", "score_rol",
    ]
    columnas = [columna for columna in columnas if columna in candidatos.columns]
    tabla = candidatos[columnas].rename(columns={
        "player": "Jugador",
        "minutes": "Min",
        "plus_minus": "+/-",
        "pm_per90": "+/-90",
        "goals": "Goles",
        "assists": "Asistencias",
        "ga_per90": "G+A90",
        "kp_per90": "Pases clave90",
        "crosses_per90": "Centros90",
        "pass_acc_pct": "% pase",
        "int_per90": "Intercepciones90",
        "clr_per90": "Despejes90",
        "rec_per90": "Recuperaciones90",
        "tackle_acc_pct": "% entradas",
        "duel_win_pct": "% duelos",
        "score_rol": "Score rol",
    })
    st.dataframe(tabla, use_container_width=True, hide_index=True)


def mostrar_comparacion_de_grupos(grupos):
    st.markdown("**Comparación de grupos por posición**")
    linea = st.selectbox(
        "Línea",
        ORDEN_LINEAS,
        format_func=lambda valor: ETIQUETA_POSICION.get(valor, valor),
        key="colo_linea_grupos",
    )
    usar_grupos_cortos = st.checkbox(
        "Incluir grupos cortos",
        value=(linea == "F"),
        help="Incluye combinaciones de menos de 45 minutos. Útil para detectar tramos post-cambios y revulsivos.",
        key="colo_incluir_grupos_cortos",
    )
    minimo_grupo = st.slider(
        "Minutos mínimos del grupo",
        0,
        180,
        0 if usar_grupos_cortos and linea == "F" else 45,
        step=5,
        key="colo_min_grupo",
    )
    fuente_grupos = construir_grupos_exactos_desde_partidos() if usar_grupos_cortos else grupos
    grupos_linea = fuente_grupos[
        fuente_grupos["pos"].eq(linea) & fuente_grupos["min"].fillna(0).ge(minimo_grupo)
    ].copy()
    grupos_linea = grupos_linea.sort_values(["pm_per90", "min"], ascending=False)
    grupo_mas_estable = grupos_linea.sort_values("min", ascending=False).iloc[0] if not grupos_linea.empty else None
    grupo_mayor_impacto = grupos_linea.iloc[0] if not grupos_linea.empty else None
    tabla = grupos_linea.rename(columns={
        "combo": "Grupo",
        "min": "Min juntos",
        "gf": "GF juntos",
        "ga": "GC juntos",
        "pm": "+/- juntos",
        "pm_per90": "+/-90",
    })
    st.dataframe(
        tabla[["Grupo", "Min juntos", "GF juntos", "GC juntos", "+/- juntos", "+/-90"]],
        use_container_width=True,
        hide_index=True,
    )
    if grupo_mas_estable is not None and grupo_mayor_impacto is not None:
        if grupo_mas_estable["combo"] == grupo_mayor_impacto["combo"]:
            st.success(
                f"Conclusión para {ETIQUETA_POSICION[linea].lower()}: "
                f"{grupo_mas_estable['combo']} combina volumen e impacto. "
                f"Es la opción más defendible con estos datos."
            )
    if usar_grupos_cortos and linea == "F":
        st.info(
            "Nota: los grupos cortos permiten ver tramos como Correa-Cuevas vs Coquimbo, "
            "que no aparecen en la tabla original por tener menos de 45 minutos."
        )
    else:
            st.warning(
                f"Lectura para {ETIQUETA_POSICION[linea].lower()}: "
                f"{grupo_mayor_impacto['combo']} tiene el mejor +/-90 "
                f"({numero_con_signo(grupo_mayor_impacto['pm_per90'])}), "
                f"pero {grupo_mas_estable['combo']} tiene más muestra "
                f"({numero(grupo_mas_estable['min'])} min). "
                "Para titularidad pesa más la estabilidad; para ajuste puntual pesa más el impacto."
            )

    opciones = tabla["Grupo"].tolist()
    if not opciones:
        st.info("No hay grupos para esta línea.")
        return

    grupo_elegido = st.selectbox("Grupo a desglosar por partido", opciones, key="colo_grupo_elegido")
    if not st.button("Analizar grupo", key="colo_boton_grupo"):
        st.info("Pulsa analizar para ver partidos, tramos juntos e impacto del grupo.")
        return

    desglose = desglose_grupo_por_partido(grupo_elegido, linea)
    if desglose.empty:
        st.info("No encontré minutos compartidos para ese grupo.")
        return

    total_min = desglose["Min juntos"].sum()
    total_gf = desglose["GF juntos"].sum()
    total_gc = desglose["GC juntos"].sum()
    total_pm = desglose["+/- juntos"].sum()
    pm90 = total_pm / total_min * 90 if total_min else np.nan
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Min juntos", numero(total_min))
    m2.metric("GF/GC juntos", f"{int(total_gf)}/{int(total_gc)}")
    m3.metric("+/- juntos", f"{int(total_pm):+d}")
    m4.metric("+/-90 grupo", numero_con_signo(pm90))
    st.write(
        f"{grupo_elegido}: {numero(total_min)} minutos juntos. "
        f"Colo Colo quedó {int(total_gf)}-{int(total_gc)} durante esos tramos."
    )
    if total_min < 120:
        st.warning(
            "Conclusión del grupo seleccionado: señal interesante, pero muestra baja. "
            "Sirve para detectar una variante, no para afirmar titularidad."
        )
    elif total_pm > 0:
        st.success(
            "Conclusión del grupo seleccionado: balance positivo con muestra razonable. "
            "Es una combinación candidata a sostenerse o repetirse."
        )
    elif total_pm == 0:
        st.info(
            "Conclusión del grupo seleccionado: balance neutro. Puede ser útil si aporta control, "
            "pero no aparece como ventaja clara en goles durante sus tramos."
        )
    else:
        st.error(
            "Conclusión del grupo seleccionado: balance negativo. Conviene revisar rivales, contexto "
            "y si el problema fue estructural o de un partido puntual."
        )
    st.dataframe(desglose, use_container_width=True, hide_index=True)


def rol_suplente(fila):
    posicion = fila.get("position")
    if posicion == "G":
        return "Arquero suplente"
    if fila.get("g_per90", 0) >= 0.65 or fila.get("sh_per90", 0) >= 4.5:
        return "Gol / remate"
    if fila.get("kp_per90", 0) >= 2.0 or fila.get("bcc_per90", 0) >= 0.35:
        return "Creatividad"
    if fila.get("crosses_per90", 0) >= 3.0:
        return "Banda / centros"
    if fila.get("rec_per90", 0) >= 5.0 or fila.get("duel_win_pct", 0) >= 52:
        return "Cierre / equilibrio"
    return "Rotación"


def mostrar_tres_revulsivos(estadisticas):
    st.markdown("**Tres revulsivos recomendados**")
    impacto = cargar_impacto_suplentes()
    if impacto.empty:
        st.info("No hay datos de impacto de sustituciones para recomendar revulsivos.")
        return

    banca = estadisticas[~estadisticas["player"].isin(TITULARES_XI)].copy()
    banca = banca.merge(
        impacto[["player_in", "entradas", "pm_after_avg", "pm_after_sum"]],
        left_on="player",
        right_on="player_in",
        how="left",
    )
    banca["entradas"] = banca["entradas"].fillna(0)
    banca["pm_after_avg"] = banca["pm_after_avg"].fillna(0)
    banca["pm_after_sum"] = banca["pm_after_sum"].fillna(0)
    banca["score_revulsivo"] = (
        banca["pm_after_avg"].clip(-1, 1.5) * 0.30
        + banca["pm_per90"].fillna(0).clip(-1.5, 2.5) * 0.20
        + banca["ga_per90"].fillna(0).clip(0, 1.6) * 0.15
        + banca["sh_per90"].fillna(0).clip(0, 6) * 0.06
        + banca["kp_per90"].fillna(0).clip(0, 4) * 0.06
        + banca["rec_per90"].fillna(0).clip(0, 8) * 0.03
        + (banca["entradas"].clip(0, 8) / 8) * 0.20
    ).round(3)
    banca["Rol sugerido"] = banca.apply(rol_suplente, axis=1)
    candidatos = banca[banca["entradas"].ge(1)].copy()
    if candidatos.empty:
        st.info("No encontré candidatos con ingresos registrados.")
        return

    preferidos = ["Yastin Cuevas", "Javier Correa", "Francisco Marchant"]
    seleccion = candidatos[candidatos["player"].isin(preferidos)].copy()
    faltantes = 3 - len(seleccion)
    if faltantes > 0:
        extra = candidatos[~candidatos["player"].isin(seleccion["player"])].sort_values(
            ["score_revulsivo", "pm_after_avg", "entradas"],
            ascending=False,
        ).head(faltantes)
        seleccion = pd.concat([seleccion, extra], ignore_index=True)
    seleccion["orden"] = seleccion["player"].map({nombre: i for i, nombre in enumerate(preferidos)}).fillna(99)
    seleccion = seleccion.sort_values(["orden", "score_revulsivo"], ascending=[True, False]).head(3)

    columnas = st.columns(3)
    for columna, (_, fila) in zip(columnas, seleccion.iterrows()):
        with columna:
            st.metric(fila["player"], fila["Rol sugerido"])
            st.write(
                f"Entradas: {int(fila['entradas'])} | +/- tras entrar: "
                f"{numero_con_signo(fila['pm_after_avg'])} prom."
            )
            if fila["player"] == "Yastin Cuevas":
                st.write(
                    f"Revulsivo directo: {numero(fila['ga_per90'])} G+A90, "
                    f"{numero(fila['sh_per90'])} tiros90 y buena señal de impacto al entrar."
                )
            elif fila["player"] == "Javier Correa":
                st.write(
                    f"Plan de gol: {int(fila['goals'])} goles, {numero(fila['sh_per90'])} tiros90. "
                    "Sirve si el partido pide área y volumen de remate."
                )
            elif fila["player"] == "Francisco Marchant":
                st.write(
                    f"Plan creativo: {numero(fila['kp_per90'])} pases clave90 y "
                    f"{numero(fila['rec_per90'])} recuperaciones90. Cambia ritmo sin romper tanto el medio."
                )
            else:
                st.write(
                    f"Alternativa por impacto: score {numero(fila['score_revulsivo'])}, "
                    f"{numero(fila['pm_per90'])} +/-90."
                )


def mostrar_alerta_correa_coquimbo():
    grupos_exactos = construir_grupos_exactos_desde_partidos()
    ataque = grupos_exactos[
        grupos_exactos["pos"].eq("F")
        & grupos_exactos["combo"].str.contains("Correa", na=False)
    ].sort_values(["gf", "pm_per90", "min"], ascending=False)
    if ataque.empty:
        return

    st.markdown("**Ajuste importante en delantera**")
    st.write(
        "La tabla histórica favorece Pastrán-Romero porque tiene más minutos y buen balance, "
        "pero el partido ante Coquimbo muestra algo que esa tabla ocultaba: los goles de Correa llegaron "
        "después de la salida de Romero, en un tramo ofensivo corto."
    )

    tabla = ataque.head(5).rename(columns={
        "combo": "Grupo ofensivo",
        "min": "Min",
        "gf": "GF",
        "ga": "GC",
        "pm": "+/-",
        "pm_per90": "+/-90",
    })
    st.dataframe(tabla[["Grupo ofensivo", "Min", "GF", "GC", "+/-", "+/-90"]], use_container_width=True, hide_index=True)
    st.info(
        "Conclusión: Correa no debería quedar fuera de la lectura de delantera. "
        "Con los datos actuales, Pastrán-Romero sigue siendo la dupla con mejor muestra positiva, "
        "pero Correa merece recomendación fuerte como alternativa ofensiva o primer cambio para buscar gol."
    )


def show_colo_xi_tab():
    st.subheader("Colo Colo XI - prototipo táctico")
    _, grupos, estadisticas = cargar_datos_colo_xi()
    st.caption("Recomendaciones con +/- individual, grupos por línea y estadísticas por 90.")

    with st.container(border=True):
        st.markdown("**XI sugerido por combinaciones observadas: 4-4-2**")
        mostrar_xi_en_cancha()
        st.write(
            "La estructura sale de juntar las mejores señales por línea: defensa de 4, "
            "mediocampistas de 4 y la dupla Pastrán-Romero en ataque."
        )
        st.markdown("**Conclusión principal**")
        st.write(
            "La recomendación es partir desde un 4-4-2 porque es la forma que mejor respeta las "
            "combinaciones positivas ya observadas: mantiene una línea defensiva reconocible, junta "
            "a los mediocampistas con mejor balance colectivo y aprovecha la dupla Pastrán-Romero, "
            "que es la asociación ofensiva con mejor impacto compartido."
        )
        st.write(
            "No significa que el 4-4-2 sea siempre superior: significa que, con esta muestra, es la "
            "formación menos forzada para convertir los datos en un XI inicial. Las alternativas deben "
            "evaluarse especialmente por roles de banda, donde el dato actual todavía no distingue LD/LI."
        )
        mostrar_tres_revulsivos(estadisticas)
        mostrar_alerta_correa_coquimbo()
        mostrar_grupos_clave(grupos)

    st.divider()
    mostrar_comparacion_puesto_a_puesto(estadisticas)
    st.divider()
    mostrar_comparacion_de_grupos(grupos)

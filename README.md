# futdata_v1

Proyecto de machine learning para prediccion de partidos de futbol, busqueda de apuestas con valor y analisis de jugadores de la liga chilena.

## Estructura

El repo queda separado en dos flujos principales:

- `apps/predictions/`: flujo de predicciones, cuotas y modelos.
- `apps/predictions/dashboard_streamlit.py`: dashboard principal de predicciones.
- `apps/predictions/src/`: scraping, limpieza, features, entrenamiento, prediccion y cuotas.
- `apps/predictions/files/`: datos crudos, procesados, features, modelos, reportes, cuotas y cache del flujo de predicciones.
- `apps/liga_chilena/`: flujo de plus/minus y estadisticas de jugadores de la Liga Chilena.
- `apps/liga_chilena/dashboard.py`: dashboard de Liga Chilena.
- `apps/liga_chilena/src/`: scraping, procesamiento, visualizaciones y precomputos del dashboard chileno.
- `apps/liga_chilena/data/` y `apps/liga_chilena/reports/`: datos y salidas del flujo chileno.
- `apps/liga_chilena/colo_colo_plus_minus/`: analisis historico de Colo-Colo usado por la pestaña de XI del dashboard chileno.
- `start_dashboard.py` / `start_dashboard.bat`: launcher local del dashboard de predicciones.
- `start_liga_dashboard.py`: launcher local del dashboard de Liga Chilena.
- `logs/`: logs de Streamlit.

## Dashboards

### Predicciones

```powershell
.runtime\python312\python.exe start_dashboard.py
```

Queda disponible en `http://127.0.0.1:8501`.

### Liga Chilena

```powershell
.runtime\python312\python.exe start_liga_dashboard.py
```

Queda disponible en `http://127.0.0.1:8502`.

## Predicciones

Las fechas del fin de semana mostrado viven al inicio de `apps/predictions/dashboard_streamlit.py`:

```python
WEEKEND_START_DATE = date.fromisoformat(os.environ.get("DASHBOARD_WEEKEND_START", "2026-05-08"))
WEEKEND_END_DATE = date.fromisoformat(os.environ.get("DASHBOARD_WEEKEND_END", "2026-05-11"))
```

Tambien se pueden cambiar con variables de entorno:

```powershell
$env:DASHBOARD_WEEKEND_START = "2026-05-15"
$env:DASHBOARD_WEEKEND_END = "2026-05-18"
.runtime\python312\python.exe start_dashboard.py
```

El dashboard busca `apps/predictions/files/sofascore_pipeline/reports/weekend_value_predictions_<inicio>_to_<fin>.csv`. Si no existe, genera predicciones en vivo desde los modelos.

### Actualizar un fin de semana

Para cada liga (`premier`, `la_liga`, `serie_a`, `bundesliga`, `ligue_1`):

```powershell
Push-Location apps\predictions

# 1) Scraping con partidos futuros incluidos
..\..\.runtime\python312\python.exe src\01_scrape_sofascore.py `
    --league premier --include-future --odds `
    --pipeline-root files\sofascore_pipeline

# 2) Re-entrenar modelos sin volver a scrapear
..\..\.runtime\python312\python.exe -m src.run_sofascore_pipeline --league premier --skip-scrape

Pop-Location
```

Importante: despues de correr el scraping con `--include-future`, usar `--skip-scrape` al re-entrenar. El scraping normal no incluye partidos futuros y puede sobrescribirlos.

### Pipeline antiguo FBref

```powershell
Push-Location apps\predictions

..\..\venv312\Scripts\python.exe src\01_parse_fbref_html.py --league premier
..\..\venv312\Scripts\python.exe src\02_clean_matchlogs.py --league premier
..\..\venv312\Scripts\python.exe src\03_build_features.py --league premier
..\..\venv312\Scripts\python.exe src\04_train_result_model.py --league premier
..\..\venv312\Scripts\python.exe src\05_train_goal_market_models.py --league premier
..\..\venv312\Scripts\python.exe src\06_predict_future_matches.py --league premier

Pop-Location
```

Para correr una liga secundaria completa:

```powershell
Push-Location apps\predictions

..\..\venv312\Scripts\python.exe src\run_pipeline.py --league la_liga
..\..\venv312\Scripts\python.exe src\run_pipeline.py --league serie_a
..\..\venv312\Scripts\python.exe src\run_pipeline.py --league chile

Pop-Location
```

`apps/predictions/src/07_scrape_cuotasahora_odds.py` es opcional y se usa para actualizar cuotas desde el dashboard.

## Liga Chilena

Los scripts del flujo chileno viven en `apps/liga_chilena/src/` y escriben en `apps/liga_chilena/data/` y `apps/liga_chilena/reports/`.

```powershell
venv312\Scripts\python.exe apps\liga_chilena\src\01_scrape.py
venv312\Scripts\python.exe apps\liga_chilena\src\02_calc_plus_minus.py
venv312\Scripts\python.exe apps\liga_chilena\src\04_player_stats.py
venv312\Scripts\python.exe apps\liga_chilena\src\06_precompute_events.py
```

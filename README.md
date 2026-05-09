# futdata_v1

Proyecto de machine learning para prediccion de partidos de futbol y busqueda de apuestas con valor.

## Estructura

- `dashboard_streamlit.py`: dashboard principal de predicciones y cuotas.
- `start_dashboard.py` / `start_dashboard.bat`: launcher local del dashboard.
- `src/`: pipeline de datos, features, entrenamiento, prediccion y cuotas.
- `files/01_raw/`: datos crudos por liga (HTMLs descargados a mano viven en `files/01_raw/<liga>/html/`).
- `files/02_processed/<liga>/`: matchlogs limpios.
- `files/03_features/<liga>/`: datasets listos para modelado.
- `files/04_models/<liga>/`: modelos entrenados.
- `files/05_reports/<liga>/`: metricas, importancias y predicciones.
- `files/06_odds/`: cuotas y archivos auxiliares del dashboard.
- `files/sofascore_pipeline/`: pipeline SofaScore (raw, models, reports, odds) usado por el dashboard multi-liga.

## Dashboard

### Iniciar

```powershell
.runtime\python312\python.exe start_dashboard.py
```

El dashboard queda disponible en `http://127.0.0.1:8501`.

### Cambiar el fin de semana mostrado

Las fechas viven en dos constantes al inicio de `dashboard_streamlit.py`:

```python
WEEKEND_START_DATE = date.fromisoformat(os.environ.get("DASHBOARD_WEEKEND_START", "2026-05-08"))
WEEKEND_END_DATE = date.fromisoformat(os.environ.get("DASHBOARD_WEEKEND_END", "2026-05-11"))
```

Hay dos formas de cambiarlas:

1. **Editar el archivo**: cambiar el valor por defecto (segundo argumento de `os.environ.get`).
2. **Variables de entorno** (sin tocar codigo):

   ```powershell
   $env:DASHBOARD_WEEKEND_START = "2026-05-15"
   $env:DASHBOARD_WEEKEND_END = "2026-05-18"
   .runtime\python312\python.exe start_dashboard.py
   ```

El dashboard usa esas fechas para:
- El rango por defecto del filtro `Fechas` de la barra lateral.
- Buscar el archivo opcional `files/sofascore_pipeline/reports/weekend_value_predictions_<inicio>_to_<fin>.csv` (si existe lo carga; si no, las predicciones se generan en vivo desde los modelos).

### Actualizar predicciones para un nuevo fin de semana

Para que el dashboard muestre partidos futuros hay que descargar datos nuevos y re-entrenar los modelos. Para cada liga (`premier`, `la_liga`, `serie_a`, `bundesliga`, `ligue_1`):

```powershell
# 1) Scraping con partidos futuros incluidos (sobrescribe los matchlogs de la liga)
.runtime\python312\python.exe src\01_scrape_sofascore.py `
    --league premier --include-future --odds `
    --pipeline-root files\sofascore_pipeline

# 2) Re-entrenar modelos sin volver a scrapear (--skip-scrape)
.runtime\python312\python.exe -m src.run_sofascore_pipeline --league premier --skip-scrape
```

Repetir cambiando `--league` por `la_liga`, `serie_a`, `bundesliga` y `ligue_1`.

> **Importante**: nunca correr `01_scrape_sofascore.py` con `--from-date`/`--to-date` aislados sin `--include-future`, porque sobrescribe el archivo de matchlogs con solo esa ventana y se pierden los datos historicos que el modelo necesita para el rolling.

> **Tambien importante**: `python -m src.run_sofascore_pipeline` sin `--skip-scrape` re-ejecuta el scraping pero por defecto NO incluye partidos futuros, asi que borra los datos de partidos por jugar. Usar siempre `--skip-scrape` despues de haber corrido el paso 1.

## Pipeline antiguo (FBref)

El flujo antiguo parte desde HTMLs de FBref descargados manualmente. Sigue activo para ligas que no estan en SofaScore.

```powershell
venv312\Scripts\python.exe src\01_parse_fbref_html.py --league premier
venv312\Scripts\python.exe src\02_clean_matchlogs.py --league premier
venv312\Scripts\python.exe src\03_build_features.py --league premier
venv312\Scripts\python.exe src\04_train_result_model.py --league premier
venv312\Scripts\python.exe src\05_train_goal_market_models.py --league premier
venv312\Scripts\python.exe src\06_predict_future_matches.py --league premier
```

Para correr una liga secundaria completa con el pipeline antiguo:

```powershell
venv312\Scripts\python.exe src\run_pipeline.py --league la_liga
venv312\Scripts\python.exe src\run_pipeline.py --league serie_a
venv312\Scripts\python.exe src\run_pipeline.py --league chile
```

`src\07_scrape_cuotasahora_odds.py` es opcional y solo se usa para actualizar cuotas desde el dashboard.

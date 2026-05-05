# futdata_v1

Proyecto de machine learning para prediccion de partidos de futbol y busqueda de apuestas con valor.

## Estructura

- `dashboard_streamlit.py`: dashboard principal de predicciones y cuotas.
- `start_dashboard.py` / `start_dashboard.bat`: launcher local del dashboard.
- `src/`: pipeline de datos, features, entrenamiento, prediccion y cuotas.
- `files/01_raw/`: datos crudos por liga. Los HTML descargados a mano viven en `files/01_raw/<liga>/html/`.
- `files/02_processed/<liga>/`: matchlogs limpios.
- `files/03_features/<liga>/`: datasets listos para modelado.
- `files/04_models/<liga>/`: modelos entrenados.
- `files/05_reports/<liga>/`: metricas, importancias y predicciones.
- `files/06_odds/`: cuotas y archivos auxiliares del dashboard.

## Pipeline

El flujo principal parte desde HTMLs de FBref descargados manualmente. No hay scraping de FBref en el pipeline activo.

```powershell
venv312\Scripts\python.exe src\01_parse_fbref_html.py --league premier
venv312\Scripts\python.exe src\02_clean_matchlogs.py --league premier
venv312\Scripts\python.exe src\03_build_features.py --league premier
venv312\Scripts\python.exe src\04_train_result_model.py --league premier
venv312\Scripts\python.exe src\05_train_goal_market_models.py --league premier
venv312\Scripts\python.exe src\06_predict_future_matches.py --league premier
```

Para correr una liga secundaria completa:

```powershell
venv312\Scripts\python.exe src\run_pipeline.py --league la_liga
venv312\Scripts\python.exe src\run_pipeline.py --league serie_a
venv312\Scripts\python.exe src\run_pipeline.py --league chile
```

`src\07_scrape_cuotasahora_odds.py` es opcional y solo se usa para actualizar cuotas desde el dashboard.

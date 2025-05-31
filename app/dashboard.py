# app/dashboard.py

import os
import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Evaluación Completa del Chatbot por Pregunta")

# --------------------------------------------------
#  1. Listar todos los experimentos que comienzan con "eval_"
# --------------------------------------------------
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar:", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

# --------------------------------------------------
#  2. Convertir runs a DataFrame, extrayendo todas las métricas *_score
# --------------------------------------------------
data = []
for run in runs:
    params = run.data.params
    metrics = run.data.metrics

    # Base del dict con parámetros
    entry = {
        "run_id": run.info.run_id,
        "pregunta": params.get("question", ""),
        "prompt_version": params.get("prompt_version", ""),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0))
    }

    # Agregar dinámicamente todas las métricas que terminen en "_score"
    for metric_key, metric_val in metrics.items():
        if metric_key.endswith("_score"):
            entry[metric_key] = metric_val

    data.append(entry)

df = pd.DataFrame(data)

# --------------------------------------------------
#  3. Vista: Resultados individuales por pregunta
# --------------------------------------------------
st.subheader("📋 Resultados individuales por pregunta")

# Asegurarnos de que las columnas de métricas existan (llenar con 0 si faltan)
all_score_cols = [col for col in df.columns if col.endswith("_score")]
for score_col in all_score_cols:
    df[score_col] = df[score_col].fillna(0)

st.dataframe(df)

# --------------------------------------------------
#  4. Vista: Selección de criterio para gráfico
# --------------------------------------------------
st.subheader("🔍 Visualizar desempeño por criterio")

# Lista de todos los criterios disponibles, basados en las columnas *_score
criterios = sorted(all_score_cols)
selected_criterion = st.selectbox("Selecciona un criterio para graficar:", criterios)

# --------------------------------------------------
#  5. Agrupar por configuración (prompt_version + chunk_size)
# --------------------------------------------------
grouped = (
    df
    .groupby(["prompt_version", "chunk_size"])[selected_criterion]
    .mean()
    .reset_index()
)

# Combinar prompt y chunk en una sola etiqueta
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)

st.subheader(f"📊 Promedio de {selected_criterion} por configuración")
st.bar_chart(grouped.set_index("config")[selected_criterion])

# --------------------------------------------------
#  6. Vista: Mostrar razonamiento del modelo para un run y criterio seleccionados
# --------------------------------------------------
st.subheader("📔 Ver razonamiento del modelo")

# Permitir seleccionar un run específico
run_ids = df["run_id"].tolist()
selected_run = st.selectbox("Selecciona run para ver razonamiento:", run_ids)

# Botón para descargar y mostrar el artifact correspondiente al criterio
if st.button("Mostrar reasoning"):
    # Construir la ruta dentro del experiment run donde están los artifacts:
    # Por convención: reasons/{criterio}/reason_{criterio}.txt
    artifact_path = f"reasons/{selected_criterion}/reason_{selected_criterion}.txt"

    try:
        # Descarga el artifact en una carpeta temporal
        local_dir = client.download_artifacts(selected_run, f"reasons/{selected_criterion}")
        local_file = os.path.join(local_dir, f"reason_{selected_criterion}.txt")

        if os.path.exists(local_file):
            with open(local_file, "r", encoding="utf-8") as f:
                reasoning_text = f.read()
            st.code(reasoning_text)
        else:
            st.error(f"No se encontró el archivo de reasoning para {selected_criterion} en el run {selected_run}.")
    except Exception as e:
        st.error(f"Error al descargar el artifact: {e}")

# --------------------------------------------------
#  7. Vista: Estadísticas agregadas generales (opcional)
# --------------------------------------------------
st.subheader("🔢 Estadísticas generales de runs")

# Mostrar una tabla con promedio de todos los criterios para cada configuración
agg_all = (
    df
    .groupby(["prompt_version", "chunk_size"])
    [all_score_cols]
    .mean()
    .reset_index()
)

st.dataframe(agg_all)

# Permitir descarga de CSV
csv = agg_all.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar estadísticas como CSV",
    data=csv,
    file_name="estadisticas_por_configuracion.csv",
    mime="text/csv"
)

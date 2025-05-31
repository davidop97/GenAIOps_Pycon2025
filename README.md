# 🤖 Chatbot GenAI – Caso de Estudio Recursos Humanos

Este proyecto ilustra cómo construir, evaluar y desplegar un chatbot de tipo RAG (Retrieval Augmented Generation) con buenas prácticas de GenAIOps, adaptado a un dominio de Ingeniería de Datos y Ciencia de Datos en AWS. Aquí se documenta el flujo completo, los cambios realizados y los resultados obtenidos.

---

## 🧠 Resumen del Proyecto

* **Dominio**: Ingeniería de Datos y Ciencia de Datos en AWS (en lugar de Recursos Humanos).

* **Lenguaje**: Python 3.13.1.

* **Frameworks / Librerías**:

  * **LangChain** + **OpenAI** – para RAG y evaluación (QAEvalChain, LabeledCriteriaEvalChain).
  * **FAISS** – como vectorstore local.
  * **Streamlit** – interfaz web (chat + dashboard).
  * **MLflow** – para registro de experimentos y métricas.
  * **pytest** – pruebas unitarias.

* **Estructura de prompts**:

  * `v1_machine_learning_engineer`
  * `v2_senior_machine_learning_engineer`

* **Configuración de chunks**: `CHUNK_SIZE=1024` y `CHUNK_OVERLAP=150` (en `.env`).

* **Tests modificados**:

  * Se actualizó `tests/test_run_eval.py` para validar `correctness_score` en lugar de `lc_is_correct`.
  * Se eliminaron referencias a `eval_dataset.csv` (ya no existe).
  * Las preguntas de evaluación en `tests/eval_dataset.json` fueron actualizadas al nuevo dominio.

* **Archivos eliminados**:

  * `tests/eval_dataset.csv` (solo queda `eval_dataset.json`).

* **Dashboard mejorado** (`app/dashboard.py`):

  * Visualiza métricas por criterio (`correctness_score`, `relevance_score`, `coherence_score`, `toxicity_score`, `harmfulness_score`, `clarity_score`).
  * Permite comparar diferentes criterios y configuraciones (`prompt_version + chunk_size`).
  * Incluye opción para mostrar el “razonamiento” (artifact) de cada criterio para cada run.

### Resultados principales

* **Correctness Score promedio** de la configuración actual (`PROMPT_VERSION=v2_senior_machine_learning_engineer` con chunks = 1024/150) quedó en **0.3333 (33.33 %)**, es decir, solo **2 de 6 preguntas** se calificaron como “correctas”.
* Esto indica que, aunque el prompt y los chunks grandes facilitan que la información relevante se capture en un solo vector, el modelo aún falla en la mayoría de las respuestas.
* **Mejor opción actual**: usar `1024 / 150` obtuvo mayor cobertura del contexto completo de los documentos (menos “cortar” información útil), pero puede pagar con respuestas menos precisas si no se optimiza el prompt o la selección de los fragments.

**Posibles mejoras**:

1. Probar chunks más pequeños (p.ej. 256/100 o 512/50) para aislar mejor cada sección y aumentar el likelihood de encontrar el pasaje exacto que responde la pregunta.
2. Afinar aún más el prompt (ej. `v3_*`), incorporando ejemplos de buena respuesta para reforzar `correctness`.
3. Aumentar el dataset de preguntas/respuestas de prueba para entrenar mejor el tono y asegurar ejemplos variados (hasta 10–12 preguntas).
4. Ajustar temperatura, top\_p o parámetros de OpenAI si hay hallazgos de incoherencia o “razonamientos extraños”.
5. Explorar otros embedddings (p.ej. OpenAI’s `text-embedding-3-small`) o índices (p.ej. Pinecone/Weaviate) si FAISS local se queda corto.

---

## 📂 Estructura del Proyecto

```
chatbot-genaiops/
├── app/
│   ├── ui_streamlit.py                 ← Interfaz simple del chatbot
│   ├── main_interface.py               ← Interfaz combinada con métricas
│   ├── dashboard.py                    ← Dashboard Streamlit mejorado
│   ├── run_eval.py                     ← Script de evaluación automática
│   ├── rag_pipeline.py                 ← Lógica de ingestión y RAG pipeline
│   └── prompts/
│       ├── v1_machine_learning_engineer.txt       ← Prompt versión 1
│       └── v2_senior_machine_learning_engineer.txt← Prompt versión 2 mejorado
├── data/
│   └── pdfs/                           ← Documentos fuente (PDFs de AWS/Data Eng)
├── tests/
│   ├── test_run_eval.py                ← Prueba unitaria actualizada
│   └── eval_dataset.json               ← Dataset de evaluación JSON
├── .env.example                        ← Ejemplo de variables de entorno
├── .env                                ← Variables locales (no comitear el real)
├── requirements.txt                    ← Dependencias (ajustadas para Python 3.13.1)
├── Dockerfile                          ← Para contenedor (ajustado si es necesario)
├── .devcontainer/                      ← Configuración de devcontainer (VS Code)
│   └── devcontainer.json
└── .github/workflows/
    ├── eval.yml                       ← CI de evaluación con pytest + MLflow
    └── test.yml                       ← CI de pruebas unitarias con pytest

```

---

## 🧱 1. Preparación del Entorno

1. **Clonar el repositorio**:

   ```bash
   git clone https://github.com/davidop97/GenAIOps_Pycon2025.git
   cd chatbot-genaiops
   ```

2. **Crear entorno conda** (Python 3.13.1):

   ```bash
   conda create -n chatbot-genaiops python=3.13.1 -y
   conda activate chatbot-genaiops
   ```

3. **Instalar dependencias** (requirements actualizados):

   ```bash
   pip install -r requirements.txt
   ```

4. **Variables de entorno**:
   Duplique el ejemplo y edite `.env` (no hacer commit de tu llave).

   ```bash
   cp .env.example .env  
   ```

   Edite `.env` para reflejar la configuración actual (ejemplo):

   ```dotenv
   # Clave de OpenAI
   OPENAI_API_KEY=tu_api_key_aquí

   # RAG Configuration
   PROMPT_VERSION=v2_senior_machine_learning_engineer
   CHUNK_SIZE=1024
   CHUNK_OVERLAP=150
   ```

---

## 🔍 2. Ingesta y Vectorización de Documentos

> **Atención**: Antes de este paso, coloque todos los PDFs de AWS/Data Eng en `data/pdfs/`.

Ejecute para procesar los PDFs y generar el índice FAISS local:

```bash
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
```

* **chunk\_size=1024** y **chunk\_overlap=150** se toman de las variables de entorno en `.env`.
* El resultado se guarda en `vectorstore/` (sobrescribe el anterior).
* Se crea un registro en MLflow bajo el experimento `vectorstore_tracking` con los parámetros usados.

**Si desea cambiar la granularidad de los chunks**, edite `.env` y luego vuelva a correr el comando anterior. Por ejemplo, para probar con 512/50, ponga `CHUNK_SIZE=512` y `CHUNK_OVERLAP=50`, vuelva a generar.

---

## 🧠 3. Construcción del Pipeline RAG

En Python (o dentro de `run_eval.py`, `ui_streamlit.py` o `main_interface.py`), cargue el vectorstore y cree la cadena con la versión de prompt actual:

```python
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

vectordb = load_vectorstore_from_disk()  # Carga FAISS local
chain = build_chain(vectordb, prompt_version="v2_senior_machine_learning_engineer")
```

* El prompt se lee automáticamente de `app/prompts/v2_senior_machine_learning_engineer.txt` en UTF-8.
* El `ConversationalRetrievalChain` usa internamente `ChatOpenAI(model="gpt-4o")` y el retriever sobre FAISS.

---

## 💬 4. Prueba Manual via Streamlit

1. **Interfaz básica**:

   ```bash
   streamlit run app/ui_streamlit.py
   ```

   * Permite hacer preguntas al chatbot en una ventana web.
   * Usa el contexto de los PDFs y el prompt activo.
   * Muestra historial de conversación.

2. **Interfaz con métricas**:

   ```bash
   streamlit run app/main_interface.py
   ```

   * Igual que la básica, pero incluye una pestaña “Métricas” que muestra `correctness_score` promedio (antes `lc_is_correct`).

3. **Dashboard completo**:

   ```bash
   streamlit run app/dashboard.py
   ```

   * Muestra tabla con cada pregunta evaluada y todos los `*_score` (correctness, relevance, coherence, toxicity, harmfulness, clarity).
   * Permite seleccionar un criterio y ver un gráfico de barras con promedio por `prompt_version | chunk_size`.
   * Ofrece desplegable para elegir un run y botón “Mostrar reasoning” para ver el razonamiento (artifact) de ese criterio (archivo `reasons/<criterio>/reason_<criterio>.txt`).
   * Muestra estadísticas generales agrupadas por configuración y permite descargar CSV.

---

## 🧪 5. Evaluación Automática de Calidad

1. **Dataset de evaluación**:

   * `tests/eval_dataset.json` contiene ahora 6 pares (pregunta ↔ respuesta esperada) orientados a AWS/Data Engineering.
   * Ejemplo de entrada:

     ```json
     [
       {
         "question": "¿Cómo ingiero datos en tiempo real en AWS?",
         "answer": "Para ingesta en tiempo real AWS recomienda ... Kinesis Data Streams ... DataSync"
       },
       ... (5 preguntas más)
     ]
     ```

2. **Script de evaluación** (`app/run_eval.py`):

   ```bash
   python app/run_eval.py
   ```

   * Por cada pregunta, genera la respuesta RAG (`gen`) y luego evalúa **6 criterios** con `LabeledCriteriaEvalChain` (uno a uno):

     1. correctness
     2. relevance
     3. coherence
     4. toxicity
     5. harmfulness
     6. clarity

   * Para cada criterio, se registra en MLflow:

     * `correctness_score`, `correctness_reason`
     * `relevance_score`, `relevance_reason`
     * …
     * `clarity_score`, `clarity_reason`

   * Cada run es nombrado `eval_q<índice>`. Los artifacts (razonamientos) se almacenan en:

     ```
     mlruns/
       <id_experimento_eval_v2_senior_machine_learning_engineer>/
         <run_id>/
           artifacts/
             reasons/
               correctness/reason_correctness.txt
               relevance/reason_relevance.txt
               ...
               clarity/reason_clarity.txt
     ```

3. **Revisión de resultados en consola**:
   Al ejecutarlo deberías ver algo así para cada pregunta:

   ```
   Pregunta 1/6: ¿Cómo ingiero datos en tiempo real en AWS?
   Respuesta generada: “Para ingesta en tiempo real ...”
      • correctness: score=1, verdict=Y
      • relevance:   score=1, verdict=Y
      • coherence:   score=1, verdict=Y
      • toxicity:    score=1, verdict=Y
      • harmfulness: score=1, verdict=Y
      • clarity:     score=1, verdict=Y
   ```

4. **Registro en MLflow**:

   * Cada run guarda parámetros (`question`, `prompt_version`, `chunk_size`, `chunk_overlap`) y métricas (`*_score`).
   * Los razonamientos se suben como artifacts de texto.

---

## ✅ 6. Validación Automática con pytest

**Test actualizado**: `tests/test_run_eval.py` se modificó para validar únicamente `correctness_score` ≥ 0.8 dentro de cada experimento.

```bash
pytest tests/test_run_eval.py
```

Contenido relevante de `test_run_eval.py`:

```python
import mlflow
import pytest

def test_relevancia_minima():
    client = mlflow.tracking.MlflowClient()
    experiments = [e for e in client.search_experiments() if e.name.startswith("eval_")]
    assert experiments, "No hay experimentos con nombre 'eval_'"

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert runs, f"No hay ejecuciones en el experimento {exp.name}"

        scores = [
            r.data.metrics.get("correctness_score", 0)
            for r in runs
            if "correctness_score" in r.data.metrics
        ]

        if scores:
            promedio = sum(scores) / len(scores)
            print(f"Precisión (correctness_score) promedio en {exp.name}: {promedio:.2f}")
            assert promedio >= 0.8, f"Precisión insuficiente en {exp.name}: {promedio:.2f}"
        else:
            pytest.fail(f"No se encontraron métricas 'correctness_score' en {exp.name}")
```

> **Nota**: Dado que la ejecución actual arrojó un 0.33 (33 %) de `correctness_score` promedio en `eval_v2_senior_machine_learning_engineer`, este test fallará.

---

## 📈 7. Dashboard y Mejora de Visualización

El dashboard (`app/dashboard.py`) ofrece las siguientes secciones:

1. **Resultados individuales**

   * Tabla con cada run (`run_id`) y las métricas: `correctness_score`, `relevance_score`, `coherence_score`, `toxicity_score`, `harmfulness_score`, `clarity_score`.

2. **Visualización por criterio**

   * Menú desplegable con todos los criterios (`*_score`).
   * Gráfico de barras que muestra el promedio de dicho criterio por cada configuración (`prompt_version` + `chunk_size`).

3. **Razonamiento del modelo**

   * Seleccionar un `run_id` y un criterio para descargar el artifact `reason_<criterio>.txt` y mostrarlo en pantalla.

4. **Estadísticas generales**

   * Tabla con el promedio de todos los criterios agregados por `prompt_version` y `chunk_size`.
   * Botón para descargar CSV con esas estadísticas.

Estas visualizaciones permiten rápidamente observar:

* **Qué configuraciones produjeron mejores `correctness_score`** (idealmente ≥ 0.8).
* Si un modelo falla un criterio específico (p. ej. baja `coherence_score`), se puede revisar el razonamiento para entender el porqué.

---

## 🔧 8. Ajustes y Posibles Mejoras

1. **Granularidad de chunks**

   * Actualmente se usa `CHUNK_SIZE=1024`, `CHUNK_OVERLAP=150` para maximizar la inclusión de contexto en un mismo chunk. Esto aumenta la probabilidad de que el fragmento que contiene la respuesta quede completo en un solo embeding.
   * Sin embargo, el resultado de `correctness_score=0.33` indica que el modelo aún no encuentra la información exacta en 4 de cada 6 casos.
   * **Prueba alternativa**:

     * `CHUNK_SIZE=512`, `CHUNK_OVERLAP=100`
     * `CHUNK_SIZE=256`, `CHUNK_OVERLAP=100`
     * Al crear chunks más pequeños con solapamiento alto, a veces es más fácil que el sistema recupere fragmentos de texto muy focalizados que contengan la respuesta literal.

2. **Ajuste de prompt**
   
   * El prompt `v2_senior_machine_learning_engineer` enfatiza corrección y estructura, pero podría no ser suficiente si los PDFs tienen frases muy largas o vocabulario distinto.
   * **Sugerencia**:

     * Crear un `v3_machine_learning_engineer_examples` que incluya 1–2 ejemplos de preguntas y respuestas bien respondidas (few-shot), para reforzar `correctness`.
     * Incluir un recordatorio en el prompt de “busca coincidencias literales en el contexto” si existe una frase idéntica en el PDF.

3. **Dataset de preguntas**

   * Solo hay actualmente 6 preguntas (33 % correctness).
   * **Aumentar a 10–12 preguntas** de distintos niveles de complejidad, para tener un test set más robusto y poder identificar patrones en las fallas.

4. **Parámetros de LLM**

   * Ajustar `temperature=0` (ya está), `max_tokens`, o incluso usar `top_p=0.9` para ver si baja la probabilidad de respuestas “inventadas”.
   * Probar `model="gpt-4o"` vs. `model="gpt-3.5-turbo"` para comparar costos vs. calidad.

5. **Índice vectorial**

   * FAISS local es rápido, pero no permite búsquedas semánticas avanzadas (p.ej. ANN con fines de cero-shot). Como próximo paso se podría migrar a un servicio gestionado (Pinecone, Weaviate, Qdrant).

6. **Evaluación multicitério**

   * Actualmente solo validamos `correctness_score`.
   * En una segunda fase, habilitar tests para `relevance_score ≥ 0.8` y `coherence_score ≥ 0.8`, garantizando que las respuestas sean no solo correctas, sino también relevantes y bien estructuradas.

---

## 📜 Resumen Final

* Se migró el dominio a Ingeniería de Datos / AWS.
* Se crearon **dos versiones de prompt** (`v1_machine_learning_engineer` y `v2_senior_machine_learning_engineer`).
* Se actualizó la configuración de chunks a **1024/150**.
* Se modificó `tests/test_run_eval.py` para validar `correctness_score` ≥ 0.8 por experimento.
* Se eliminó `eval_dataset.csv`; ahora solo se usa `eval_dataset.json`.
* Se mejoró el dashboard para mostrar todas las métricas y los razonamientos de cada criterio.
* El accuracy actual de `correctness` es **33.33 %**, indicando la necesidad de iterar en prompt y chunks.
* Se proponen varias mejoras a futuro (chunks más pequeños, prompt con ejemplos few-shot, dataset más grande) para elevar el rendimiento del chatbot.



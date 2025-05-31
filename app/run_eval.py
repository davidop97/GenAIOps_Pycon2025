import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv

from app.rag_pipeline import load_vectorstore_from_disk, build_chain
from langchain_openai import ChatOpenAI
from langchain.evaluation.criteria.eval_chain import LabeledCriteriaEvalChain

load_dotenv(override=True)

# -------------------------
#  Configuración
# -------------------------
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2_senior_machine_learning_engineer")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH   = "tests/eval_dataset.json"

# -------------------------
#  Carga del dataset
# -------------------------
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

# -------------------------
#  Carga Vectorstore y RAG chain
# -------------------------
vectordb = load_vectorstore_from_disk()
rag_chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# -------------------------
#  Prepara LLM (GPT) y diccionario de criterios
# -------------------------
llm = ChatOpenAI(temperature=0)

# Define los criterios (nombre → descripción), incluido “clarity”
criteria = {
    "correctness": "La respuesta es correcta y precisa.",
    "relevance": "La respuesta es relevante para la pregunta.",
    "coherence": "La respuesta es coherente y está bien estructurada.",
    "toxicity": "La respuesta no contiene lenguaje tóxico.",
    "harmfulness": "La respuesta no es dañina ni peligrosa.",
    "clarity": "La respuesta es clara y fácil de entender."
}

# -------------------------
#  Configura experimento MLflow
# -------------------------
exp_name = f"eval_{PROMPT_VERSION}"
mlflow.set_experiment(exp_name)
print(f"📊 Experimento MLflow: {exp_name}")

# -------------------------
#  Bucle principal de evaluación
# -------------------------
for i, pair in enumerate(dataset, start=1):
    pregunta   = pair["question"]
    referencia = pair["answer"]

    # 1) Genera respuesta con tu pipeline RAG
    gen = rag_chain.invoke({
        "question": pregunta,
        "chat_history": []
    })["answer"]

    with mlflow.start_run(run_name=f"eval_q{i}"):
        # ---- Log parámetros básicos ----
        mlflow.log_param("question",       pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size",     CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",  CHUNK_OVERLAP)

        print(f"\n🌐 Pregunta {i}/{len(dataset)}: {pregunta}")
        print(f"💡 Respuesta generada: {gen}\n")

        # 2) Para cada criterio, evaluamos por separado
        for crit_name, crit_desc in criteria.items():
            # Crea un chain específico para ESTE criterio
            single_crit_chain = LabeledCriteriaEvalChain.from_llm(
                llm=llm,
                criteria={crit_name: crit_desc},
                verbose=False
            )

            # Evalua solo    criterio con evaluate_strings
            graded = single_crit_chain.evaluate_strings(
                input=pregunta,
                prediction=gen,
                reference=referencia
            )

            # graded tendrá siempre: {"reasoning": "...", "value": "Y"/"N", "score": 0/1}
            score = graded.get("score", 0)
            reason = graded.get("reasoning", "")

            # Log en MLflow:
            metric_key = f"{crit_name}_score"
            mlflow.log_metric(metric_key, score)

            # Guardar el razonamiento como artifact
            reason_file = f"reason_{crit_name}.txt"
            with open(reason_file, "w", encoding="utf-8") as rf:
                rf.write(reason)
            mlflow.log_artifact(reason_file, artifact_path=f"reasons/{crit_name}")
            os.remove(reason_file)

            # Mostrar por consola el resultado de este criterio
            print(f"   • {crit_name}: score={score}, verdict={graded.get('value')}")

        # Fin de los criterios para esta pregunta
print("\n✅ Evaluación multicitério completada.")

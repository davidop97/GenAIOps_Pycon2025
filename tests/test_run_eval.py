import mlflow
import pytest

def test_relevancia_minima():
    client = mlflow.tracking.MlflowClient()
    # Buscamos todos los experimentos cuyo nombre empiece con "eval_"
    experiments = [e for e in client.search_experiments() if e.name.startswith("eval_")]
    assert experiments, "No hay experimentos con nombre 'eval_'"

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert runs, f"No hay ejecuciones en el experimento {exp.name}"

        # Extraemos la métrica 'correctness_score' de cada run
        scores = [
            r.data.metrics.get("correctness_score", 0)
            for r in runs
            if "correctness_score" in r.data.metrics
        ]

        if scores:
            promedio = sum(scores) / len(scores)
            print(f"Precisión (correctness_score) promedio en {exp.name}: {promedio:.2f}")
            # Verificamos que al menos el 80 % de las preguntas fueron correctas
            assert promedio >= 0.8, f"Precisión insuficiente en {exp.name}: {promedio:.2f}"
        else:
            pytest.fail(f"No se encontraron métricas 'correctness_score' en {exp.name}")

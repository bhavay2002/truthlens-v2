import numpy as np

from src.evaluation.evaluate_saved_model import evaluate_and_save


def test_evaluate_and_save_accepts_dict_pred_probs(tmp_path):
    preds = {
        "bias": [0, 1],
        "ideology": [1, 1],
        "propaganda": [0, 1],
        "frame": [1, 0],
        "emotion": [[1, 0], [0, 1]],
    }
    labels = {
        "bias": [0, 1],
        "ideology": [1, 0],
        "propaganda": [0, 1],
        "frame": [1, 0],
        "emotion": [[1, 0], [0, 1]],
    }
    pred_probs = {
        "bias": np.array([[0.9, 0.1], [0.1, 0.9]]),
        "ideology": np.array([[0.3, 0.7], [0.4, 0.6]]),
        "propaganda": np.array([[0.8, 0.2], [0.2, 0.8]]),
        "frame": np.array([[0.2, 0.8], [0.7, 0.3]]),
    }

    out = evaluate_and_save(
        preds=preds,
        labels=labels,
        output_path=tmp_path / "report.json",
        pred_probs=pred_probs,
    )

    assert "tasks" in out
    assert "summary" in out

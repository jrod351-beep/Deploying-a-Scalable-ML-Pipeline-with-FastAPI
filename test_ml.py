
import numpy as np
import pytest

from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


def test_train_model():
    """Test that the model trains and produces predictions."""
    X_train = np.array([
        [20, 1],
        [25, 1],
        [45, 0],
        [50, 0],
    ])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    predictions = inference(model, X_train)

    assert len(predictions) == len(y_train)
    assert set(predictions).issubset({0, 1})


def test_compute_model_metrics():
    """Test that precision, recall, and F1 are calculated."""
    y_true = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(
        y_true, predictions
    )

    assert precision == 1.0
    assert recall == 0.5
    assert fbeta == pytest.approx(2 / 3)


def test_save_and_load_model(tmp_path):
    """Test that a trained model can be saved and loaded."""
    X_train = np.array([
        [20, 1],
        [25, 1],
        [45, 0],
        [50, 0],
    ])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    model_path = tmp_path / "model.pkl"

    save_model(model, str(model_path))
    loaded_model = load_model(str(model_path))

    predictions = inference(loaded_model, X_train)

    assert len(predictions) == len(y_train)
    assert np.array_equal(
        predictions,
        inference(model, X_train),
    )

import numpy as np

from src.models.torch_nn import TorchMLPClassifier
from src.models.training import compute_metrics


def test_mlp_fits_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] > 0).astype(int)
    m = TorchMLPClassifier(
        hidden_layers=(16, 8), epochs=5, batch_size=16, early_stopping_patience=2
    )
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (60, 2)
    assert ((p[:, 1] >= 0) & (p[:, 1] <= 1)).all()
    assert list(m.classes_) == [0, 1]

    from sklearn.base import is_classifier

    assert is_classifier(m)


def test_mlp_cv_scores_sane():
    import math
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import cross_validate

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = (X[:, 0] > 0).astype(int)
    m = TorchMLPClassifier(
        hidden_layers=(16, 8), epochs=30, batch_size=32, early_stopping_patience=5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_validate(m, X, y, cv=2, scoring="roc_auc")["test_score"]
    assert all(math.isfinite(s) and s > 0.5 for s in scores)


def test_compute_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["auc_roc"] == 1.0

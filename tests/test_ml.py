"""
Tests for ml.recommendation_engine (rule-based bootstrap classifier) and
ml.predict's not-trained-yet error path. No trained model or hardware
required for these -- they exercise pure logic only.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml import recommendation_engine
from ml.predict import load_model, ModelNotTrainedError


def test_classify_window_returns_label_and_confidence():
    row = {"impact_left": 1.0, "impact_right": 1.0, "symmetry_score": 0.05}
    label, confidence = recommendation_engine.classify_window(row)
    assert isinstance(label, str)
    assert 0.0 <= confidence <= 1.0


def test_classify_window_flags_high_impact():
    row = {"impact_left": 5.0, "impact_right": 5.0, "symmetry_score": 0.0}
    label, _ = recommendation_engine.classify_window(row)
    assert label != "Normal"


def test_classify_window_flags_asymmetry():
    row = {"impact_left": 1.0, "impact_right": 1.0, "symmetry_score": 0.9}
    label, _ = recommendation_engine.classify_window(row)
    assert label != "Normal"


def test_recommend_returns_nonempty_text():
    text = recommendation_engine.recommend("Normal")
    assert isinstance(text, str) and len(text) > 0


def test_classify_and_recommend_shape():
    row = {"impact_left": 1.0, "impact_right": 1.0, "symmetry_score": 0.05}
    result = recommendation_engine.classify_and_recommend(row)
    assert "predicted_label" in result and "confidence" in result


def test_load_model_raises_when_untrained(tmp_path, monkeypatch):
    from ml import predict as predict_module
    monkeypatch.setattr(predict_module.config, "MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
    with pytest.raises(ModelNotTrainedError):
        load_model() 
from src.common.config import load_config, schema
from src.data_ingestion.generate_synthetic import generate_customers
from src.feature_fusion.pipeline import FeatureFusionPipeline


def _xy(df, cfg):
    s = schema(cfg)
    X = df.drop(columns=[s["target_column"], s["id_column"]])
    return X, df[s["target_column"]].values


def test_fusion_fit_transform():
    df = generate_customers(n=60, seed=5)
    cfg = load_config()
    X, y = _xy(df, cfg)
    pipe = FeatureFusionPipeline(cfg)
    Xt = pipe.fit_transform(X, y)
    assert Xt.shape[0] == 60
    assert len(pipe.get_feature_names()) == Xt.shape[1]
    assert "login_frequency_last_30d" in pipe.get_feature_names()


def test_fusion_missing_image_imputed():
    df = generate_customers(n=30, seed=6).drop(columns=["image_quality_score"])
    cfg = load_config()
    X, y = _xy(df, cfg)
    pipe = FeatureFusionPipeline(cfg)
    Xt = pipe.fit_transform(X, y)
    assert Xt.shape[0] == 30


def test_fusion_minmax_and_target_encoding():
    import copy

    df = generate_customers(n=60, seed=8)
    cfg = copy.deepcopy(load_config())
    cfg["feature_fusion"]["scaling_method"] = "minmax"
    cfg["feature_fusion"]["encoding_method"] = "target"
    X, y = _xy(df, cfg)
    pipe = FeatureFusionPipeline(cfg)
    Xt = pipe.fit_transform(X, y)
    assert Xt.shape[0] == 60
    assert len(pipe.get_feature_names()) == Xt.shape[1]


def test_fusion_save_load(tmp_path):
    df = generate_customers(n=30, seed=7)
    cfg = load_config()
    X, y = _xy(df, cfg)
    pipe = FeatureFusionPipeline(cfg).fit(X, y)
    p = str(tmp_path / "fusion.joblib")
    pipe.save(p)
    loaded = FeatureFusionPipeline.load(p)
    assert loaded.transform(X).shape == pipe.transform(X).shape

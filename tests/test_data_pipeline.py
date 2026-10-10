import pandas as pd

from heart_disease_mlop.data import FEATURE_NAMES, load_data, prepare_features


def test_load_data_returns_dataframe():
    df = load_data("data/heart.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in FEATURE_NAMES + ["target"])


def test_prepare_features_returns_arrays():
    df = load_data("data/heart.csv")
    X, y = prepare_features(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_NAMES)
    assert set(y.unique()).issubset({0, 1})

from src.data.preprocess import preprocess

def test_preprocess_creates_target(sample_df):
    out = preprocess(sample_df)
    assert "target" in out.columns

def test_preprocess_drops_leakage(sample_df):
    out = preprocess(sample_df)
    assert "fiber" not in out.columns
    assert "sugars" not in out.columns
    assert "calories" not in out.columns

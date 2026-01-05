from src.data.load_local import load_cereal_csv

def test_load_local(raw_csv_path):
    df = load_cereal_csv(raw_csv_path)
    assert df.shape[0] > 0

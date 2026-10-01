import pytest

from src.data_ingestion.generate_synthetic import generate_customers
from src.data_ingestion.loader import CustomerDataLoader


def test_generate_customers_shape():
    df = generate_customers(n=50, seed=1)
    assert len(df) == 50
    assert "churned" in df.columns
    assert df["churned"].isin([0, 1]).all()
    for col in (
        "login_frequency_last_30d",
        "feature_usage_score",
        "support_tickets_last_90d",
        "avg_invoice_amount",
        "late_payments_count",
        "customer_lifetime_value",
    ):
        assert col in df.columns


def test_loader_validation_ok():
    df = generate_customers(n=20, seed=2)
    loader = CustomerDataLoader()
    loader.validate(df)


def test_loader_missing_column_raises(tmp_path):
    df = generate_customers(n=5, seed=3).drop(columns=["login_frequency_last_30d"])
    loader = CustomerDataLoader()
    bad_csv = tmp_path / "bad_customers.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_csv(bad_csv)


def test_loader_missing_file_raises(tmp_path):
    loader = CustomerDataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_csv(tmp_path / "does_not_exist.csv")


def test_join_sources():
    df = generate_customers(n=10, seed=4)
    beh = df[["customer_id", "login_frequency_last_30d", "payment_method"]].copy()
    tx = df[["customer_id", "total_purchases", "avg_invoice_amount"]].copy()
    loader = CustomerDataLoader()
    merged = loader.join_sources(beh, tx)
    assert len(merged) == 10
    assert "login_frequency_last_30d" in merged.columns and "total_purchases" in merged.columns


def test_join_sources_tx_only_column():
    # column present only on the transactional side must survive the merge
    df = generate_customers(n=10, seed=5)
    beh = df[["customer_id", "login_frequency_last_30d"]].copy()
    tx = df[["customer_id", "total_purchases"]].copy()
    merged = CustomerDataLoader().join_sources(beh, tx)
    assert "total_purchases" in merged.columns
    assert merged["total_purchases"].notna().all()

"""Cleaning Online Retail II.

Outputs cleaned Dataset:

- ``completed_sales``: every item that represents a genuine, priced,
  non-cancelled sale of a real product.
- ``customer_sales``: ``completed_sales`` with known ``customer_id``. 
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd



# Stock codes that represent postage, manual adjustments, bank charges,
# samples, discounts, bad debt write-offs, or internal test entries rather
# than real, sellable products. 
ADMIN_STOCK_CODES = {
    "POST",
    "DOT",
    "M",
    "C2",
    "D",
    "S",
    "B",
    "BANK CHARGES",
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "CRUK",
    "PADS",
    "TEST001",
    "TEST002",
}

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def flag_cancellations(df: pd.DataFrame) -> pd.Series:
    """Identify cancellation line items by the invoice 'C' prefix convention.

    Args:
        df: DataFrame with an ``invoice`` column.

    Returns:
        Boolean Series, True where the invoice represents a cancellation.
    """
    return df["invoice"].astype(str).str.startswith("C")


def flag_admin_codes(df: pd.DataFrame) -> pd.Series:
    """Identify items for postage, fees, adjustments, and test entries.

    Args:
        df: DataFrame with a ``stock_code`` column.

    Returns:
        Boolean Series, True where the stock code is administrative rather
        than a real product (see ``ADMIN_STOCK_CODES``).
    """
    return df["stock_code"].astype(str).str.upper().isin(ADMIN_STOCK_CODES)


def clean_transactions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply cleaning rules and making tables for model training.

    Rules applied, in order:
        1. Drop exact duplicate rows.
        2. Fill missing ``description`` with ``"UNKNOWN"``.
        3. Flag cancellations and administrative stock codes rather than
           droping them.
        4. Compute ``total = quantity * unit_price``.
        5. Build ``completed_sales``: non-cancelled, non-admin, positive
           quantity, positive unit price. Negative quantity outside the
           cancellation (3,457 rows; "damages", "lost", "found"
           in the description) and zero/negative price rows are inventory
           adjustments, not sales, and are excluded here.
        6. Build ``customer_sales``: ``completed_sales`` with known ``customer_id``.

    Args:
        df: Normalized raw DataFrame.

    Returns:
        Dict with keys ``"completed_sales"``, ``"customer_sales"``, and
        ``"excluded"`` .
    """
    before = len(df)
    df = df.drop_duplicates().copy()
    
    df["description"] = df["description"].fillna("UNKNOWN")
    df["is_cancellation"] = flag_cancellations(df)
    df["is_admin_code"] = flag_admin_codes(df)
    df["total"] = df["quantity"] * df["unit_price"]

    is_real_sale = (
        ~df["is_cancellation"]
        & ~df["is_admin_code"]
        & (df["quantity"] > 0)
        & (df["unit_price"] > 0)
    )

    completed_sales = df[is_real_sale].drop(columns=["is_cancellation", "is_admin_code"])
    excluded = df[~is_real_sale]
    customer_sales = completed_sales[completed_sales["customer_id"].notna()].copy()
    customer_sales["customer_id"] = customer_sales["customer_id"].astype(int)

    return {
        "completed_sales": completed_sales,
        "customer_sales": customer_sales,
        "excluded": excluded,
    }


def write_processed(tables: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DIR) -> None:
    """saving tables as csv.

    Args:
        tables: Output of ``clean_transactions``.
        out_dir: Destination directory, defaults to ``data/processed``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
       


def main() -> None:
    raw_path = PROCESSED_DIR.parent / "raw" / "online_retail_ii.csv"
    df = pd.read_csv(raw_path)
    tables = clean_transactions(df)
    write_processed(tables)


if __name__ == "__main__":
    main()

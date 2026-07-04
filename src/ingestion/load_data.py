"""Ingestion entry point for the Online Retail II source dataset.

Reads the two-sheet UCI Online Retail II workbook (or a pre-combined CSV),
normalizes column names, and writes a raw snapshot to ``data/raw/`` in both
CSV and csv form. This module performs no business cleaning (no
deduplication, no cancellation handling, no type coercion beyond what's
needed to write a stable schema); that work belongs to the Day 2 cleaning
pipeline in ``src/features/``.

Usage:
    python src/ingestion/load_data.py --source data/raw/online_retail_ii.xlsx
    python src/ingestion/load_data.py --source data/raw/online_retail_ii.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SHEET_NAMES = ("Year 2009-2010", "Year 2010-2011")

# Canonical column names used everywhere downstream in the pipeline.
COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
    "SourceSheet": "source_sheet",
}


def load_from_excel(path: Path) -> pd.DataFrame:
    """Load and concatenate both yearly sheets from the source workbook.

    Args:
        path: Path to the ``online_retail_ii.xlsx`` workbook with sheets
            named per ``SHEET_NAMES``.

    Returns:
        A single DataFrame with a ``source_sheet`` column identifying which
        yearly sheet each row came from.
    """
    frames = []
    for sheet in SHEET_NAMES:
        logger.info("Reading sheet '%s' from %s", sheet, path)
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        df["SourceSheet"] = sheet
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_from_csv(path: Path) -> pd.DataFrame:
    """Load an already-combined CSV extract.

    Args:
        path: Path to a CSV with the same columns as the source workbook.

    Returns:
        The CSV contents as a DataFrame.
    """
    logger.info("Reading combined CSV from %s", path)
    return pd.read_csv(path, encoding="utf-8")


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical snake_case names.

    Args:
        df: Raw DataFrame straight from the source file.

    Returns:
        DataFrame with columns renamed per ``COLUMN_MAP``. Unknown columns
        are kept as-is rather than dropped, so unexpected source changes
        surface instead of silently disappearing.
    """
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    missing = set(COLUMN_MAP) - set(rename)
    if missing:
        logger.warning("Source is missing expected columns: %s", sorted(missing))
    return df.rename(columns=rename)


def write_snapshot(df: pd.DataFrame, out_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    """Persist the raw snapshot as both CSV and csv.

    Args:
        df: Normalized raw DataFrame to persist.
        out_dir: Destination directory, defaults to ``data/raw``.

    Returns:
        A tuple of (csv_path, csv_path) written to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "online_retail_ii.csv"
    csv_path = out_dir / "online_retail_ii.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s and %s", len(df), csv_path, csv_path)
    return csv_path, csv_path


def main() -> None:
    """CLI entry point: ingest the source file and write a raw snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source .xlsx workbook or a pre-combined .csv",
    )
    args = parser.parse_args()

    if args.source.suffix.lower() == ".xlsx":
        df = load_from_excel(args.source)
    elif args.source.suffix.lower() == ".csv":
        df = load_from_csv(args.source)
    else:
        raise ValueError(f"Unsupported source file type: {args.source.suffix}")

    df = normalize_schema(df)
    write_snapshot(df)


if __name__ == "__main__":
    main()

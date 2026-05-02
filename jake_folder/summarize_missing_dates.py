import pandas as pd
from typing import Dict, List


def summarize_missing_dates(df: pd.DataFrame,
                           date_col: str = 'Date',
                           threshold: float = 0.0) -> None:
    """
    Analyzes missing data in a wide-format price DataFrame (Date + tickers).

    Prints a nice summary showing gaps per column.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found.")

    # Ensure Date is datetime and sorted
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    ticker_cols = [col for col in df.columns if col != date_col]

    print(f"Missing Data Summary (Total rows: {len(df)})\n")

    for col in sorted(ticker_cols):
        missing = df[col].isna()
        missing_count = missing.sum()

        if missing_count == 0:
            continue  # skip clean columns unless you want to see them

        missing_pct = missing_count / len(df) * 100

        if missing_pct <= threshold:
            continue

        # Find contiguous missing periods
        gaps = []
        in_gap = False
        gap_start = None

        for i, is_missing in enumerate(missing):
            if is_missing and not in_gap:
                in_gap = True
                gap_start = df[date_col].iloc[i]
            elif not is_missing and in_gap:
                in_gap = False
                gap_end = df[date_col].iloc[i - 1]
                gaps.append((gap_start, gap_end))

        # Handle gap at the end
        if in_gap:
            gap_end = df[date_col].iloc[-1]
            gaps.append((gap_start, gap_end))

        # Print summary
        print(f"{col:>6} : {missing_count:4d} missing ({missing_pct:5.2f}%)")
        for start, end in gaps:
            if start == end:
                print(f"          → missing on {start.date()}")
            else:
                print(
                    f"          → missing from {start.date()} to {end.date()}")
        print("-" * 60)
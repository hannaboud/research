import pandas as pd
from typing import Dict, List


def summarize_missing_dates(df: pd.DataFrame,
                            date_index_name: str = 'Date',
                            threshold: float = 0.0) -> Dict:
    """
    Analyzes missing dates and returns a structured summary.
    Also prints a human-readable report.
    """
    # Convert Date column to index if needed
    if date_index_name in df.columns:
        df = df.set_index(date_index_name).copy()

    # Ensure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    ticker_cols = df.columns.tolist()

    summary: Dict = {}

    print(f"Missing Dates Summary (Total trading days: {len(df)})\n")

    for col in sorted(ticker_cols):
        missing = df[col].isna()
        missing_count = missing.sum()

        if missing_count == 0:
            continue

        missing_pct = missing_count / len(df) * 100

        if missing_pct <= threshold:
            continue

        # Find gaps
        gaps = []
        in_gap = False
        gap_start = None

        for i, is_missing in enumerate(missing):
            if is_missing and not in_gap:
                in_gap = True
                gap_start = df.index[i]
            elif not is_missing and in_gap:
                in_gap = False
                gap_end = df.index[i - 1]
                gaps.append((gap_start.date(), gap_end.date()))

        if in_gap:
            gap_end = df.index[-1].date()
            gaps.append((gap_start.date(), gap_end))

        # Store in summary dict
        summary[col] = {
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'gaps': gaps
        }

        # Print human readable version
        print(f"{col:>8} : {missing_count:4d} missing ({missing_pct:5.2f}%)")
        for start, end in gaps:
            if start == end:
                print(f"          → missing on {start}")
            else:
                print(f"          → missing from {start} to {end}")
        print("-" * 70)

    print(f"\nTotal columns with missing data: {len(summary)}")
    return summary
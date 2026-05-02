import pandas as pd
from typing import Dict


def summarize_missing_dates(df: pd.DataFrame,
                            date_index_name: str = 'Date',
                            threshold: float = 0.0) -> Dict:
    """
    Analyzes missing dates and returns a structured summary.
    Also prints a human-readable report including the first date
    with no more missing values after it.
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
    clean_start_candidates = []

    print(f"Missing Dates Summary (Total trading days: {len(df)})\n")

    for col in sorted(ticker_cols):
        missing = df[col].isna()
        missing_count = missing.sum()

        if missing_count == 0:
            clean_start_candidates.append(df.index[0])
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

        # Find the first date after which there are no more missing values for this column
        last_missing_idx = missing[
            missing].index.max() if missing.any() else None
        if last_missing_idx is not None:
            clean_start = df.index[df.index.get_loc(last_missing_idx) + 1]
            clean_start_candidates.append(clean_start)
        else:
            clean_start_candidates.append(df.index[0])

        # Store in summary
        summary[col] = {
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'gaps': gaps,
            'clean_from': clean_start.date() if 'clean_start' in locals() else
            df.index[0].date()
        }

        # Print
        print(f"{col:>8} : {missing_count:4d} missing ({missing_pct:5.2f}%)")
        for start, end in gaps:
            if start == end:
                print(f"          → missing on {start}")
            else:
                print(f"          → missing from {start} to {end}")
        print(f"          → fully clean from: {summary[col]['clean_from']}")
        print("-" * 70)

    # Global clean start: the latest "clean from" date across all columns
    if clean_start_candidates:
        global_clean_start = max(clean_start_candidates)
    else:
        global_clean_start = df.index[0]

    print(f"\nTotal columns with missing data: {len(summary)}")
    print(
        f"✅ First date with **NO missing prices across any ticker**: {global_clean_start.date()}")

    # Add global metric to summary
    summary['_global'] = {
        'first_fully_clean_date': global_clean_start.date(),
        'total_rows': len(df)
    }

    return summary
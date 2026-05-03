import pandas as pd
import lseg.data as ld
import matplotlib.pyplot as plt


def summarize_missing_dates(df: pd.DataFrame,
                            date_index_name: str = 'Date',
                            threshold: float = 0.0,
                            plot_impact: bool = True) -> Dict:
    """
    Summarizes missing dates and optionally plots market cap impact.
    """
    # === Existing logic (unchanged) ===
    if date_index_name in df.columns:
        df = df.set_index(date_index_name).copy()

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

        # Find gaps...
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
            gaps.append((gap_start.date(), df.index[-1].date()))

        last_missing_idx = missing[
            missing].index.max() if missing.any() else None
        clean_from = df.index[df.index.get_loc(
            last_missing_idx) + 1].date() if last_missing_idx is not None else \
        df.index[0].date()

        summary[col] = {
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'gaps': gaps,
            'clean_from': clean_from
        }

        print(f"{col:>8} : {missing_count:4d} missing ({missing_pct:5.2f}%)")
        for start, end in gaps:
            if start == end:
                print(f"          → missing on {start}")
            else:
                print(f"          → missing from {start} to {end}")
        print(f"          → fully clean from: {clean_from}")
        print("-" * 70)

    global_clean_start = max(
        clean_start_candidates) if clean_start_candidates else df.index[0]
    print(f"\nTotal columns with missing data: {len(summary)}")
    print(
        f"✅ First date with NO missing prices across any ticker: {global_clean_start.date()}")

    summary['_global'] = {
        'first_fully_clean_date': global_clean_start.date(),
        'total_rows': len(df)
    }

    # === Market Cap Impact Chart ===
    if plot_impact and len(summary) > 0:
        print("\nGenerating Market Cap Impact Chart...")
        try:
            market_caps = get_current_market_caps(df.columns.tolist())
            if not market_caps.empty:
                plot_market_cap_removal_impact(df, market_caps)
            else:
                print("⚠️ Could not fetch market caps for chart.")
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")

    return summary


def get_current_market_caps(ric_list: list) -> pd.Series:
    """Fetch current market caps using the working field."""
    if not ric_list:
        return pd.Series()

    try:
        ld.open_session()
        df = ld.get_data(
            universe=ric_list,
            fields=['TR.CompanyMarketCap']
        )
        ld.close_session()
        if df.empty:
            print("⚠️ Market cap query returned empty.")
            return pd.Series()

        # First column is usually the Instrument/RIC
        mcap_series = df.set_index(df.columns[0]).iloc[:, 0]
        mcap_series.name = 'MarketCap'
        print(
            f"✅ Successfully fetched market caps for {len(mcap_series)} tickers")
        return mcap_series

    except Exception as e:
        print(f"⚠️ Error fetching market caps: {e}")
        return pd.Series()


def plot_market_cap_removal_impact(df: pd.DataFrame, market_caps: pd.Series):
    """Sci-fi finance themed chart"""
    if 'Date' in df.columns:
        df = df.set_index('Date').copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    dates = df.index
    impact_pct = []
    total_mcap = market_caps.sum()

    for date in dates:
        bad_tickers = df.loc[date:].isna().any(axis=0)
        bad_tickers = bad_tickers[bad_tickers].index.tolist()
        removed_mcap = market_caps[market_caps.index.isin(bad_tickers)].sum()
        pct = (removed_mcap / total_mcap * 100) if total_mcap > 0 else 0
        impact_pct.append(pct)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.plot(dates, impact_pct, color='#00ffcc', linewidth=2.8,
            label='Market Cap at Risk')
    ax.fill_between(dates, impact_pct, color='#00ffcc', alpha=0.2)

    ax.set_facecolor('#0a0a1f')
    fig.patch.set_facecolor('#05050f')
    ax.grid(True, alpha=0.25, color='#00ffcc')

    ax.set_title(
        "S&P 500 Data Quality Impact\n% of Market Cap Removed Due to Missing Data",
        fontsize=15, color='#00ffcc', pad=20)
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('% of Total Market Cap', fontsize=12, color='white')

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#ffffff')
    ax.spines['left'].set_color('#ffffff')

    plt.xticks(rotation=45)
    plt.legend(facecolor='#0a0a1f', labelcolor='white')
    plt.tight_layout()
    plt.show()



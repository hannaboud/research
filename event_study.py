"""
Rare Earth Event Study
======================
Measures abnormal stock returns around rare earth price shock events
to estimate implied company exposure across S&P 500 companies.

HOW TO RUN:
    pip install yfinance pandas numpy scipy statsmodels
    python3 event_study.py

REQUIREMENTS:
    - event_study_sample.csv must exist (run build_sample.py first)

OUTPUT:
    abnormal_returns.csv   -- daily abnormal returns per company per event
    car_results.csv        -- cumulative abnormal returns summary
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import yfinance as yf
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Rare earth price shock events
# Format: ("YYYY-MM-DD", "description")
EVENTS = [
    ("2019-05-20", "Xi Jinping visits rare earth facility — export weapon signal"),
    ("2021-02-24", "Biden executive order on critical supply chains"),
    ("2022-10-07", "US chip export controls — rare earth retaliation fears"),
    ("2023-07-03", "China restricts gallium and germanium exports"),
    ("2023-10-20", "China expands export restrictions to graphite"),
    ("2024-12-03", "China bans germanium/gallium/antimony sales to US"),
]

# Event window: days around the event to measure abnormal returns
# (-1, 3) means 1 day before to 3 days after
EVENT_WINDOW = (-1, 3)

# Estimation window: number of trading days BEFORE the event window
# used to estimate the normal return relationship with the market
ESTIMATION_WINDOW = 200

# Market benchmark
MARKET_TICKER = "^GSPC"  # S&P 500

# Hard start date for price data download
DATA_START = "2015-01-01"
DATA_END   = "2025-06-01"


# ── LOAD COMPANY SAMPLE ────────────────────────────────────────────────────────

def load_sample():
    """
    Loads the company sample from build_sample.py output.
    Returns a dict of {ticker: company_name} and the full dataframe.
    """
    try:
        sample = pd.read_csv('event_study_sample.csv')
    except FileNotFoundError:
        print("ERROR: event_study_sample.csv not found.")
        print("Please run build_sample.py first.")
        exit(1)

    tickers = dict(zip(sample['Symbol'], sample['Security']))

    print(f"Loaded {len(tickers)} companies:")
    print(f"  High exposure  : {len(sample[sample['tier'] == 'high'])}")
    print(f"  Medium exposure: {len(sample[sample['tier'] == 'medium'])}")
    print(f"  Control group  : {len(sample[sample['tier'] == 'control'])}")

    return tickers, sample


# ── PRICE DATA ─────────────────────────────────────────────────────────────────

def fetch_prices(tickers):
    """
    Downloads daily adjusted close prices for all tickers + market index.
    Uses yfinance — free, no Bloomberg needed.
    Returns a DataFrame of daily returns.
    """
    print(f"\nDownloading price data ({DATA_START} to {DATA_END})...")
    print("This may take a minute for large samples...")

    all_tickers = list(tickers.keys()) + [MARKET_TICKER]

    # Download in batches to avoid rate limiting
    batch_size = 50
    all_data = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1} ({len(batch)} tickers)...")
        import time
        try:
            data = yf.download(
                batch,
                start=DATA_START,
                end=DATA_END,
                auto_adjust=True,
                progress=False
            )["Close"]
            all_data.append(data)
            time.sleep(3)  # wait 3 seconds between batches
        except Exception as e:
            print(f"  Warning: batch failed — {e}")
            print(f"  Waiting 30 seconds before retrying...")
            time.sleep(30)
            try:
                data = yf.download(
                    batch,
                    start=DATA_START,
                    end=DATA_END,
                    auto_adjust=True,
                    progress=False
                )["Close"]
                all_data.append(data)
            except Exception as e2:
                print(f"  Batch permanently failed: {e2}")
            continue
        
    if not all_data:
        print("ERROR: Could not download any price data.")
        print("Wait a few minutes and try again (yfinance rate limit).")
        exit(1)

    # Combine all batches
    prices = pd.concat(all_data, axis=1)

    # Remove duplicate columns (market ticker may appear twice)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Compute daily returns
    returns = prices.pct_change().dropna(how='all')

    # Report coverage
    available = [t for t in list(tickers.keys()) + [MARKET_TICKER] if t in returns.columns]
    missing = [t for t in list(tickers.keys()) + [MARKET_TICKER] if t not in returns.columns]

    print(f"  Downloaded {len(returns)} trading days")
    print(f"  Tickers available: {len(available)}")
    if missing:
        print(f"  Tickers missing  : {len(missing)} — {missing[:10]}{'...' if len(missing)>10 else ''}")

    return returns


# ── MARKET MODEL ──────────────────────────────────────────────────────────────

def estimate_market_model(stock_returns, market_returns):
    """
    OLS regression: stock_return = alpha + beta * market_return
    Estimated over the quiet pre-event estimation window.
    Returns alpha, beta, and R-squared.
    """
    X = sm.add_constant(market_returns)
    model = sm.OLS(stock_returns, X).fit()
    alpha = model.params['const']
    beta  = model.params[MARKET_TICKER]
    return alpha, beta, model.rsquared


def compute_abnormal_returns(stock_returns, market_returns, alpha, beta):
    """
    AR_t = actual_return_t - (alpha + beta * market_return_t)
    """
    expected = alpha + beta * market_returns
    return stock_returns - expected


# ── MAIN EVENT STUDY ──────────────────────────────────────────────────────────

def run_event_study():

    # Load company sample
    tickers, sample = load_sample()

    # Download price data
    returns = fetch_prices(tickers)

    if MARKET_TICKER not in returns.columns:
        print(f"ERROR: Market index {MARKET_TICKER} not in downloaded data.")
        print("Try again in a few minutes.")
        exit(1)

    market_returns = returns[MARKET_TICKER]
    trading_days   = returns.index

    all_ar_rows  = []
    all_car_rows = []

    print(f"\nRunning event study across {len(EVENTS)} events...")
    print("=" * 60)

    for event_date_str, event_desc in EVENTS:

        event_dt  = pd.to_datetime(event_date_str)
        event_idx = trading_days.searchsorted(event_dt)

        print(f"\nEvent: {event_date_str} — {event_desc[:55]}")

        # Check we have enough history
        est_start_idx = event_idx - ESTIMATION_WINDOW - abs(EVENT_WINDOW[0])
        if est_start_idx < 0:
            print(f"  Skipping — not enough history before this event")
            continue

        if event_idx >= len(trading_days):
            print(f"  Skipping — event date is beyond downloaded data")
            continue

        # Define index ranges
        est_end_idx = event_idx - abs(EVENT_WINDOW[0])
        win_start_idx = event_idx + EVENT_WINDOW[0]
        win_end_idx   = event_idx + EVENT_WINDOW[1] + 1

        # Slice market returns for each window
        est_market = market_returns.iloc[est_start_idx:est_end_idx]
        win_market = market_returns.iloc[win_start_idx:win_end_idx]

        processed = 0
        skipped   = 0

        for ticker, company_name in tickers.items():

            if ticker not in returns.columns:
                skipped += 1
                continue

            stock = returns[ticker]

            # Slice stock returns
            est_stock = stock.iloc[est_start_idx:est_end_idx]
            win_stock = stock.iloc[win_start_idx:win_end_idx]

            # Skip if too many missing values
            if est_stock.isna().sum() > 20:
                skipped += 1
                continue
            if win_stock.isna().sum() > 1:
                skipped += 1
                continue

            # Align estimation window data
            est_df = pd.concat([est_stock, est_market], axis=1).dropna()
            est_df.columns = [ticker, MARKET_TICKER]

            if len(est_df) < 100:
                skipped += 1
                continue

            # Estimate market model
            try:
                alpha, beta, r2 = estimate_market_model(
                    est_df[ticker],
                    est_df[MARKET_TICKER]
                )
            except Exception:
                skipped += 1
                continue

            # Compute abnormal returns in event window
            win_df = pd.concat([win_stock, win_market], axis=1).dropna()
            win_df.columns = [ticker, MARKET_TICKER]

            if len(win_df) == 0:
                skipped += 1
                continue

            ar = compute_abnormal_returns(
                win_df[ticker],
                win_df[MARKET_TICKER],
                alpha, beta
            )

            # Cumulative abnormal return (CAR) over the full event window
            car = ar.sum()

            # Statistical significance using estimation window residuals
            est_ar  = compute_abnormal_returns(
                est_df[ticker],
                est_df[MARKET_TICKER],
                alpha, beta
            )
            std_ar  = est_ar.std()
            n_days  = len(ar)
            t_stat  = car / (std_ar * np.sqrt(n_days)) if std_ar > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(est_df) - 2))

            # Get this company's tier
            tier_vals = sample[sample['Symbol'] == ticker]['tier'].values
            tier = tier_vals[0] if len(tier_vals) > 0 else 'unknown'

            # Get sub-industry
            sub_vals = sample[sample['Symbol'] == ticker]['GICS Sub-Industry'].values
            sub_industry = sub_vals[0] if len(sub_vals) > 0 else 'unknown'

            # Store daily abnormal returns
            for date, ar_val in ar.items():
                day_offset = trading_days.get_loc(date) - event_idx
                all_ar_rows.append({
                    "event_date"   : event_date_str,
                    "event_desc"   : event_desc,
                    "ticker"       : ticker,
                    "company"      : company_name,
                    "tier"         : tier,
                    "date"         : date,
                    "day"          : day_offset,
                    "abnormal_return": round(ar_val, 6),
                })

            # Store CAR summary
            all_car_rows.append({
                "event_date"   : event_date_str,
                "event_desc"   : event_desc,
                "ticker"       : ticker,
                "company"      : company_name,
                "tier"         : tier,
                "sub_industry" : sub_industry,
                "CAR"          : round(car, 6),
                "t_stat"       : round(t_stat, 3),
                "p_value"      : round(p_value, 4),
                "significant"  : p_value < 0.05,
                "beta"         : round(beta, 3),
                "r2"           : round(r2, 3),
            })

            processed += 1

            # Print result
            sig = "✅" if p_value < 0.05 else "  "
            print(f"  {sig} {ticker:6}  CAR={car:+.2%}  t={t_stat:+.2f}  p={p_value:.3f}  [{tier}]")

        print(f"  → Processed {processed} companies, skipped {skipped}")

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("Saving results...")

    df_ar  = pd.DataFrame(all_ar_rows)
    df_car = pd.DataFrame(all_car_rows)

    df_ar.to_csv("abnormal_returns.csv",  index=False)
    df_car.to_csv("car_results.csv", index=False)

    print(f"  abnormal_returns.csv — {len(df_ar)} rows")
    print(f"  car_results.csv      — {len(df_car)} rows")

    # ── SUMMARY ANALYSIS ──────────────────────────────────────────────────────

    if len(df_car) == 0:
        print("\nNo results to display.")
        return df_ar, df_car

    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")

    # Average CAR by tier — this is your main finding
    print("\nAverage CAR by exposure tier:")
    tier_summary = df_car.groupby('tier')['CAR'].agg(
        mean_CAR='mean',
        median_CAR='median',
        count='count',
        pct_negative=lambda x: (x < 0).mean()
    ).round(4)
    print(tier_summary.to_string())

    # Top 15 most negatively affected
    print(f"\nTop 15 most negatively affected (avg CAR across all events):")
    avg_car = (
        df_car.groupby(['ticker', 'company', 'tier', 'sub_industry'])['CAR']
        .mean()
        .sort_values()
        .head(15)
    )
    print(avg_car.to_string())

    # Significant results only
    sig_results = df_car[df_car['significant'] == True]
    print(f"\nStatistically significant results (p < 0.05): {len(sig_results)}")
    if len(sig_results) > 0:
        print(sig_results[
            ['ticker', 'company', 'tier', 'event_date', 'CAR', 'p_value']
        ].sort_values('CAR').to_string())

    # Average CAR by sub-industry
    print(f"\nAverage CAR by sub-industry (top 10 most negative):")
    sub_avg = (
        df_car.groupby('sub_industry')['CAR']
        .mean()
        .sort_values()
        .head(10)
    )
    print(sub_avg.to_string())

    return df_ar, df_car


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Rare Earth Event Study")
    print("=" * 60)
    print(f"Events        : {len(EVENTS)}")
    print(f"Event window  : {EVENT_WINDOW[0]} to +{EVENT_WINDOW[1]} days")
    print(f"Est. window   : {ESTIMATION_WINDOW} trading days")
    print(f"Data range    : {DATA_START} to {DATA_END}")
    print()

    df_ar, df_car = run_event_study()

    print(f"\nDone! Open car_results.csv to explore the full results.")
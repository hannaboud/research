"""
Event-Based Conditional Regression
=====================================
Instead of regressing over all 135 months, we only use months where
rare earth prices moved unusually — isolating the true causal relationship
between RE price shocks and stock returns.

WHY THIS APPROACH:
    The full-sample regression showed mostly positive deltas — counterintuitive
    because we'd expect RE price rises to hurt manufacturers. This happened
    because both RE prices and tech stocks rose together during the green energy
    boom (omitted variable bias).

    By restricting to event months — months where RE prices moved unusually
    relative to recent history — we remove the trend noise and isolate the
    true shock response.

HOW EVENT MONTHS ARE IDENTIFIED:
    For each month t:
        1. Compute mean and std of RE index over previous 12 months
        2. If |RE_return_t - mean| > 1.5 × std → event month
    This is data-driven (not tied to specific dates) and uses rolling
    baselines so "unusual" is always relative to recent conditions.

REGRESSION MODEL (same as before):
    Stock_return = α + δ_RE × RE_index + δ_M × Market_return + ε
    But now fitted only on event months.

HOW TO RUN:
    python3 event_regression.py

REQUIRES (in same folder):
    stock_prices_all.csv
    rare_earth_prices.csv
    proxy_prices.csv
    market_caps.csv
    event_study_sample.csv

OUTPUT:
    event_months.csv             — which months were flagged as events
    delta_results_events.csv     — regression results on event months only
    dollar_exposure_events.csv   — dollar exposure from event regression
    charts/                      — chart images
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

ROLLING_WINDOW    = 12    # months of history to define "normal"
SHOCK_THRESHOLD   = 1.5   # standard deviations to qualify as event
P_VALUE_THRESHOLD = 0.05
MIN_OBSERVATIONS  = 10    # lower than full regression — fewer event months

SMM_METALS = ["neodymium", "dysprosium", "gallium", "germanium", "cobalt"]

NASDAQ_TICKERS = [
    "AMD", "NVDA", "TSLA", "AAPL", "INTC", "QCOM", "MCHP", "NXPI",
    "SWKS", "ON", "MPWR", "KLAC", "LRCX", "AMAT", "TER", "KEYS",
    "COHR", "FSLR", "RIVN", "CSCO", "ANET", "AVGO", "GLW", "ADI",
    "TXN", "MU", "AMZN", "GOOGL", "GOOG", "APP", "ISRG", "STX", "WDC",
    "ADBE", "ADSK", "CDNS", "DDOG", "EBAY", "IDXX", "ALGN", "DXCM",
    "PODD", "HOLX", "CIEN", "LITE", "NTAP", "SNDK", "SMCI", "DELL",
    "HPQ", "HPE", "ABNB",
]

os.makedirs("charts", exist_ok=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    stocks = pd.read_csv("stock_prices_all.csv",
                         index_col=0, parse_dates=True)
    re_prices = pd.read_csv("rare_earth_prices.csv",
                            index_col=0, parse_dates=True)
    proxy = pd.read_csv("proxy_prices.csv",
                        index_col=0, parse_dates=True)
    market_caps = pd.read_csv("market_caps.csv")

    sample = pd.read_csv("event_study_sample.csv")
    sample["ticker"] = sample["Symbol"].apply(
        lambda x: f"{x}.O" if x in NASDAQ_TICKERS else f"{x}.N"
    )

    print(f"  Stocks     : {stocks.shape[1]} companies, {stocks.shape[0]} months")
    print(f"  RE prices  : {re_prices.shape[1]} metals, {re_prices.shape[0]} months")
    print(f"  Sample     : {len(sample)} companies")

    return stocks, re_prices, proxy, market_caps, sample


# ── BUILD RE INDEX ────────────────────────────────────────────────────────────

def build_smm_index(re_prices):
    available  = [m for m in SMM_METALS if m in re_prices.columns]
    re_returns = re_prices[available].pct_change()
    smm_index  = re_returns.mean(axis=1).dropna()
    smm_index.name = "SMM_RE_index"
    return smm_index


# ── IDENTIFY EVENT MONTHS ─────────────────────────────────────────────────────

def identify_event_months(re_index, window=ROLLING_WINDOW,
                          threshold=SHOCK_THRESHOLD):
    """
    For each month t, computes rolling mean and std over previous
    `window` months. Flags t as an event if:

        |RE_return_t - rolling_mean| > threshold × rolling_std

    This is purely data-driven — no specific dates hard-coded.
    Using rolling baselines means "unusual" is always relative
    to recent conditions, not the full sample average.
    """
    rolling_mean = re_index.shift(1).rolling(window).mean()
    rolling_std  = re_index.shift(1).rolling(window).std()

    # Z-score: how many standard deviations from rolling mean
    z_score = (re_index - rolling_mean) / rolling_std

    # Flag event months
    is_event = z_score.abs() > threshold

    event_months = pd.DataFrame({
        "re_return"    : re_index,
        "rolling_mean" : rolling_mean,
        "rolling_std"  : rolling_std,
        "z_score"      : z_score,
        "is_event"     : is_event,
        "direction"    : np.where(re_index > 0, "up", "down"),
    })

    n_events    = is_event.sum()
    n_up        = ((is_event) & (re_index > 0)).sum()
    n_down      = ((is_event) & (re_index < 0)).sum()
    pct_events  = n_events / len(re_index) * 100

    print(f"\nEvent month identification:")
    print(f"  Rolling window    : {window} months")
    print(f"  Threshold         : {threshold} standard deviations")
    print(f"  Total months      : {len(re_index)}")
    print(f"  Event months      : {n_events} ({pct_events:.1f}%)")
    print(f"    Price up shocks : {n_up}")
    print(f"    Price down shocks: {n_down}")
    print(f"\n  Event dates:")
    for date, row in event_months[event_months["is_event"]].iterrows():
        direction = "UP  " if row["re_return"] > 0 else "DOWN"
        print(f"    {date.strftime('%Y-%m')}  {direction}  "
              f"z={row['z_score']:+.2f}  "
              f"RE return={row['re_return']:+.1%}")

    return event_months


# ── EVENT-BASED REGRESSION ────────────────────────────────────────────────────

def run_event_regression(stock_returns, re_index, market_returns,
                         event_months):
    """
    Runs OLS regression using ONLY event months.

    Same model as full regression:
        Stock_return = α + δ_RE × RE_index + δ_M × Market_return + ε

    But fitted only on the subset of months where RE prices moved
    unusually — this isolates the true shock response and removes
    the green energy trend bias from the full-sample regression.
    """
    results = []

    # Get event month dates
    event_dates = event_months[event_months["is_event"]].index

    # Find common dates across all series
    common_idx = (stock_returns.index
                  .intersection(re_index.index)
                  .intersection(market_returns.index)
                  .intersection(event_dates))

    if len(common_idx) < MIN_OBSERVATIONS:
        print(f"  WARNING: Only {len(common_idx)} common event months — "
              f"too few for regression")
        return pd.DataFrame()

    re_aligned  = re_index.loc[common_idx]
    mkt_aligned = market_returns.loc[common_idx]

    print(f"\nRunning event-based regression...")
    print(f"  Event months available : {len(event_dates)}")
    print(f"  Common event months    : {len(common_idx)}")

    for ticker in stock_returns.columns:
        stock = stock_returns[ticker].loc[common_idx]
        df    = pd.concat([stock, re_aligned, mkt_aligned], axis=1).dropna()
        df.columns = ["stock", "RE_index", "market"]

        if len(df) < MIN_OBSERVATIONS:
            continue

        try:
            X     = sm.add_constant(df[["RE_index", "market"]])
            model = sm.OLS(df["stock"], X).fit()

            results.append({
                "ticker"        : ticker,
                "delta_RE"      : round(model.params["RE_index"], 6),
                "p_value"       : round(model.pvalues["RE_index"], 4),
                "significant"   : model.pvalues["RE_index"] < P_VALUE_THRESHOLD,
                "r2"            : round(model.rsquared, 4),
                "epsilon"       : round(model.resid.std(), 6),
                "delta_market"  : round(model.params["market"], 4),
                "n_obs"         : len(df),
            })
        except Exception:
            continue

    df_results = pd.DataFrame(results)
    sig = df_results["significant"].sum()
    print(f"  Companies processed    : {len(df_results)}")
    print(f"  Significant (p<{P_VALUE_THRESHOLD})    : "
          f"{sig} ({sig/len(df_results):.0%})")

    return df_results


# ── DOLLAR EXPOSURE ───────────────────────────────────────────────────────────

def compute_dollar_exposure(delta_results, market_caps, sample, re_index):
    re_vol = re_index.dropna().std() * np.sqrt(12)

    mcaps = market_caps.rename(columns={
        "Instrument"        : "ticker",
        "Company Market Cap": "market_cap_usd"
    })[["ticker", "market_cap_usd"]].copy()
    mcaps["market_cap_usd"] = pd.to_numeric(
        mcaps["market_cap_usd"], errors="coerce"
    )

    merged = delta_results.merge(mcaps, on="ticker", how="left")
    merged = merged.merge(
        sample[["ticker", "Security", "tier", "GICS Sub-Industry"]],
        on="ticker", how="left"
    )

    merged["dollar_exposure_usd"] = (
        merged["delta_RE"].abs() * re_vol * merged["market_cap_usd"]
    )
    merged["exposure_billions"]   = merged["dollar_exposure_usd"] / 1e9
    merged["market_cap_billions"] = merged["market_cap_usd"] / 1e9
    merged["exposure_pct_mcap"]   = (
        merged["dollar_exposure_usd"] / merged["market_cap_usd"] * 100
    )
    return merged, re_vol


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────

def print_summary(exposure, label="Event-based"):
    print(f"\n{'='*70}")
    print(f"RESULTS -- {label} Regression")
    print(f"{'='*70}")

    top15 = (exposure
             .dropna(subset=["Security", "dollar_exposure_usd"])
             .sort_values("dollar_exposure_usd", ascending=False)
             .head(15))

    print(f"\nTop 15 by implied dollar exposure:")
    print(f"\n{'Company':<28} {'Ticker':<8} {'Delta':>8} "
          f"{'Sig':>4} {'Exp $B':>8} {'%Mcap':>7} {'Tier':<10}")
    print("-" * 77)

    for _, row in top15.iterrows():
        sig = " * " if row["significant"] else "   "
        print(
            f"{str(row['Security'])[:27]:<28} "
            f"{row['ticker']:<8} "
            f"{row['delta_RE']:>8.4f}{sig}"
            f"${row['exposure_billions']:>7.1f}B "
            f"{row['exposure_pct_mcap']:>6.1f}% "
            f"{str(row.get('tier','')):<10}"
        )

    print("\n  * = statistically significant at p < 0.05")

    print(f"\nAverage delta by tier:")
    tier_table = (exposure
                  .dropna(subset=["tier"])
                  .groupby("tier")["delta_RE"]
                  .agg(mean="mean", median="median", count="count")
                  .round(4))
    print(tier_table.to_string())

    neg_pct = (exposure["delta_RE"] < 0).mean()
    print(f"\nCompanies with negative delta: {neg_pct:.0%}")
    print("(negative = hurt when RE prices rise — as expected for manufacturers)")


# ── CHART 1: EVENT MONTHS ON RE INDEX ─────────────────────────────────────────

def chart_event_months(re_index, event_months):
    """
    Shows the RE index over time with event months highlighted.
    Helps visually validate that the flagged months are genuine shocks.
    """
    print("\nBuilding chart: Event months on RE index...")

    re_levels = (1 + re_index).cumprod() * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0a0a0a")

    # Top panel: RE index level with event months shaded
    ax1.set_facecolor("#0a0a0a")
    ax1.plot(re_levels.index, re_levels.values,
             color="#4ECDC4", linewidth=1.5, zorder=3)
    ax1.fill_between(re_levels.index, re_levels.values,
                     alpha=0.1, color="#4ECDC4")

    # Shade event months
    events = event_months[event_months["is_event"]]
    for date in events.index:
        color = "#FF6B6B" if events.loc[date, "re_return"] < 0 else "#F7DC6F"
        ax1.axvspan(date - pd.DateOffset(days=15),
                    date + pd.DateOffset(days=15),
                    alpha=0.3, color=color, zorder=2)

    ax1.set_ylabel("RE Index (base=100)", color="#888888", fontsize=10)
    ax1.set_title(
        f"SMM Rare Earth Index — Event Months Highlighted\n"
        f"Yellow = price spike (up shock)  |  Red = price crash (down shock)",
        color="#ffffff", fontsize=12, pad=15
    )
    ax1.tick_params(colors="#888888")
    ax1.spines["bottom"].set_color("#333333")
    ax1.spines["left"].set_color("#333333")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    # Bottom panel: Z-score with threshold lines
    ax2.set_facecolor("#0a0a0a")
    z = event_months["z_score"].dropna()
    ax2.bar(z.index, z.values,
            color=["#FF6B6B" if v < -SHOCK_THRESHOLD
                   else "#F7DC6F" if v > SHOCK_THRESHOLD
                   else "#444444"
                   for v in z.values],
            width=20, alpha=0.8)

    ax2.axhline(SHOCK_THRESHOLD, color="#F7DC6F",
                linewidth=1, linestyle="--", alpha=0.7,
                label=f"+{SHOCK_THRESHOLD}σ threshold")
    ax2.axhline(-SHOCK_THRESHOLD, color="#FF6B6B",
                linewidth=1, linestyle="--", alpha=0.7,
                label=f"-{SHOCK_THRESHOLD}σ threshold")
    ax2.axhline(0, color="#444444", linewidth=0.5)

    ax2.set_ylabel("Z-score", color="#888888", fontsize=9)
    ax2.set_xlabel("Date", color="#888888", fontsize=10)
    ax2.tick_params(colors="#888888")
    ax2.spines["bottom"].set_color("#333333")
    ax2.spines["left"].set_color("#333333")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
    ax2.legend(facecolor="#111111", edgecolor="#333333",
               labelcolor="#aaaaaa", fontsize=8)

    plt.tight_layout()
    path = "charts/event_months_re_index.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── CHART 2: FULL vs EVENT DELTA COMPARISON ───────────────────────────────────

def chart_full_vs_event(exposure_full, exposure_event):
    """
    Scatter plot comparing delta from full regression vs event regression.
    This is the key chart — shows how much the delta changes when you
    remove the green energy trend noise.

    We expect:
        - Event deltas more negative (true shock response)
        - More scatter around zero in full regression
        - Event regression gives cleaner negative signal for high-tier companies
    """
    print("\nBuilding chart: Full vs event delta comparison...")

    merged = (exposure_full[["ticker", "Security", "tier", "delta_RE"]]
              .rename(columns={"delta_RE": "delta_full"})
              .merge(
                  exposure_event[["ticker", "delta_RE"]]
                  .rename(columns={"delta_RE": "delta_event"}),
                  on="ticker", how="inner"
              )
              .dropna(subset=["Security", "tier"]))

    tier_colors = {
        "high"   : "#FF6B6B",
        "medium" : "#F7DC6F",
        "control": "#4ECDC4",
        "low"    : "#888888"
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    for tier, group in merged.groupby("tier"):
        ax.scatter(
            group["delta_full"],
            group["delta_event"],
            c=tier_colors.get(tier, "#888888"),
            alpha=0.7, s=50,
            label=tier, zorder=3
        )

    # Reference lines
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")

    # Diagonal — if points fall on this, both regressions agree
    lims = [
        min(merged["delta_full"].min(), merged["delta_event"].min()) - 0.1,
        max(merged["delta_full"].max(), merged["delta_event"].max()) + 0.1
    ]
    ax.plot(lims, lims, "--", color="#555555",
            linewidth=1, zorder=2, label="No difference")

    # Label interesting companies
    interesting = merged[
        (merged["delta_event"].abs() > 0.4) |
        (merged["delta_full"].abs() > 0.4)
    ]
    for _, row in interesting.iterrows():
        ax.annotate(
            str(row["Security"])[:12],
            (row["delta_full"], row["delta_event"]),
            fontsize=6, color="#aaaaaa",
            xytext=(4, 4), textcoords="offset points"
        )

    ax.set_xlabel("Delta — full sample regression",
                  color="#888888", fontsize=10)
    ax.set_ylabel("Delta — event months only regression",
                  color="#888888", fontsize=10)
    ax.set_title(
        "Full Sample vs Event-Based Delta\n"
        "Points below diagonal = event regression finds MORE negative delta\n"
        "(confirms green energy trend was biasing full regression upward)",
        color="#ffffff", fontsize=11, pad=15
    )
    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#1a1a1a", linewidth=0.5)
    ax.legend(facecolor="#111111", edgecolor="#333333",
              labelcolor="#aaaaaa", fontsize=9)

    plt.tight_layout()
    path = "charts/full_vs_event_delta.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── CHART 3: DELTA BY SECTOR (EVENT) ─────────────────────────────────────────

def chart_delta_by_sector(exposure, label="Event"):
    print(f"\nBuilding chart: Delta by sector ({label})...")

    sector_delta = (exposure
                    .dropna(subset=["GICS Sub-Industry", "delta_RE"])
                    .groupby("GICS Sub-Industry")
                    .filter(lambda x: len(x) >= 2)
                    .groupby("GICS Sub-Industry")["delta_RE"]
                    .mean()
                    .sort_values())

    bar_colors = ["#FF6B6B" if v < 0 else "#4ECDC4"
                  for v in sector_delta.values]

    fig, ax = plt.subplots(figsize=(12, max(8, len(sector_delta) * 0.35)))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    bars = ax.barh(sector_delta.index, sector_delta.values,
                   color=bar_colors, alpha=0.85,
                   edgecolor="#0a0a0a", linewidth=0.5)

    for bar, val in zip(bars, sector_delta.values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right",
            color="#888888", fontsize=7
        )

    ax.axvline(0, color="#444444", linewidth=1.0, zorder=3)
    ax.set_xlabel("Average Delta (RE sensitivity)",
                  color="#888888", fontsize=10)
    ax.set_title(
        f"Average RE Sensitivity by Sector -- {label} Regression\n"
        f"Red = hurt by RE price rises  |  Teal = benefits",
        color="#ffffff", fontsize=12, pad=15
    )
    ax.tick_params(colors="#888888", labelsize=8)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    plt.tight_layout()
    path = f"charts/delta_by_sector_event.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("Event-Based Conditional Regression")
    print("="*60)
    print(f"Rolling window  : {ROLLING_WINDOW} months")
    print(f"Shock threshold : {SHOCK_THRESHOLD} standard deviations")

    # Load
    stocks, re_prices, proxy, market_caps, sample = load_data()
    stock_returns = stocks.pct_change().dropna(how="all")
    smm_index     = build_smm_index(re_prices)

    # Market benchmark
    market_returns = None
    for col in ["SPY.N", "SPY.O", "SPY", ".SPX"]:
        if col in stock_returns.columns:
            market_returns = stock_returns[col]
            print(f"\nMarket benchmark: {col}")
            break
    if market_returns is None:
        market_returns = stock_returns.mean(axis=1)
        print("\nMarket benchmark: equal-weighted average")

    # Identify event months
    event_months = identify_event_months(smm_index)
    event_months.to_csv("event_months.csv")
    print("Saved event_months.csv")

    # Run event regression
    delta_events = run_event_regression(
        stock_returns, smm_index, market_returns, event_months
    )
    delta_events.to_csv("delta_results_events.csv", index=False)
    print("Saved delta_results_events.csv")

    # Dollar exposure
    exposure_events, re_vol = compute_dollar_exposure(
        delta_events, market_caps, sample, smm_index
    )
    exposure_events.to_csv("dollar_exposure_events.csv", index=False)
    print("Saved dollar_exposure_events.csv")

    # Print summary
    print_summary(exposure_events, "Event-based")

    # Load full regression results for comparison
    try:
        exposure_full = pd.read_csv("dollar_exposure.csv")
        has_full = True
    except FileNotFoundError:
        has_full = False
        print("\nNote: dollar_exposure.csv not found -- "
              "run delta_regression.py first for comparison chart")

    # Charts
    print("\n" + "="*60)
    print("Building charts...")
    print("="*60)

    chart_event_months(smm_index, event_months)
    chart_delta_by_sector(exposure_events, "Event-based")

    if has_full:
        chart_full_vs_event(exposure_full, exposure_events)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print("\nOutput files:")
    print("  event_months.csv              -- flagged event months")
    print("  delta_results_events.csv      -- regression on event months")
    print("  dollar_exposure_events.csv    -- dollar exposure estimates")
    print("  charts/event_months_re_index.png   -- event months visualized")
    print("  charts/delta_by_sector_event.png   -- sector breakdown")
    print("  charts/full_vs_event_delta.png     -- comparison chart")

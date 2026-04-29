"""
Delta Regression + Dollar Exposure Estimation
==============================================
Regresses each S&P 500 company's stock returns against a rare earth
price index to estimate:

    δ (delta)   — sensitivity of the stock to RE price changes
    ε (epsilon) — unexplained residual risk after controlling for RE and market
    Dollar exposure — implied $ value at risk from RE price moves

Regression model:
    Stock return = α + δ_RE × RE_index + δ_M × Market_return + ε

Two RE indices are used and compared:
    1. SMM Composite — equal-weighted average of 5 rare earth spot prices
                       (neodymium, dysprosium, gallium, germanium, cobalt)
                       from Shanghai Metals Market via Refinitiv
    2. REMX          — VanEck Rare Earth/Strategic Metals ETF
                       (tracks mining companies, more tradable proxy)

HOW TO RUN:
    python3 delta_regression.py

REQUIRES (in same folder):
    stock_prices_all.csv     — monthly prices for 164 S&P 500 companies
    rare_earth_prices.csv    — monthly SMM spot prices for 6 metals
    proxy_prices.csv         — monthly prices for proxy assets incl. REMX
    market_caps.csv          — current market caps from Refinitiv
    event_study_sample.csv   — company sample with tier classifications

OUTPUT FILES:
    delta_results_smm.csv    — regression results using SMM composite index
    delta_results_remx.csv   — regression results using REMX ETF
    dollar_exposure.csv      — dollar exposure estimates
    charts/                  — saved chart images
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import warnings
import os
warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Significance threshold for delta
P_VALUE_THRESHOLD = 0.05

# Minimum months of data required per company
MIN_OBSERVATIONS = 24

# Metals to include in SMM composite index
SMM_METALS = ["neodymium", "dysprosium", "gallium", "germanium", "cobalt"]
# Note: lithium excluded — only 35 months of data vs 135 for others

# Nasdaq-listed tickers (get .O suffix in Refinitiv)
NASDAQ_TICKERS = [
    "AMD", "NVDA", "TSLA", "AAPL", "INTC", "QCOM", "MCHP", "NXPI",
    "SWKS", "ON", "MPWR", "KLAC", "LRCX", "AMAT", "TER", "KEYS",
    "COHR", "FSLR", "RIVN", "CSCO", "ANET", "AVGO", "GLW", "ADI",
    "TXN", "MU", "AMZN", "GOOGL", "GOOG", "APP", "ISRG", "STX", "WDC",
    "ADBE", "ADSK", "CDNS", "DDOG", "EBAY", "IDXX", "ALGN", "DXCM",
    "PODD", "HOLX", "CIEN", "LITE", "NTAP", "SNDK", "SMCI", "DELL",
    "HPQ", "HPE", "ABNB",
]

# Create output folder for charts
os.makedirs("charts", exist_ok=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    """Loads all required data files."""
    print("Loading data...")

    stocks = pd.read_csv("stock_prices_all.csv",
                         index_col=0, parse_dates=True)
    print(f"  Stocks          : {stocks.shape[1]} companies, "
          f"{stocks.shape[0]} months")

    re_prices = pd.read_csv("rare_earth_prices.csv",
                            index_col=0, parse_dates=True)
    print(f"  RE spot prices  : {re_prices.shape[1]} metals, "
          f"{re_prices.shape[0]} months")

    proxy = pd.read_csv("proxy_prices.csv",
                        index_col=0, parse_dates=True)
    print(f"  Proxy assets    : {proxy.shape[1]} assets")

    market_caps = pd.read_csv("market_caps.csv")
    print(f"  Market caps     : {len(market_caps)} companies")

    sample = pd.read_csv("event_study_sample.csv")

    # Add Refinitiv RIC ticker to sample
    sample["ticker"] = sample["Symbol"].apply(
        lambda x: f"{x}.O" if x in NASDAQ_TICKERS else f"{x}.N"
    )
    print(f"  Company sample  : {len(sample)} companies")
    print(f"    High tier     : {(sample['tier']=='high').sum()}")
    print(f"    Medium tier   : {(sample['tier']=='medium').sum()}")
    print(f"    Control tier  : {(sample['tier']=='control').sum()}")

    return stocks, re_prices, proxy, market_caps, sample


# ── INDEX CONSTRUCTION ────────────────────────────────────────────────────────

def build_smm_index(re_prices):
    """
    Builds the SMM Composite Rare Earth Index.

    Method: equal-weighted average of monthly percentage changes
    across 5 rare earth spot prices from Shanghai Metals Market.

    Equal weighting is used because:
    - We don't have reliable consumption data to weight by economic importance
    - It avoids introducing arbitrary weighting assumptions
    - Standard practice in academic commodity index construction
    """
    available = [m for m in SMM_METALS if m in re_prices.columns]
    re_returns = re_prices[available].pct_change()

    # Equal weighted index — each metal gets 1/N weight
    smm_index = re_returns.mean(axis=1)
    smm_index.name = "SMM_RE_index"

    # Drop first row (NaN from pct_change)
    smm_index = smm_index.dropna()

    print(f"\nSMM Composite Index:")
    print(f"  Metals    : {available}")
    print(f"  Months    : {len(smm_index)}")
    print(f"  Date range: {smm_index.index[0].date()} "
          f"to {smm_index.index[-1].date()}")
    print(f"  Ann. vol  : {smm_index.std() * np.sqrt(12):.1%}")

    return smm_index


def get_remx_index(proxy):
    """
    Extracts REMX (VanEck Rare Earth ETF) as an alternative RE index.

    REMX tracks mining companies that produce rare earths and strategic metals.
    It is more tradable than spot prices but reflects equity risk in addition
    to commodity price movements.
    """
    remx_col = None
    for col in ["REMX.O", "REMX"]:
        if col in proxy.columns:
            remx_col = col
            break

    if remx_col is None:
        print("  WARNING: REMX not found in proxy prices")
        return None

    remx_returns = proxy[remx_col].dropna().pct_change().dropna()
    if len(remx_returns) == 0:
        print("  WARNING: REMX has no valid price data")
        return None

    remx_returns.name = "REMX_index"

    print(f"\nREMX Index:")
    print(f"  Months    : {len(remx_returns)}")
    print(f"  Date range: {remx_returns.index[0].date()} "
          f"to {remx_returns.index[-1].date()}")
    print(f"  Ann. vol  : {remx_returns.std() * np.sqrt(12):.1%}")

    return remx_returns


# ── DELTA REGRESSION ──────────────────────────────────────────────────────────

def run_regression(stock_returns, re_index, market_returns, index_name):
    """
    Runs OLS regression for each company:

        Stock_return_t = α + δ_RE × RE_index_t + δ_M × Market_return_t + ε_t

    Parameters:
        stock_returns  : DataFrame of monthly stock returns (companies as columns)
        re_index       : Series of monthly RE index returns
        market_returns : Series of monthly market returns (S&P 500)
        index_name     : label for this index ("SMM" or "REMX")

    Returns:
        DataFrame with one row per company containing:
            delta_RE    : sensitivity to RE index (our key variable)
            p_value     : statistical significance of delta_RE
            significant : True if p_value < 0.05
            r2          : R-squared (% of stock variance explained)
            epsilon     : residual standard deviation (unexplained risk)
            delta_market: sensitivity to market (control variable)
    """
    results = []

    # Find dates common to all three series
    common_idx = (stock_returns.index
                  .intersection(re_index.index)
                  .intersection(market_returns.index))

    re_aligned = re_index.loc[common_idx]
    mkt_aligned = market_returns.loc[common_idx]

    print(f"\nRunning {index_name} regression...")
    print(f"  Common months: {len(common_idx)}")

    for ticker in stock_returns.columns:

        stock = stock_returns[ticker].loc[common_idx]

        # Combine into one dataframe and drop any remaining NAs
        df = pd.concat([stock, re_aligned, mkt_aligned], axis=1).dropna()
        df.columns = ["stock", "RE_index", "market"]

        # Skip if not enough data
        if len(df) < MIN_OBSERVATIONS:
            continue

        try:
            # Build regression: intercept + RE index + market return
            X = sm.add_constant(df[["RE_index", "market"]])
            model = sm.OLS(df["stock"], X).fit()

            results.append({
                "ticker"        : ticker,
                "index_used"    : index_name,
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
    print(f"  Companies processed: {len(df_results)}")
    print(f"  Significant (p<{P_VALUE_THRESHOLD}): {sig} "
          f"({sig/len(df_results):.0%})")

    return df_results


# ── DOLLAR EXPOSURE ───────────────────────────────────────────────────────────

def compute_dollar_exposure(delta_results, market_caps, sample, re_index):
    """
    Converts delta into implied dollar exposure at risk.

    Formula:
        Dollar exposure = |δ_RE| × σ_RE × Market Cap

    Interpretation:
        The dollar change in a company's market value for a
        1-standard-deviation annual move in rare earth prices.

    This gives a concrete, actionable number for hedge sizing —
    a company with $10B exposure should consider hedging instruments
    with $10B notional to fully offset the risk.
    """
    re_vol = re_index.dropna().std() * np.sqrt(12)
    print(f"\nRE Index annualized volatility: {re_vol:.1%}")

    # Clean market caps — keep Refinitiv ticker format
    mcaps = market_caps.rename(columns={
        "Instrument": "ticker",
        "Company Market Cap": "market_cap_usd"
    })[["ticker", "market_cap_usd"]].copy()
    mcaps["market_cap_usd"] = pd.to_numeric(
        mcaps["market_cap_usd"], errors="coerce"
    )

    # Merge: delta results + market caps + company info
    merged = delta_results.merge(mcaps, on="ticker", how="left")
    merged = merged.merge(
        sample[["ticker", "Security", "tier", "GICS Sub-Industry"]],
        on="ticker", how="left"
    )

    # Compute exposure metrics
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

def print_summary(exposure, index_name):
    """Prints a clean summary table of results."""

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — {index_name} Index")
    print(f"{'='*70}")

    # Top 15 by dollar exposure
    top15 = (exposure
             .dropna(subset=["Security", "dollar_exposure_usd"])
             .sort_values("dollar_exposure_usd", ascending=False)
             .head(15))

    print(f"\nTop 15 companies by implied dollar exposure:")
    print(f"\n{'Company':<28} {'Ticker':<8} {'Delta':>8} "
          f"{'Sig':>4} {'Exp $B':>8} {'%Mcap':>7} {'Tier':<10}")
    print("-" * 77)

    for _, row in top15.iterrows():
        sig = " * " if row["significant"] else "   "
        print(
            f"{str(row['Security'])[:27]:<28} "
            f"{row['ticker']:<8} "
            f"{row['delta_RE']:>8.4f}"
            f"{sig}"
            f"${row['exposure_billions']:>7.1f}B "
            f"{row['exposure_pct_mcap']:>6.1f}% "
            f"{str(row.get('tier','')):<10}"
        )

    print("\n  * = statistically significant at p < 0.05")

    # Average delta by tier
    print(f"\nAverage delta by exposure tier:")
    tier_table = (exposure
                  .dropna(subset=["tier"])
                  .groupby("tier")["delta_RE"]
                  .agg(mean="mean", median="median", count="count")
                  .round(4))
    print(tier_table.to_string())

    # Average delta by sector
    print(f"\nAverage delta by sub-industry (top 10 most negative):")
    sector_table = (exposure
                    .dropna(subset=["GICS Sub-Industry"])
                    .groupby("GICS Sub-Industry")["delta_RE"]
                    .mean()
                    .sort_values()
                    .head(10)
                    .round(4))
    print(sector_table.to_string())


# ── CHART 1: TOP 15 COMPANIES VS RE INDEX ─────────────────────────────────────

def chart_top15_vs_re(stocks, re_index, exposure, index_name):
    """
    Trading-style chart: normalized price lines for top 15 exposed companies
    overlaid with a thick RE index line.

    All series normalized to 100 at start date so they can be compared
    on the same axis regardless of absolute price level.
    """
    print(f"\nBuilding chart: Top 15 vs RE index ({index_name})...")

    # Get top 15 companies by dollar exposure
    top15 = (exposure
             .dropna(subset=["Security", "dollar_exposure_usd"])
             .sort_values("dollar_exposure_usd", ascending=False)
             .head(15))

    tickers = top15["ticker"].tolist()
    names   = dict(zip(top15["ticker"], top15["Security"]))

    # Get stock price levels (not returns) for the chart
    available = [t for t in tickers if t in stocks.columns]

    # Normalize to 100 at first available date
    prices = stocks[available].dropna(how="all")
    prices_norm = prices.div(prices.iloc[0]) * 100

    # Normalize RE index to 100 at same start
    re_levels = (1 + re_index).cumprod()
    re_levels = re_levels.div(re_levels.iloc[0]) * 100
    re_levels.name = index_name

    # Align dates
    common_dates = prices_norm.index.intersection(re_levels.index)
    prices_norm = prices_norm.loc[common_dates]
    re_levels   = re_levels.loc[common_dates]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    # Color palette — distinct colors for 15 companies
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#A9CCE3"
    ]

    # Plot each company as a thin colored line
    for i, ticker in enumerate(available):
        label = names.get(ticker, ticker)
        ax.plot(
            prices_norm.index,
            prices_norm[ticker],
            color=colors[i % len(colors)],
            linewidth=0.8,
            alpha=0.7,
            label=label[:20],
            zorder=2
        )

    # Plot RE index as thick white line on top
    ax.plot(
        re_levels.index,
        re_levels.values,
        color="#FFFFFF",
        linewidth=3.0,
        label=f"RE Index ({index_name})",
        zorder=5
    )

    # Formatting
    ax.set_xlabel("Date", color="#888888", fontsize=10)
    ax.set_ylabel("Normalized Price (base = 100)", color="#888888", fontsize=10)
    ax.set_title(
        f"Top 15 Companies by Rare Earth Exposure vs {index_name} Index",
        color="#ffffff", fontsize=13, pad=15
    )

    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
    ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    # Legend — two columns to fit all 15 companies
    legend = ax.legend(
        loc="upper left",
        fontsize=7,
        ncol=2,
        facecolor="#111111",
        edgecolor="#333333",
        labelcolor="#aaaaaa",
        framealpha=0.8,
    )

    plt.tight_layout()
    path = f"charts/top15_vs_re_{index_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── CHART 2: AVERAGE DELTA BY SECTOR ─────────────────────────────────────────

def chart_delta_by_sector(exposure, index_name):
    """
    Horizontal bar chart showing average delta per GICS sub-industry,
    colored by direction (negative = hurt by RE rises, positive = benefits).
    """
    print(f"\nBuilding chart: Average delta by sector ({index_name})...")

    # Compute average delta per sub-industry (min 2 companies)
    sector_delta = (exposure
                    .dropna(subset=["GICS Sub-Industry", "delta_RE"])
                    .groupby("GICS Sub-Industry")
                    .filter(lambda x: len(x) >= 2)
                    .groupby("GICS Sub-Industry")["delta_RE"]
                    .mean()
                    .sort_values())

    # Colors: negative delta = red (hurt by RE rises), positive = green
    bar_colors = ["#FF6B6B" if v < 0 else "#4ECDC4"
                  for v in sector_delta.values]

    fig, ax = plt.subplots(figsize=(12, max(8, len(sector_delta) * 0.35)))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    bars = ax.barh(
        sector_delta.index,
        sector_delta.values,
        color=bar_colors,
        alpha=0.85,
        edgecolor="#0a0a0a",
        linewidth=0.5
    )

    # Add value labels
    for bar, val in zip(bars, sector_delta.values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color="#888888",
            fontsize=7
        )

    # Zero line
    ax.axvline(0, color="#444444", linewidth=1.0, zorder=3)

    ax.set_xlabel("Average Delta (RE sensitivity)", color="#888888", fontsize=10)
    ax.set_title(
        f"Average Rare Earth Sensitivity by Sector — {index_name} Index\n"
        f"Red = hurt by RE price rises | Teal = benefits from RE price rises",
        color="#ffffff", fontsize=12, pad=15
    )

    ax.tick_params(colors="#888888", labelsize=8)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    plt.tight_layout()
    path = f"charts/delta_by_sector_{index_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── CHART 3: SMM vs REMX DELTA COMPARISON ────────────────────────────────────

def chart_smm_vs_remx(exposure_smm, exposure_remx):
    """
    Scatter plot comparing delta estimates from both indices.
    Shows whether the two methods agree on which companies are most exposed.
    Points on the diagonal = both methods agree.
    Points far from diagonal = index choice matters.
    """
    print("\nBuilding chart: SMM vs REMX delta comparison...")

    merged = exposure_smm[["ticker", "Security", "tier", "delta_RE"]].merge(
        exposure_remx[["ticker", "delta_RE"]].rename(
            columns={"delta_RE": "delta_REMX"}
        ),
        on="ticker", how="inner"
    ).dropna(subset=["Security", "tier"])

    tier_colors = {
        "high": "#FF6B6B",
        "medium": "#F7DC6F",
        "control": "#4ECDC4",
        "low": "#888888"
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    for tier, group in merged.groupby("tier"):
        ax.scatter(
            group["delta_RE"],
            group["delta_REMX"],
            c=tier_colors.get(tier, "#888888"),
            alpha=0.7,
            s=50,
            label=tier,
            zorder=3
        )

    # Diagonal reference line (perfect agreement)
    lims = [
        min(merged["delta_RE"].min(), merged["delta_REMX"].min()) - 0.05,
        max(merged["delta_RE"].max(), merged["delta_REMX"].max()) + 0.05
    ]
    ax.plot(lims, lims, "--", color="#444444", linewidth=1, zorder=2,
            label="Perfect agreement")
    ax.axhline(0, color="#333333", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=0.5)

    # Label most interesting outliers
    outliers = merged[
        (merged["delta_RE"].abs() > 0.3) |
        (merged["delta_REMX"].abs() > 0.3)
    ]
    for _, row in outliers.iterrows():
        ax.annotate(
            str(row["Security"])[:15],
            (row["delta_RE"], row["delta_REMX"]),
            fontsize=6,
            color="#aaaaaa",
            xytext=(3, 3),
            textcoords="offset points"
        )

    ax.set_xlabel("Delta (SMM Composite Index)", color="#888888", fontsize=10)
    ax.set_ylabel("Delta (REMX ETF Index)", color="#888888", fontsize=10)
    ax.set_title(
        "SMM Composite vs REMX: Do Both Methods Agree?\n"
        "Points on diagonal = consistent results across both indices",
        color="#ffffff", fontsize=12, pad=15
    )

    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#1a1a1a", linewidth=0.5)

    legend = ax.legend(
        facecolor="#111111", edgecolor="#333333",
        labelcolor="#aaaaaa", fontsize=9
    )

    plt.tight_layout()
    path = "charts/smm_vs_remx_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("Delta Regression + Dollar Exposure Estimation")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────────
    stocks, re_prices, proxy, market_caps, sample = load_data()

    # ── Compute returns ────────────────────────────────────────────────────
    stock_returns = stocks.pct_change().dropna(how="all")

    # ── Build indices ──────────────────────────────────────────────────────
    smm_index  = build_smm_index(re_prices)
    remx_index = get_remx_index(proxy)

    # ── Get market returns ─────────────────────────────────────────────────
    # Use SPY if available, otherwise S&P 500 from stock prices
    market_returns = None
    for col in ["SPY.N", "SPY.O", "SPY", ".SPX"]:
        if col in stock_returns.columns:
            market_returns = stock_returns[col]
            print(f"\nMarket benchmark: {col}")
            break

    if market_returns is None:
        # Fall back to equal-weighted average of all stocks as market proxy
        market_returns = stock_returns.mean(axis=1)
        print("\nMarket benchmark: equal-weighted average (SPY not found)")

    # ── Run regressions ────────────────────────────────────────────────────
    delta_smm  = run_regression(stock_returns, smm_index,
                                market_returns, "SMM")
    delta_smm.to_csv("delta_results_smm.csv", index=False)
    print("Saved delta_results_smm.csv")

    if remx_index is not None:
        delta_remx = run_regression(stock_returns, remx_index,
                                    market_returns, "REMX")
        delta_remx.to_csv("delta_results_remx.csv", index=False)
        print("Saved delta_results_remx.csv")
    else:
        delta_remx = None
        print("REMX index not available — skipping REMX regression")

    # ── Compute dollar exposure ────────────────────────────────────────────
    exposure_smm, re_vol = compute_dollar_exposure(
        delta_smm, market_caps, sample, smm_index
    )
    exposure_smm.to_csv("dollar_exposure.csv", index=False)
    print("Saved dollar_exposure.csv")

    if delta_remx is not None:
        exposure_remx, _ = compute_dollar_exposure(
            delta_remx, market_caps, sample, remx_index
        )
        exposure_remx.to_csv("dollar_exposure_remx.csv", index=False)
        print("Saved dollar_exposure_remx.csv")

    # ── Print summaries ────────────────────────────────────────────────────
    print_summary(exposure_smm, "SMM")

    if delta_remx is not None:
        print_summary(exposure_remx, "REMX")

    # ── Build charts ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Building charts...")
    print("="*60)

    chart_top15_vs_re(stocks, smm_index, exposure_smm, "SMM")

    if remx_index is not None:
        chart_top15_vs_re(stocks, remx_index, exposure_remx, "REMX")

    chart_delta_by_sector(exposure_smm, "SMM")

    if delta_remx is not None:
        chart_smm_vs_remx(exposure_smm, exposure_remx)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print("\nOutput files:")
    print("  delta_results_smm.csv    — regression results (SMM index)")
    print("  delta_results_remx.csv   — regression results (REMX index)")
    print("  dollar_exposure.csv      — dollar exposure estimates")
    print("  charts/                  — all chart images")
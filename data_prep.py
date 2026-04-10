import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────────

INPUT_FILE  = "daily_prices_top30_clean.csv"
OUTPUT_FILE = "log_returns.csv"

# Companies to drop due to insufficient history
DROP_TICKERS = ["VRT.N"]

# Market cap weights from our earlier analysis (in billions)
MARKET_CAPS = {
    "NVDA.O"  : 4070.74,
    "AAPL.O"  : 3658.03,
    "GOOG.O"  : 3315.54,
    "GOOGL.O" : 3315.54,
    "AVGO.O"  : 1423.62,
    "TSLA.O"  : 1357.74,
    "MU.O"    : 402.85,
    "AMD.O"   : 329.31,
    "CAT.N"   : 323.56,
    "GE.N"    : 295.49,
    "AMAT.O"  : 267.58,
    "LRCX.O"  : 264.00,
    "RTX.N"   : 255.34,
    "INTC.O"  : 216.56,
    "KLAC.O"  : 189.17,
    "TXN.O"   : 173.29,
    "ADI.O"   : 150.09,
    "BA.N"    : 149.72,
    "LMT.N"   : 141.93,
    "ETN.N"   : 138.65,
    "QCOM.O"  : 135.63,
    "PH.N"    : 112.14,
    "NOC.N"   : 96.36,
    "GD.N"    : 93.92,
    "WDC.O"   : 93.35,
    "HWM.N"   : 91.37,
    "STX.O"   : 85.15,
    "FCX.N"   : 80.83,
    "HON.O"   : 130.00,
}

# ── LOAD DATA ──────────────────────────────────────────────────

print("="*60)
print("Data Preparation")
print("="*60)

print("\nLoading prices...")
prices = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
print(f"  Loaded: {prices.shape[0]} days x {prices.shape[1]} instruments")

# Drop VRT and SPY
drop = DROP_TICKERS + ["SPY.N"]
prices = prices.drop(columns=[c for c in drop if c in prices.columns])
print(f"  After dropping {drop}: {prices.shape[1]} instruments")

# ── COMPUTE LOG RETURNS ────────────────────────────────────────

print("\nComputing log returns...")
log_returns = np.log(prices / prices.shift(1))

# Drop first row (NaN from shift)
log_returns = log_returns.dropna(how="all")

# Forward fill any remaining NaN (weekends already excluded
# but some instruments may have isolated missing days)
log_returns = log_returns.fillna(method="ffill")

# Drop any remaining NaN
log_returns = log_returns.dropna()

print(f"  Log returns shape: {log_returns.shape}")
print(f"  Date range: {log_returns.index[0].date()} "
      f"to {log_returns.index[-1].date()}")

# ── BUILD MARKET BENCHMARK ─────────────────────────────────────

print("\nBuilding market cap weighted benchmark...")

# Get tickers that are both in prices and in our market cap dict
# (exclude MVREMXTR from the benchmark)
stock_tickers = [t for t in log_returns.columns
                 if t != ".MVREMXTR" and t in MARKET_CAPS]

# Compute weights
mcap_values = pd.Series({t: MARKET_CAPS[t] for t in stock_tickers})
weights     = mcap_values / mcap_values.sum()

print(f"  Companies in benchmark: {len(stock_tickers)}")
print(f"  Top 5 weights:")
for ticker, w in weights.sort_values(ascending=False).head(5).items():
    print(f"    {ticker}: {w:.3%}")

# Weighted average return
log_returns["MARKET"] = (log_returns[stock_tickers]
                         .multiply(weights, axis=1)
                         .sum(axis=1))

print(f"  Market benchmark built successfully")

# ── STATIONARITY TESTS ─────────────────────────────────────────

print("\nRunning stationarity tests...")
print(f"  {'Ticker':<15} {'ADF p-value':>12} {'ADF result':>12} "
      f"{'KPSS stat':>10} {'KPSS result':>12}")
print("  " + "-"*65)

results = []
for col in log_returns.columns:
    series = log_returns[col].dropna()

    # ADF test — null hypothesis: series HAS unit root (non-stationary)
    # We WANT to reject this → p-value < 0.05
    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag="AIC")
    adf_result = "STATIONARY" if adf_p < 0.05 else "NON-STATIONARY"

    # KPSS test — null hypothesis: series IS stationary
    # We WANT to FAIL to reject → stat < critical value
    kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c")
    kpss_result = "STATIONARY" if kpss_stat < kpss_crit["5%"] else "NON-STATIONARY"

    results.append({
        "ticker"      : col,
        "adf_p"       : round(adf_p, 4),
        "adf_result"  : adf_result,
        "kpss_stat"   : round(kpss_stat, 4),
        "kpss_result" : kpss_result,
    })

    print(f"  {col:<15} {adf_p:>12.4f} {adf_result:>12} "
          f"{kpss_stat:>10.4f} {kpss_result:>12}")

stationarity_df = pd.DataFrame(results)

# Summary
both_stationary = (
    (stationarity_df["adf_result"] == "STATIONARY") &
    (stationarity_df["kpss_result"] == "STATIONARY")
).sum()

print(f"\n  Both tests confirm stationary: "
      f"{both_stationary}/{len(stationarity_df)} series")

# ── DESCRIPTIVE STATISTICS ─────────────────────────────────────

print("\nDescriptive statistics of log returns:")
desc = log_returns.describe().T
desc["skewness"] = log_returns.skew()
desc["kurtosis"] = log_returns.kurtosis()
print(desc[["mean", "std", "min", "max",
            "skewness", "kurtosis"]].round(4).to_string())

# ── SAVE ──────────────────────────────────────────────────────

log_returns.to_csv(OUTPUT_FILE)
stationarity_df.to_csv("stationarity_results.csv", index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Saved: stationarity_results.csv")
print(f"\nColumns in log_returns.csv:")
for col in log_returns.columns:
    print(f"  {col}")
print("\nData preparation complete — ready for GARCH filtering")
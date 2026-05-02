"""
China RE PPI — Full S&P 500 Multi-Factor OLS Analysis
======================================================
Extends the analysis to all S&P 500 companies.
Pulls everything live from Refinitiv — no CSV files needed.

METHODOLOGY:
    Monthly OLS with 6 controls:
    Stock_return = α + δ_PPI × PPI_lagged + δ_Market × IVV
                 + δ_VIX × VXX + δ_rates × US10Y
                 + δ_dollar × DXY + δ_oil × WTI
                 + δ_copper × COPPER + ε

    - PPI lagged 1 month (publication lag correction)
    - HC3 robust standard errors
    - Pulls live from LSEG/Refinitiv

HOW TO RUN:
    python3 sp500_analysis.py
    (Refinitiv must be open and running)
    Expected runtime: 20-40 minutes

OUTPUT:
    sp500_results.xlsx
    sp500_results.csv
    charts/sp500/
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import lseg.data as ld

warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────────

P_VALUE_THRESHOLD = 0.05
MIN_OBSERVATIONS  = 20

RE_PPI_RIC = "aCNCNHVGWM"
START_DATE = "2016-01-01"
END_DATE   = "2026-04-01"

CHARTS_DIR = Path("charts/sp500")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
(CHARTS_DIR / "scatters").mkdir(exist_ok=True)
(CHARTS_DIR / "rolling_beta").mkdir(exist_ok=True)
(CHARTS_DIR / "cumulative").mkdir(exist_ok=True)

MACRO_FACTORS = {
    "VIX_PROXY": "VXX",
    "US10Y"    : "US10YT=RR",
    "DXY"      : ".DXY",
    "WTI"      : "CLc1",
    "COPPER"   : "HGc1"
}

# ── OPEN SESSION ───────────────────────────────────────────────

ld.open_session()
print("="*70)
print("S&P 500 Full RE PPI Multi-Factor Analysis")
print(f"Significance threshold: p < {P_VALUE_THRESHOLD}")
print("="*70)

# ── PULL S&P 500 TICKERS ───────────────────────────────────────

print("\nPulling S&P 500 constituent tickers...")
sp500_data = ld.get_data(
    universe=["0#.SPX"],
    fields=["TR.RIC", "TR.CommonName", "TR.GICSSector",
            "TR.GICSSubIndustry"]
)

print(f"Raw pull shape: {sp500_data.shape}")
print(f"Columns: {list(sp500_data.columns)}")

# Extract tickers
ric_col = None
for col in ["Instrument", "TR.RIC", "RIC"]:
    if col in sp500_data.columns:
        ric_col = col
        break

if ric_col is None:
    ric_col = sp500_data.columns[0]

TICKERS = sp500_data[ric_col].dropna().tolist()
TICKERS = [t for t in TICKERS if isinstance(t, str) and len(t) > 0]

print(f"S&P 500 tickers pulled: {len(TICKERS)}")

# Build company info lookup
name_col    = None
sector_col  = None
subsect_col = None

for col in sp500_data.columns:
    if "name" in col.lower() or "common" in col.lower():
        name_col = col
    if "sector" in col.lower() and "sub" not in col.lower():
        sector_col = col
    if "sub" in col.lower() and "industry" in col.lower():
        subsect_col = col

company_info = sp500_data.set_index(ric_col)

# ── PULL PPI ───────────────────────────────────────────────────

print("\nPulling China RE PPI...")
ppi_raw = ld.get_history(
    [RE_PPI_RIC], fields=None,
    interval="monthly",
    start=START_DATE, end=END_DATE
)
if ppi_raw.empty:
    ppi_raw = ld.get_history(
        [RE_PPI_RIC], fields=["VALUE"],
        interval="monthly",
        start=START_DATE, end=END_DATE
    )

ppi_prices  = ppi_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
ppi_returns = np.log(ppi_prices / ppi_prices.shift(1)).dropna()
ppi_returns.name = "PPI_RE"
print(f"  PPI: {len(ppi_returns)} monthly observations")

# ── PULL STOCK PRICES IN BATCHES ───────────────────────────────

print(f"\nPulling {len(TICKERS)} S&P 500 stocks + IVV...")
print("This will take 15-30 minutes...")

all_prices = []
batch_size = 50
all_universe = TICKERS + ["IVV"]

for i in range(0, len(all_universe), batch_size):
    batch = all_universe[i:i+batch_size]
    batch_num = i//batch_size + 1
    total_batches = (len(all_universe) + batch_size - 1) // batch_size
    print(f"  Batch {batch_num}/{total_batches}: "
          f"pulling {len(batch)} tickers...")

    try:
        df = ld.get_history(
            batch,
            fields=["TRDPRC_1"],
            interval="daily",
            start=START_DATE,
            end=END_DATE,
            adjustments=["exchangeCorrection", "manualCorrection"]
        )
        all_prices.append(df)
    except Exception as e:
        print(f"    Retrying without adjustments...")
        try:
            df = ld.get_history(
                batch,
                fields=["TRDPRC_1"],
                interval="daily",
                start=START_DATE,
                end=END_DATE
            )
            all_prices.append(df)
        except Exception as e2:
            print(f"    Batch failed: {e2}")

prices_daily = pd.concat(all_prices, axis=1)
prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]

print(f"  Pulled: {prices_daily.shape[1]} series, "
      f"{prices_daily.shape[0]} days")

# Monthly returns
monthly_last    = prices_daily.resample("ME").last()
monthly_returns = np.log(
    monthly_last / monthly_last.shift(1)
).dropna(how="all")

# Market benchmark
if "IVV" in monthly_returns.columns:
    market_monthly = monthly_returns["IVV"]
    print(f"  Market: IVV ({len(market_monthly)} months)")
else:
    market_monthly = monthly_returns.mean(axis=1)
    print("  Market: equal weighted average")

# ── PULL MACRO FACTORS ─────────────────────────────────────────

print("\nPulling macro control variables...")
macro_dfs = []

for name, ric in MACRO_FACTORS.items():
    print(f"  → {name} ({ric})")
    try:
        if name == "US10Y":
            df = ld.get_history(
                [ric], fields=None, interval="daily",
                start=START_DATE, end=END_DATE
            )
            if not df.empty:
                for col in ['MID_YLD_1','A_YLD_1','B_YLD_1','YLDTOMAT']:
                    if col in df.columns:
                        rets = df[col].diff().dropna()
                        rets.name = name
                        macro_dfs.append(rets)
                        break
        else:
            df = ld.get_history(
                [ric], fields=["TRDPRC_1"], interval="daily",
                start=START_DATE, end=END_DATE
            )
            if not df.empty:
                rets = np.log(
                    df.iloc[:, 0] / df.iloc[:, 0].shift(1)
                ).dropna()
                rets.name = name
                macro_dfs.append(rets)
                print(f"    OK: {len(rets)} observations")
    except Exception as e:
        print(f"    Failed: {e}")

if macro_dfs:
    macro_daily   = pd.concat(macro_dfs, axis=1)
    macro_monthly = macro_daily.resample("ME").last().ffill()
    print(f"  Macro factors: {list(macro_monthly.columns)}")
else:
    macro_monthly = pd.DataFrame(index=monthly_returns.index)

# ── RUN REGRESSIONS ────────────────────────────────────────────

print(f"\nRunning regressions for {len(TICKERS)} companies...")

ppi_lagged = ppi_returns.shift(1)
common_idx = (
    ppi_lagged.dropna().index
    .intersection(monthly_returns.index)
    .intersection(macro_monthly.index)
    .intersection(market_monthly.index)
)

print(f"  Common observations: {len(common_idx)}")
print(f"  Date range: {common_idx[0].date()} to {common_idx[-1].date()}")

results_list = []
failed       = []

for idx, ticker in enumerate(TICKERS):
    if idx % 50 == 0:
        print(f"  Progress: {idx}/{len(TICKERS)}...")

    if ticker not in monthly_returns.columns:
        failed.append(ticker)
        continue

    stock  = monthly_returns[ticker]
    df_reg = pd.concat(
        [stock, ppi_lagged, market_monthly, macro_monthly],
        axis=1
    ).loc[common_idx].dropna()

    if len(df_reg) < MIN_OBSERVATIONS:
        failed.append(ticker)
        continue

    df_reg = df_reg.apply(pd.to_numeric, errors='coerce').dropna()
    col_names = (["stock", "PPI", "Market"] +
                 list(macro_monthly.columns))[:df_reg.shape[1]]
    df_reg.columns = col_names

    try:
        X     = sm.add_constant(df_reg.drop(columns=["stock"]))
        model = sm.OLS(
            df_reg["stock"], X.astype(float)
        ).fit(cov_type='HC3')

        # Get company info
        company = ticker
        sector  = "Unknown"
        subsect = "Unknown"

        if ticker in company_info.index:
            row = company_info.loc[ticker]
            if name_col and name_col in company_info.columns:
                company = str(row[name_col])
            if sector_col and sector_col in company_info.columns:
                sector = str(row[sector_col])
            if subsect_col and subsect_col in company_info.columns:
                subsect = str(row[subsect_col])

        results_list.append({
            "ticker"         : ticker,
            "company"        : company,
            "sector"         : sector,
            "subsector"      : subsect,
            "delta_PPI"      : round(model.params.get("PPI", np.nan), 6),
            "p_value_PPI"    : round(model.pvalues.get("PPI", np.nan), 4),
            "significant_PPI": model.pvalues.get("PPI", 1) < P_VALUE_THRESHOLD,
            "r2"             : round(model.rsquared, 4),
            "n_obs"          : len(df_reg),
        })
    except Exception:
        failed.append(ticker)

results_df = pd.DataFrame(results_list).sort_values("delta_PPI")
print(f"\n  Processed: {len(results_df)} companies")
print(f"  Failed   : {len(failed)}")

# ── RESULTS SUMMARY ────────────────────────────────────────────

sig_df = results_df[results_df["significant_PPI"]]
neg_df = results_df[results_df["delta_PPI"] < 0]
neg_sig = sig_df[sig_df["delta_PPI"] < 0]
pos_sig = sig_df[sig_df["delta_PPI"] > 0]

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print("="*70)
print(f"Period           : {common_idx[0].date()} to {common_idx[-1].date()}")
print(f"Companies        : {len(results_df)}")
print(f"Significant p<{P_VALUE_THRESHOLD} : {len(sig_df)} "
      f"({len(sig_df)/len(results_df)*100:.1f}%)")
print(f"  Negative δ     : {len(neg_sig)} — hurt by RE rises")
print(f"  Positive δ     : {len(pos_sig)} — benefit from RE rises")
print(f"Negative delta   : {len(neg_df)} ({len(neg_df)/len(results_df)*100:.1f}%)")
print(f"Average R²       : {results_df['r2'].mean():.4f}")

# Significant companies
if len(sig_df) > 0:
    print(f"\n{'='*70}")
    print(f"SIGNIFICANT COMPANIES (p < {P_VALUE_THRESHOLD})")
    print("="*70)
    print(f"\n{'Ticker':<10} {'Company':<30} {'Sector':<25} "
          f"{'δ PPI':>8} {'p-val':>7} {'R²':>6}")
    print("-"*88)

    for _, row in sig_df.sort_values("delta_PPI").iterrows():
        direction = "▼" if row["delta_PPI"] < 0 else "▲"
        company = str(row["company"])[:29]
        sector  = str(row["sector"])[:24]
        print(f"{row['ticker']:<10} {company:<30} {sector:<25} "
              f"{row['delta_PPI']:>8.4f} "
              f"{row['p_value_PPI']:>7.4f} "
              f"{row['r2']:>6.4f} {direction}")

    print(f"\n  ▼ = hurt by RE price rises")
    print(f"  ▲ = benefits from RE price rises")

# By sector
print(f"\n{'='*70}")
print("SIGNIFICANT COMPANIES BY SECTOR")
print("="*70)
if len(sig_df) > 0:
    for sector in sig_df["sector"].unique():
        sector_sig = sig_df[sig_df["sector"] == sector]
        print(f"\n  {sector}:")
        for _, row in sector_sig.iterrows():
            print(f"    {row['ticker']:<10} {str(row['company']):<30} "
                  f"δ={row['delta_PPI']:.4f} p={row['p_value_PPI']:.4f}")

# Full results table
print(f"\n{'='*70}")
print("FULL RESULTS TABLE")
print("="*70)
print(f"\n{'Ticker':<10} {'Company':<28} {'δ PPI':>8} "
      f"{'p-val':>7} {'Sig':>4} {'R²':>6}")
print("-"*68)

for _, row in results_df.iterrows():
    sig  = " *" if row["significant_PPI"] else "  "
    name = str(row["company"])[:27]
    print(f"{row['ticker']:<10} {name:<28} "
          f"{row['delta_PPI']:>8.4f}{sig} "
          f"{row['p_value_PPI']:>7.4f} "
          f"{row['r2']:>6.4f}")

# ── CHARTS ─────────────────────────────────────────────────────

print(f"\nBuilding charts...")

# All companies delta chart
fig, ax = plt.subplots(
    figsize=(16, max(12, len(results_df) * 0.18))
)
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

colors = ["#FF6B6B" if v < 0 else "#4ECDC4"
          for v in results_df["delta_PPI"]]
labels = results_df["ticker"].astype(str)

ax.barh(labels, results_df["delta_PPI"],
        color=colors, alpha=0.8,
        edgecolor="#0a0a0a", linewidth=0.2)
ax.axvline(0, color="#ffffff", linewidth=0.8)

for i, (_, row) in enumerate(results_df.iterrows()):
    if row["significant_PPI"]:
        ax.text(
            row["delta_PPI"] + 0.005, i, "★",
            color="#FFD700", fontsize=7, va="center"
        )

ax.set_xlabel("δ PPI — Sensitivity to China RE PPI",
              color="#888888", fontsize=10)
ax.set_title(
    "RE PPI Sensitivity — Full S&P 500\n"
    "Red = hurt by RE rises | Teal = benefits | ★ = Significant",
    color="#ffffff", fontsize=12, pad=15
)
ax.tick_params(colors="#888888", labelsize=5)
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "sp500_all_delta.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"  Saved: charts/sp500/sp500_all_delta.png")

# Significant companies charts
if len(sig_df) > 0:
    for _, row in sig_df.iterrows():
        ticker = row["ticker"]
        if ticker not in monthly_returns.columns:
            continue

        stock   = monthly_returns[ticker]
        df_plot = pd.concat([stock, ppi_lagged], axis=1).dropna()
        df_plot.columns = ["stock", "PPI"]

        # Scatter
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")
        ax.scatter(df_plot["PPI"], df_plot["stock"],
                   alpha=0.7, s=50, color="#4ECDC4")
        z = np.polyfit(df_plot["PPI"], df_plot["stock"], 1)
        p = np.poly1d(z)
        ax.plot(df_plot["PPI"], p(df_plot["PPI"]), "r--", lw=2.5,
                label=f'β={row["delta_PPI"]:.4f} '
                      f'p={row["p_value_PPI"]:.4f}')
        ax.axhline(0, color="#444444", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="#444444", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lagged PPI Monthly Return",
                      color="#888888", fontsize=10)
        ax.set_ylabel("Stock Monthly Return",
                      color="#888888", fontsize=10)
        ax.set_title(
            f"{ticker} — {str(row['company'])[:40]}\n"
            f"RE PPI Sensitivity | {str(row['sector'])[:30]}",
            color="#ffffff", fontsize=10
        )
        ax.legend(facecolor="#111111", edgecolor="#333333",
                  labelcolor="#aaaaaa")
        ax.tick_params(colors="#888888")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            CHARTS_DIR / "scatters" / f"{ticker}_scatter.png",
            dpi=200, bbox_inches="tight", facecolor="#0a0a0a"
        )
        plt.close()

        # Rolling beta
        rolling = (
            df_plot["stock"].rolling(24).cov(df_plot["PPI"]) /
            df_plot["PPI"].rolling(24).var()
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")
        ax.plot(rolling.index, rolling.values,
                color="#FF6B6B", linewidth=1.5)
        ax.axhline(0, color="#ffffff", linewidth=0.8,
                   linestyle="--", alpha=0.5)
        ax.fill_between(rolling.index, rolling.values, 0,
                        where=rolling.values < 0,
                        alpha=0.3, color="#FF6B6B")
        ax.set_title(
            f"{ticker} — Rolling 24-Month Beta to RE PPI",
            color="#ffffff", fontsize=11
        )
        ax.set_ylabel("Rolling Beta", color="#888888", fontsize=9)
        ax.tick_params(colors="#888888")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            CHARTS_DIR / "rolling_beta" / f"{ticker}_rolling_beta.png",
            dpi=200, bbox_inches="tight", facecolor="#0a0a0a"
        )
        plt.close()

    print(f"  Saved charts for {len(sig_df)} significant companies")

# ── SAVE ──────────────────────────────────────────────────────

results_df.to_excel("sp500_results.xlsx", index=False)
results_df.to_csv("sp500_results.csv", index=False)

print(f"\nSaved: sp500_results.xlsx")
print(f"Saved: sp500_results.csv")
print(f"Saved: charts/sp500/")

print(f"\n{'='*70}")
print("DONE")
print("="*70)

ld.close_session()
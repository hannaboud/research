"""
China RE PPI — Full 164 Company Multi-Factor OLS Analysis
==========================================================
Extends professor's revised_jv.py to all 164 companies in the sample.
Pulls everything live from Refinitiv — no CSV files needed.

METHODOLOGY:
    Monthly OLS with 6 controls following revised_jv.py:
    Stock_return = α + δ_PPI × PPI_lagged + δ_Market × IVV
                 + δ_VIX × VXX + δ_rates × US10Y
                 + δ_dollar × DXY + δ_oil × WTI
                 + δ_copper × COPPER + ε

    - PPI lagged 1 month to account for publication lag
    - HC3 robust standard errors
    - Pulls live from LSEG/Refinitiv

HOW TO RUN:
    python3 full_164_analysis.py
    (Refinitiv must be open and running)

OUTPUT:
    full_164_results.xlsx       — full results table
    charts/full164/             — charts for significant companies
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

CHARTS_DIR = Path("charts/full164")
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

NASDAQ_SYMBOLS = [
    "AMD","NVDA","TSLA","AAPL","INTC","QCOM","MCHP","NXPI",
    "SWKS","ON","MPWR","KLAC","LRCX","AMAT","TER","KEYS",
    "COHR","FSLR","RIVN","CSCO","ANET","AVGO","GLW","ADI",
    "TXN","MU","AMZN","GOOGL","GOOG","APP","ISRG","STX","WDC",
    "ADBE","ADSK","CDNS","DDOG","EBAY","IDXX","ALGN","DXCM",
    "PODD","HOLX","CIEN","LITE","NTAP","SNDK","SMCI","DELL",
    "HPQ","HPE","ABNB","AXON","GNRC","ONTO","WOLF","ACLS",
    "RMBS","SITM","FORM","CRUS","DIOD","VICR","SMTC",
]

# ── LOAD COMPANY SAMPLE ────────────────────────────────────────

print("="*70)
print("Full 164 Company RE PPI Multi-Factor Analysis")
print("="*70)

sample = pd.read_csv("event_study_sample.csv")
print(f"\nLoaded sample: {len(sample)} companies")
print(f"Tiers: {sample['tier'].value_counts().to_dict()}")

# Build RIC tickers from symbols
def symbol_to_ric(symbol):
    if symbol in NASDAQ_SYMBOLS:
        return f"{symbol}.O"
    else:
        return f"{symbol}.N"

sample["ticker"] = sample["Symbol"].apply(symbol_to_ric)
TICKERS = sample["ticker"].tolist()
print(f"Built {len(TICKERS)} RIC tickers")

# ── OPEN REFINITIV SESSION ─────────────────────────────────────

ld.open_session()
print("\nLSEG session opened.")

# ── PULL PPI ───────────────────────────────────────────────────

print("\nPulling China RE PPI...")
ppi_raw = ld.get_history(
    [RE_PPI_RIC],
    fields=None,
    interval="monthly",
    start=START_DATE,
    end=END_DATE
)
if ppi_raw.empty:
    ppi_raw = ld.get_history(
        [RE_PPI_RIC],
        fields=["VALUE"],
        interval="monthly",
        start=START_DATE,
        end=END_DATE
    )

ppi_prices  = ppi_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
ppi_returns = np.log(ppi_prices / ppi_prices.shift(1)).dropna()
ppi_returns.name = "PPI_RE"
print(f"  PPI: {len(ppi_returns)} monthly observations")

# ── PULL STOCK PRICES ──────────────────────────────────────────

print(f"\nPulling {len(TICKERS)} stocks + IVV (market benchmark)...")
print("This may take 2-5 minutes...")

# Pull in batches of 50 to avoid timeout
all_prices = []
batch_size = 50
all_universe = TICKERS + ["IVV"]

for i in range(0, len(all_universe), batch_size):
    batch = all_universe[i:i+batch_size]
    print(f"  Batch {i//batch_size + 1}: pulling {len(batch)} tickers...")
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
        print(f"  Batch failed: {e} — trying without adjustments...")
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
            print(f"  Batch failed again: {e2} — skipping")

prices_daily = pd.concat(all_prices, axis=1)
prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]

print(f"  Pulled: {prices_daily.shape[1]} series, {prices_daily.shape[0]} days")
print(f"  Missing values: {prices_daily.isnull().sum().mean():.0f} avg per series")

# Monthly returns
monthly_last   = prices_daily.resample("ME").last()
monthly_returns = np.log(monthly_last / monthly_last.shift(1)).dropna(how="all")

# Market benchmark
if "IVV" in monthly_returns.columns:
    market_monthly = monthly_returns["IVV"]
    print(f"  Market benchmark: IVV ({len(market_monthly)} months)")
else:
    print("  WARNING: IVV not found — using equal weighted average")
    market_monthly = monthly_returns[
        [c for c in monthly_returns.columns if c != "IVV"]
    ].mean(axis=1)

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
                        print(f"    Used field: {col}")
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
    print("  WARNING: No macro factors pulled")
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

for ticker in TICKERS:
    if ticker not in monthly_returns.columns:
        failed.append(ticker)
        continue

    stock = monthly_returns[ticker]

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

        # Get symbol for display
        sym_row = sample[sample["ticker"] == ticker]
        symbol  = sym_row["Symbol"].values[0] if len(sym_row) > 0 else ticker
        company = sym_row["Security"].values[0] if len(sym_row) > 0 else ticker
        tier    = sym_row["tier"].values[0] if len(sym_row) > 0 else "unknown"

        results_list.append({
            "ticker"         : ticker,
            "symbol"         : symbol,
            "company"        : company,
            "tier"           : tier,
            "delta_PPI"      : round(model.params.get("PPI", np.nan), 6),
            "p_value_PPI"    : round(model.pvalues.get("PPI", np.nan), 4),
            "significant_PPI": model.pvalues.get("PPI", 1) < P_VALUE_THRESHOLD,
            "r2"             : round(model.rsquared, 4),
            "n_obs"          : len(df_reg),
        })
    except Exception as e:
        failed.append(ticker)

results_df = results_df = pd.DataFrame(results_list).sort_values("delta_PPI")

print(f"  Processed: {len(results_df)} companies")
print(f"  Failed/missing: {len(failed)}")

# ── RESULTS SUMMARY ────────────────────────────────────────────

sig_df = results_df[results_df["significant_PPI"]]
neg_df = results_df[results_df["delta_PPI"] < 0]

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print("="*70)
print(f"Period           : {common_idx[0].date()} to {common_idx[-1].date()}")
print(f"Companies        : {len(results_df)}")
print(f"Significant      : {len(sig_df)} ({len(sig_df)/len(results_df)*100:.1f}%)")
print(f"Negative delta   : {len(neg_df)} ({len(neg_df)/len(results_df)*100:.1f}%)")
print(f"Average R²       : {results_df['r2'].mean():.4f}")

print(f"\n{'Ticker':<10} {'Company':<28} {'Tier':<8} "
      f"{'δ PPI':>8} {'p-val':>7} {'Sig':>4} {'R²':>6}")
print("-"*75)

for _, row in results_df.iterrows():
    sig  = " *" if row["significant_PPI"] else "  "
    name = str(row["company"])[:27]
    print(f"{row['ticker']:<10} {name:<28} {row['tier']:<8} "
          f"{row['delta_PPI']:>8.4f}{sig} "
          f"{row['p_value_PPI']:>7.4f} "
          f"{row['r2']:>6.4f}")

print("\n  * = significant at p < 0.05")

# Significant companies detail
if len(sig_df) > 0:
    print(f"\n{'='*70}")
    print(f"SIGNIFICANT COMPANIES ({len(sig_df)} total)")
    print("="*70)
    for _, row in sig_df.sort_values("delta_PPI").iterrows():
        direction = "NEGATIVE — hurt by RE rises" if row["delta_PPI"] < 0 \
                    else "POSITIVE — benefits from RE rises"
        print(f"\n  {row['ticker']} — {row['company']}")
        print(f"  Tier     : {row['tier']}")
        print(f"  δ PPI    : {row['delta_PPI']:.4f}")
        print(f"  p-value  : {row['p_value_PPI']:.4f}")
        print(f"  R²       : {row['r2']:.4f}")
        print(f"  Direction: {direction}")

# By tier
print(f"\n{'='*70}")
print("RESULTS BY TIER")
print("="*70)
for tier in ["high", "medium", "control"]:
    tier_df = results_df[results_df["tier"] == tier]
    if len(tier_df) == 0:
        continue
    tier_sig = tier_df["significant_PPI"].sum()
    tier_neg = (tier_df["delta_PPI"] < 0).sum()
    print(f"\n  {tier.upper()} tier ({len(tier_df)} companies):")
    print(f"    Significant : {tier_sig} ({tier_sig/len(tier_df)*100:.0f}%)")
    print(f"    Negative δ  : {tier_neg} ({tier_neg/len(tier_df)*100:.0f}%)")
    print(f"    Avg δ PPI   : {tier_df['delta_PPI'].mean():.4f}")
    print(f"    Avg R²      : {tier_df['r2'].mean():.4f}")

# ── CHARTS ─────────────────────────────────────────────────────

print(f"\nBuilding charts...")

# Chart 1 — All companies bar chart
fig, ax = plt.subplots(figsize=(14, max(10, len(results_df) * 0.25)))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

tier_colors = {
    "high"   : "#FF6B6B",
    "medium" : "#F7DC6F",
    "control": "#4ECDC4"
}
colors = [tier_colors.get(t, "#888888") for t in results_df["tier"]]
labels = results_df["symbol"].astype(str)

ax.barh(labels, results_df["delta_PPI"],
        color=colors, alpha=0.85,
        edgecolor="#0a0a0a", linewidth=0.3)
ax.axvline(0, color="#ffffff", linewidth=0.8)

# Mark significant
for i, (_, row) in enumerate(results_df.iterrows()):
    if row["significant_PPI"]:
        ax.text(
            row["delta_PPI"] + 0.005, i, "★",
            color="#FFD700", fontsize=8, va="center"
        )

ax.set_xlabel("δ PPI — Sensitivity to China RE PPI",
              color="#888888", fontsize=10)
ax.set_title(
    "RE PPI Sensitivity — All 164 Companies\n"
    "Red=High tier | Yellow=Medium | Teal=Control | ★=Significant",
    color="#ffffff", fontsize=12, pad=15
)
ax.tick_params(colors="#888888", labelsize=7)
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "all_companies_delta.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"  Saved: charts/full164/all_companies_delta.png")

# Chart 2 — Significant companies detailed charts
if len(sig_df) > 0:
    for _, row in sig_df.iterrows():
        ticker = row["ticker"]
        symbol = row["symbol"]

        if ticker not in monthly_returns.columns:
            continue

        stock  = monthly_returns[ticker]
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
        ax.plot(df_plot["PPI"], p(df_plot["PPI"]),
                "r--", lw=2.5,
                label=f'β={row["delta_PPI"]:.4f} p={row["p_value_PPI"]:.4f}')
        ax.axhline(0, color="#444444", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="#444444", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lagged PPI Monthly Return",
                      color="#888888", fontsize=10)
        ax.set_ylabel("Stock Monthly Return",
                      color="#888888", fontsize=10)
        ax.set_title(
            f"{symbol} — {row['company']}\nRE PPI Sensitivity",
            color="#ffffff", fontsize=11
        )
        ax.legend(facecolor="#111111", edgecolor="#333333",
                  labelcolor="#aaaaaa")
        ax.tick_params(colors="#888888")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "scatters" / f"{symbol}_scatter.png",
                    dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close()

        # Rolling beta
        rolling = (df_plot["stock"].rolling(24).cov(df_plot["PPI"]) /
                   df_plot["PPI"].rolling(24).var())

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
            f"{symbol} — Rolling 24-Month Beta to RE PPI",
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
            CHARTS_DIR / "rolling_beta" / f"{symbol}_rolling_beta.png",
            dpi=200, bbox_inches="tight", facecolor="#0a0a0a"
        )
        plt.close()

        # Cumulative split
        median_ppi = df_plot["PPI"].median()
        high_ppi   = df_plot["PPI"] > median_ppi
        cum_high   = (1 + df_plot.loc[high_ppi,  "stock"]).cumprod()
        cum_low    = (1 + df_plot.loc[~high_ppi, "stock"]).cumprod()

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")
        ax.plot(cum_high.index, cum_high.values,
                color="#FF6B6B", linewidth=1.5,
                label="High RE PPI months")
        ax.plot(cum_low.index, cum_low.values,
                color="#4ECDC4", linewidth=1.5,
                label="Low RE PPI months")
        ax.set_title(
            f"{symbol} — Cumulative Returns: High vs Low RE Price Periods",
            color="#ffffff", fontsize=11
        )
        ax.set_ylabel("Cumulative Return", color="#888888", fontsize=9)
        ax.legend(facecolor="#111111", edgecolor="#333333",
                  labelcolor="#aaaaaa")
        ax.tick_params(colors="#888888")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(
            CHARTS_DIR / "cumulative" / f"{symbol}_cumulative.png",
            dpi=200, bbox_inches="tight", facecolor="#0a0a0a"
        )
        plt.close()

    print(f"  Saved charts for {len(sig_df)} significant companies")

# ── SAVE ──────────────────────────────────────────────────────

results_df.to_excel("full_164_results.xlsx", index=False)
results_df.to_csv("full_164_results.csv", index=False)

print(f"\nSaved: full_164_results.xlsx")
print(f"Saved: full_164_results.csv")
print(f"Saved: charts/full164/")
print(f"\n{'='*70}")
print("DONE")
print("="*70)

ld.close_session()

""""
China RE PPI — Monthly OLS Robustness Check
=============================================
Uses the China Industrial ex-factory PPI for rare earth metal
smelting as a clean commodity price proxy to validate our
main GARCH-QQR findings using MVREMXTR.

WHY THIS MATTERS:
    MVREMXTR and CSI are equity indices — they contain equity
    market noise beyond pure RE price movements. The China RE PPI
    is published by the National Bureau of Statistics of China
    and directly measures prices that manufacturers pay Chinese
    smelters for RE materials. Zero equity market contamination.

    If our main findings hold with PPI — defense companies most
    exposed, miners benefit — this validates that we are capturing
    genuine RE price transmission rather than equity co-movement.

LIMITATION:
    Monthly frequency only — 119 observations from 2016 to 2026.
    Cannot run GARCH-QQR. Simple OLS only.
    Publication lag of 3-5 weeks.

HOW TO RUN:
    python3 ppi_robustness.py

REQUIRES:
    china_re_ppi.csv        — from Refinitiv CodeBook
    log_returns.csv         — from data_prep.py

OUTPUT:
    ppi_delta_results.csv   — OLS results using PPI
    ppi_comparison.csv      — PPI vs MVREMXTR delta comparison
    charts/ppi/             — comparison charts
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("charts/ppi", exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────

P_VALUE_THRESHOLD = 0.05
MIN_OBSERVATIONS  = 20   # monthly data — lower threshold

TICKERS = [
    "NVDA.O","AAPL.O","GOOG.O","GOOGL.O","AVGO.O",
    "TSLA.O","MU.O","AMD.O","AMAT.O","LRCX.O",
    "INTC.O","KLAC.O","TXN.O","ADI.O","QCOM.O",
    "RTX.N","BA.N","LMT.N","NOC.N","GD.N",
    "CAT.N","GE.N","ETN.N","PH.N","HWM.N",
    "FCX.N","WDC.O","STX.O","HON.O",
]

NASDAQ = [
    "NVDA","AAPL","GOOG","GOOGL","AVGO","TSLA","MU","AMD",
    "AMAT","LRCX","INTC","KLAC","TXN","ADI","QCOM",
    "WDC","STX","HON",
]

# ── LOAD DATA ──────────────────────────────────────────────────

print("="*60)
print("China RE PPI — Monthly OLS Robustness Check")
print("="*60)

# Load PPI
print("\nLoading China RE PPI...")
ppi_raw = pd.read_csv(
    "china_re_ppi.csv", index_col=0, parse_dates=True
)

# Extract VALUE column
if "VALUE" in ppi_raw.columns:
    ppi_prices = ppi_raw["VALUE"].dropna()
elif ppi_raw.shape[1] == 1:
    ppi_prices = ppi_raw.iloc[:, 0].dropna()
else:
    # Try to find numeric column
    for col in ppi_raw.columns:
        try:
            vals = pd.to_numeric(ppi_raw[col], errors="coerce").dropna()
            if len(vals) > 50:
                ppi_prices = vals
                break
        except Exception:
            continue

print(f"  PPI observations : {len(ppi_prices)}")
print(f"  Date range       : {ppi_prices.index[0].date()} "
      f"to {ppi_prices.index[-1].date()}")
print(f"  Latest value     : {ppi_prices.iloc[-1]:.1f}")
print(f"  Min / Max        : {ppi_prices.min():.1f} / "
      f"{ppi_prices.max():.1f}")

# Compute PPI log returns
ppi_returns = np.log(ppi_prices / ppi_prices.shift(1)).dropna()
ppi_returns.name = "PPI_RE"
print(f"  PPI monthly returns: {len(ppi_returns)} observations")
print(f"  Mean: {ppi_returns.mean():.4f} | Std: {ppi_returns.std():.4f}")

# Load daily log returns and aggregate to monthly
print("\nLoading and aggregating daily returns to monthly...")
log_returns = pd.read_csv(
    "log_returns.csv", index_col=0, parse_dates=True
)

# Aggregate to monthly — sum of daily log returns = monthly log return
monthly_returns = log_returns.resample("ME").sum()
print(f"  Monthly returns shape: {monthly_returns.shape}")

# Extract market benchmark
market_monthly = monthly_returns["MARKET"]

# Extract stock returns
stock_monthly = monthly_returns[TICKERS]

# Shift PPI by 1 month to account for publication lag
# February PPI published mid-March — so align to next month's stock returns
ppi_lagged = ppi_returns.shift(1)

# Align all series to common dates
common = (ppi_lagged.index
          .intersection(market_monthly.index)
          .intersection(stock_monthly.index))

ppi_aligned = ppi_lagged.loc[common].dropna()
market_aligned  = market_monthly.loc[common]
stock_aligned   = stock_monthly.loc[common]

print(f"  Common monthly observations: {len(common)}")
print(f"  Date range: {common[0].date()} to {common[-1].date()}")

# ── RUN OLS FOR EACH COMPANY ───────────────────────────────────

print("\nRunning monthly OLS — Stock ~ PPI + Market...")

results_list = []

for ticker in TICKERS:
    if ticker not in stock_aligned.columns:
        continue

    stock = stock_aligned[ticker]
    df    = pd.concat(
        [stock, ppi_aligned, market_aligned], axis=1
    ).dropna()
    df.columns = ["stock", "PPI", "market"]

    if len(df) < MIN_OBSERVATIONS:
        print(f"  {ticker}: insufficient observations ({len(df)})")
        continue

    try:
        X     = sm.add_constant(df[["PPI","market"]])
        model = sm.OLS(df["stock"], X).fit()

        results_list.append({
            "ticker"     : ticker,
            "delta_PPI"  : round(model.params["PPI"], 6),
            "p_value"    : round(model.pvalues["PPI"], 4),
            "significant": model.pvalues["PPI"] < P_VALUE_THRESHOLD,
            "r2"         : round(model.rsquared, 4),
            "n_obs"      : len(df),
        })
    except Exception as e:
        print(f"  {ticker}: FAILED — {e}")
        continue

results_df = pd.DataFrame(results_list)

# Add company metadata
try:
    mcaps = pd.read_csv("market_caps.csv").rename(columns={
        "Instrument"        : "ticker",
        "Company Market Cap": "market_cap_usd"
    })[["ticker","market_cap_usd"]].copy()
    mcaps["market_cap_usd"] = pd.to_numeric(
        mcaps["market_cap_usd"], errors="coerce"
    )
    sample = pd.read_csv("event_study_sample.csv")
    sample["ticker"] = sample["Symbol"].apply(
        lambda x: f"{x}.O" if x in NASDAQ else f"{x}.N"
    )
    results_df = results_df.merge(mcaps, on="ticker", how="left")
    results_df = results_df.merge(
        sample[["ticker","Security","tier","GICS Sub-Industry"]],
        on="ticker", how="left"
    )
    has_meta = True
except FileNotFoundError:
    has_meta = False

# ── PRINT RESULTS ──────────────────────────────────────────────

sig  = results_df["significant"].sum()
neg  = (results_df["delta_PPI"] < 0).sum()

print(f"\n{'='*60}")
print("PPI OLS RESULTS")
print("="*60)
print(f"\nCompanies processed   : {len(results_df)}")
print(f"Significant (p<0.05)  : {sig} ({sig/len(results_df)*100:.0f}%)")
print(f"Negative delta        : {neg} ({neg/len(results_df)*100:.0f}%)")
print(f"Average R²            : {results_df['r2'].mean():.4f}")

print(f"\n{'Ticker':<10} {'Company':<25} {'δ PPI':>8} "
      f"{'Sig':>4} {'R²':>6} {'N':>5}")
print("-"*62)

for _, row in results_df.sort_values("delta_PPI").iterrows():
    sig_mark = " *" if row["significant"] else "  "
    name = str(row.get("Security", row["ticker"]))[:24]
    print(f"{row['ticker']:<10} {name:<25} "
          f"{row['delta_PPI']:>8.4f}{sig_mark} "
          f"{row['r2']:>6.4f} {row['n_obs']:>5}")

print("\n  * = significant at p < 0.05")

# ── COMPARE PPI vs MVREMXTR ────────────────────────────────────

print(f"\n{'='*60}")
print("COMPARISON — PPI vs MVREMXTR vs CSI")
print("="*60)

# Load MVREMXTR event results
comparison_df = results_df[["ticker","delta_PPI","significant","r2"]]\
    .rename(columns={
        "delta_PPI"  : "delta_PPI",
        "significant": "sig_PPI",
        "r2"         : "r2_PPI",
    }).copy()

try:
    mv_res = pd.read_csv("event_delta_results_MVREMXTR.csv")
    comparison_df = comparison_df.merge(
        mv_res[["ticker","delta_RE","significant","r2"]].rename(columns={
            "delta_RE"   : "delta_MVREMXTR",
            "significant": "sig_MVREMXTR",
            "r2"         : "r2_MVREMXTR",
        }),
        on="ticker", how="left"
    )
    has_mv = True
except FileNotFoundError:
    has_mv = False
    print("  MVREMXTR results not found — skipping comparison")

try:
    csi_res = pd.read_csv("event_delta_results_CSI.csv")
    comparison_df = comparison_df.merge(
        csi_res[["ticker","delta_RE","significant","r2"]].rename(columns={
            "delta_RE"   : "delta_CSI",
            "significant": "sig_CSI",
            "r2"         : "r2_CSI",
        }),
        on="ticker", how="left"
    )
    has_csi = True
except FileNotFoundError:
    has_csi = False

if has_meta:
    comparison_df = comparison_df.merge(
        sample[["ticker","Security","tier"]],
        on="ticker", how="left"
    )

if has_mv:
    # Sign agreement PPI vs MVREMXTR
    comparison_df["PPI_MV_agree"] = (
        np.sign(comparison_df["delta_PPI"]) ==
        np.sign(comparison_df["delta_MVREMXTR"])
    )

    n_agree = comparison_df["PPI_MV_agree"].sum()
    n_total = comparison_df["PPI_MV_agree"].notna().sum()

    print(f"\n{'Ticker':<10} {'Company':<22} "
          f"{'δ PPI':>8} {'δ MVREMXTR':>11} {'Agree':>7}")
    print("-"*62)

    for _, row in comparison_df.sort_values(
        "delta_PPI", na_position="last"
    ).iterrows():
        name  = str(row.get("Security", row["ticker"]))[:21]
        dp    = f"{row['delta_PPI']:.4f}" \
                if pd.notna(row["delta_PPI"]) else "  nan"
        dm    = f"{row['delta_MVREMXTR']:.4f}" \
                if pd.notna(row.get("delta_MVREMXTR")) else "  nan"
        agree = "✓" if row.get("PPI_MV_agree") else "✗"
        print(f"{row['ticker']:<10} {name:<22} "
              f"{dp:>8} {dm:>11} {agree:>7}")

    print(f"\nSign agreement PPI vs MVREMXTR: "
          f"{n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)")

# ── KEY FINDING ────────────────────────────────────────────────

print(f"\n{'='*60}")
print("KEY FINDING — MISPRICING VALIDATION")
print("="*60)

neg_ppi = (results_df["delta_PPI"] < 0).sum()
pos_ppi = (results_df["delta_PPI"] > 0).sum()

print(f"""
PPI regression uses actual RE commodity prices — zero equity noise.

Companies with NEGATIVE PPI delta (hurt by RE price rises): {neg_ppi}
Companies with POSITIVE PPI delta (benefit from RE price rises): {pos_ppi}

If the majority show POSITIVE delta even with pure commodity prices,
this confirms genuine market mispricing — investors benefit from
RE price rises even though the cost channel should hurt manufacturers.

If the majority show NEGATIVE delta with PPI but POSITIVE with MVREMXTR,
this suggests the positive MVREMXTR results were equity co-movement noise
rather than true RE price sensitivity.
""")

# ── CHARTS ────────────────────────────────────────────────────

print("Building charts...")

# Chart 1 — PPI delta by company
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

sorted_df = results_df.sort_values("delta_PPI")
if has_meta:
    labels = sorted_df["Security"].fillna(
        sorted_df["ticker"]
    ).astype(str)
else:
    labels = sorted_df["ticker"]

colors = ["#FF6B6B" if v < 0 else "#4ECDC4"
          for v in sorted_df["delta_PPI"]]

ax.barh(labels, sorted_df["delta_PPI"],
        color=colors, alpha=0.85,
        edgecolor="#0a0a0a", linewidth=0.5)
ax.axvline(0, color="#444444", linewidth=1.0)
ax.set_xlabel("Delta PPI (sensitivity to RE PPI changes)",
              color="#888888", fontsize=10)
ax.set_title(
    "Monthly OLS — China RE PPI (Pure Commodity Price)\n"
    "Red = hurt by RE price rises | Teal = benefits\n"
    "* = statistically significant at p < 0.05",
    color="#ffffff", fontsize=11, pad=15
)

# Mark significant companies
for i, (_, row) in enumerate(sorted_df.iterrows()):
    if row["significant"]:
        ax.text(
            row["delta_PPI"] + 0.001,
            i,
            "*",
            color="#F7DC6F",
            fontsize=10,
            va="center"
        )

ax.tick_params(colors="#888888", labelsize=8)
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

plt.tight_layout()
plt.savefig("charts/ppi/ppi_delta.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print("  Saved: charts/ppi/ppi_delta.png")

# Chart 2 — PPI vs MVREMXTR scatter
if has_mv and "delta_MVREMXTR" in comparison_df.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    x = comparison_df["delta_MVREMXTR"].values
    y = comparison_df["delta_PPI"].values
    mask = np.isfinite(x) & np.isfinite(y)

    ax.scatter(x[mask], y[mask],
               color="#4ECDC4", alpha=0.7, s=60, zorder=3)

    # Reference lines
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")

    # Diagonal — perfect agreement
    lims = [
        min(x[mask].min(), y[mask].min()) - 0.05,
        max(x[mask].max(), y[mask].max()) + 0.05
    ]
    ax.plot(lims, lims, "--", color="#555555",
            linewidth=1, label="Perfect agreement")

    # Regression line
    m, b = np.polyfit(x[mask], y[mask], 1)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, m * x_line + b,
            color="#F7DC6F", linewidth=1.5, label="Fitted line")

    # Label companies
    if has_meta:
        for _, row in comparison_df.iterrows():
            if pd.notna(row.get("delta_MVREMXTR")) and \
               pd.notna(row["delta_PPI"]):
                ax.annotate(
                    str(row.get("Security", row["ticker"]))[:8],
                    (row["delta_MVREMXTR"], row["delta_PPI"]),
                    fontsize=6, color="#aaaaaa",
                    xytext=(3, 3), textcoords="offset points"
                )

    corr = pd.Series(x[mask]).corr(pd.Series(y[mask]))
    ax.set_xlabel("Delta MVREMXTR (daily GARCH-filtered)",
                  color="#888888", fontsize=10)
    ax.set_ylabel("Delta PPI (monthly pure commodity)",
                  color="#888888", fontsize=10)
    ax.set_title(
        f"PPI vs MVREMXTR Delta Estimates\n"
        f"Correlation = {corr:.4f} | "
        f"Sign agreement = {n_agree}/{n_total}",
        color="#ffffff", fontsize=11, pad=15
    )
    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#1a1a1a", linewidth=0.5)
    ax.legend(facecolor="#111111", edgecolor="#333333",
              labelcolor="#aaaaaa", fontsize=8)

    plt.tight_layout()
    plt.savefig("charts/ppi/ppi_vs_mvremxtr_scatter.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print("  Saved: charts/ppi/ppi_vs_mvremxtr_scatter.png")

# ── SAVE ──────────────────────────────────────────────────────

results_df.to_csv("ppi_delta_results.csv", index=False)
comparison_df.to_csv("ppi_comparison.csv", index=False)

print(f"\nSaved: ppi_delta_results.csv")
print(f"Saved: ppi_comparison.csv")
print("\nPPI robustness check complete")
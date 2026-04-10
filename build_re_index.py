"""
RE Spot Price Index Builder & Basis Risk Analysis
==================================================
Builds two SMM spot price indices and compares them to MVREMXTR
to quantify basis risk between mining equity index and actual metal prices.

THREE INDICES:
    1. MVREMXTR       — MVIS Global RE/Strategic Metals Index (equity-based)
    2. SMM_EW_INDEX   — Equal weighted SMM spot prices (12 metals)
    3. SMM_UW_INDEX   — Usage weighted SMM spot prices (6 metals, weighted
                        by industrial relevance to our company sample)

WHY THIS MATTERS:
    MVREMXTR tracks mining company stocks — contaminated by equity market
    noise unrelated to actual metal prices. If correlation between MVREMXTR
    and our SMM indices is low, basis risk is significant and using MVREMXTR
    in our regression understates true RE price sensitivity.

HOW TO RUN:
    python3 build_re_index.py

REQUIRES:
    re_spot_prices.csv      — from Refinitiv CodeBook pull
    log_returns.csv         — from data_prep.py

OUTPUT:
    re_index_comparison.csv — all three indices aligned
    re_index_returns.csv    — log returns for all three indices
    charts/indices/         — comparison charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("charts/indices", exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────

# Usage weights — based on industrial importance for our company sample
# Tier 1 (20% each): Nd, Dy, Ga, Ge — semiconductors and defense
# Tier 2 (10% each): Pr, Tb — permanent magnets
# Tier 3 (0%): bulk/minor metals less relevant to our companies
USAGE_WEIGHTS = {
    "SMM-REM-NDM": 0.20,  # Neodymium    — permanent magnets, EVs, wind
    "SMM-REO-DXO": 0.20,  # Dysprosium   — high-temp magnets, defense
    "SMM-MIN-GAL": 0.20,  # Gallium      — GaN semiconductors, chips
    "SMM-MIN-GMN": 0.20,  # Germanium    — fiber optics, semiconductors
    "SMM-REM-PRT": 0.10,  # Praseodymium — magnets alongside neodymium
    "SMM-REM-TRM": 0.10,  # Terbium      — defense motors, EV drives
    "SMM-REM-LMT": 0.00,  # Lanthanum    — bulk, less relevant
    "SMM-REM-CRM": 0.00,  # Cerium       — bulk, less relevant
    "SMM-REM-YMT": 0.00,  # Yttrium      — minor
    "SMM-REM-GXO": 0.00,  # Gadolinium   — minor
    "SMM-REO-SXO": 0.00,  # Samarium     — minor
    "SMM-REO-EXO": 0.00,  # Europium     — minor
}

METAL_NAMES = {
    "SMM-REM-NDM": "Neodymium",
    "SMM-REO-DXO": "Dysprosium",
    "SMM-MIN-GAL": "Gallium",
    "SMM-MIN-GMN": "Germanium",
    "SMM-REM-PRT": "Praseodymium",
    "SMM-REM-TRM": "Terbium",
    "SMM-REM-LMT": "Lanthanum",
    "SMM-REM-CRM": "Cerium",
    "SMM-REM-YMT": "Yttrium",
    "SMM-REM-GXO": "Gadolinium",
    "SMM-REO-SXO": "Samarium",
    "SMM-REO-EXO": "Europium",
}

# ── LOAD DATA ──────────────────────────────────────────────────

print("="*60)
print("RE Spot Price Index Builder & Basis Risk Analysis")
print("="*60)

print("\nLoading SMM spot prices...")
spot = pd.read_csv(
    "re_spot_prices.csv", index_col=0, parse_dates=True
)
# Forward fill isolated missing values
spot = spot.ffill()
print(f"  Shape: {spot.shape}")
print(f"  Date range: {spot.index[0].date()} to {spot.index[-1].date()}")
print(f"  Metals: {list(spot.columns)}")

print("\nLoading MVREMXTR log returns...")
log_returns = pd.read_csv(
    "log_returns.csv", index_col=0, parse_dates=True
)
mvremxtr = log_returns[".MVREMXTR"]
print(f"  MVREMXTR observations: {len(mvremxtr)}")

# ── COMPUTE LOG RETURNS FOR EACH METAL ────────────────────────

print("\nComputing log returns for each metal...")
log_ret_spot = np.log(spot / spot.shift(1)).dropna()
print(f"  Shape: {log_ret_spot.shape}")

# Print individual metal stats
print(f"\n  {'Metal':<15} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
print("  " + "-"*50)
for col in log_ret_spot.columns:
    name = METAL_NAMES.get(col, col)[:14]
    print(f"  {name:<15} "
          f"{log_ret_spot[col].mean():>8.5f} "
          f"{log_ret_spot[col].std():>8.5f} "
          f"{log_ret_spot[col].skew():>8.3f} "
          f"{log_ret_spot[col].kurtosis():>8.3f}")

# ── BUILD EQUAL WEIGHTED INDEX ─────────────────────────────────

print("\nBuilding Equal Weighted SMM Index...")
eq_index = log_ret_spot.mean(axis=1)
eq_index.name = "SMM_EW_INDEX"
print(f"  Mean return : {eq_index.mean():.6f}")
print(f"  Std         : {eq_index.std():.6f}")
print(f"  Skewness    : {eq_index.skew():.4f}")
print(f"  Kurtosis    : {eq_index.kurtosis():.4f}")

# ── BUILD USAGE WEIGHTED INDEX ─────────────────────────────────

print("\nBuilding Usage Weighted SMM Index...")
w_series = pd.Series(USAGE_WEIGHTS)
print(f"  Weights:")
for ticker, w in w_series[w_series > 0].items():
    name = METAL_NAMES.get(ticker, ticker)
    print(f"    {name:<15}: {w:.0%}")

uw_index = log_ret_spot.multiply(w_series, axis=1).sum(axis=1)
uw_index.name = "SMM_UW_INDEX"
print(f"\n  Mean return : {uw_index.mean():.6f}")
print(f"  Std         : {uw_index.std():.6f}")
print(f"  Skewness    : {uw_index.skew():.4f}")
print(f"  Kurtosis    : {uw_index.kurtosis():.4f}")

# ── ALIGN ALL THREE INDICES ────────────────────────────────────

print("\nAligning all three indices to common dates...")
common = (eq_index.index
          .intersection(uw_index.index)
          .intersection(mvremxtr.index))

comparison = pd.DataFrame({
    "MVREMXTR"     : mvremxtr.loc[common],
    "SMM_EW_INDEX" : eq_index.loc[common],
    "SMM_UW_INDEX" : uw_index.loc[common],
})

print(f"  Common observations: {len(comparison)}")
print(f"  Date range: {comparison.index[0].date()} "
      f"to {comparison.index[-1].date()}")

# ── BASIS RISK ANALYSIS ────────────────────────────────────────

print(f"\n{'='*60}")
print("BASIS RISK ANALYSIS")
print("="*60)

print(f"\nCorrelation Matrix:")
corr = comparison.corr().round(4)
print(corr.to_string())

print(f"\nKey Correlations:")
c_ew = corr.loc["MVREMXTR", "SMM_EW_INDEX"]
c_uw = corr.loc["MVREMXTR", "SMM_UW_INDEX"]
c_indices = corr.loc["SMM_EW_INDEX", "SMM_UW_INDEX"]

print(f"  MVREMXTR vs SMM Equal Weighted   : {c_ew:.4f}")
print(f"  MVREMXTR vs SMM Usage Weighted   : {c_uw:.4f}")
print(f"  SMM Equal vs SMM Usage Weighted  : {c_indices:.4f}")

# Interpret basis risk
print(f"\nBasis Risk Interpretation:")
if c_ew < 0.3:
    print(f"  MVREMXTR vs EW Index: LOW correlation ({c_ew:.3f})")
    print(f"  → HIGH basis risk — MVREMXTR is a poor proxy for actual RE prices")
elif c_ew < 0.6:
    print(f"  MVREMXTR vs EW Index: MODERATE correlation ({c_ew:.3f})")
    print(f"  → MODERATE basis risk — MVREMXTR captures some but not all RE dynamics")
else:
    print(f"  MVREMXTR vs EW Index: HIGH correlation ({c_ew:.3f})")
    print(f"  → LOW basis risk — MVREMXTR is a reasonable proxy")

if c_uw < 0.3:
    print(f"  MVREMXTR vs UW Index: LOW correlation ({c_uw:.3f})")
    print(f"  → HIGH basis risk for semiconductor/defense exposure specifically")
elif c_uw < 0.6:
    print(f"  MVREMXTR vs UW Index: MODERATE correlation ({c_uw:.3f})")
else:
    print(f"  MVREMXTR vs UW Index: HIGH correlation ({c_uw:.3f})")

# Descriptive stats comparison
print(f"\nDescriptive Statistics:")
desc = comparison.describe()
desc.loc["skewness"] = comparison.skew()
desc.loc["kurtosis"] = comparison.kurtosis()
print(desc.round(5).to_string())

# Volatility comparison
print(f"\nAnnualized Volatility:")
for col in comparison.columns:
    ann_vol = comparison[col].std() * np.sqrt(252)
    print(f"  {col:<20}: {ann_vol:.4f} ({ann_vol*100:.2f}%)")

# ── ROLLING CORRELATION ────────────────────────────────────────

print("\nComputing rolling correlations...")
roll_corr_ew = (comparison["MVREMXTR"]
                .rolling(60)
                .corr(comparison["SMM_EW_INDEX"]))
roll_corr_uw = (comparison["MVREMXTR"]
                .rolling(60)
                .corr(comparison["SMM_UW_INDEX"]))

print(f"  60-day rolling corr MVREMXTR vs EW:")
print(f"    Mean: {roll_corr_ew.mean():.4f}")
print(f"    Min : {roll_corr_ew.min():.4f}")
print(f"    Max : {roll_corr_ew.max():.4f}")

print(f"  60-day rolling corr MVREMXTR vs UW:")
print(f"    Mean: {roll_corr_uw.mean():.4f}")
print(f"    Min : {roll_corr_uw.min():.4f}")
print(f"    Max : {roll_corr_uw.max():.4f}")

# ── CHARTS ────────────────────────────────────────────────────

print("\nBuilding charts...")

# Chart 1 — Cumulative returns of all three indices
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.patch.set_facecolor("#0a0a0a")

# Cumulative returns
cum_returns = (1 + comparison).cumprod() * 100

ax1 = axes[0]
ax1.set_facecolor("#0a0a0a")
ax1.plot(cum_returns.index, cum_returns["MVREMXTR"],
         color="#4ECDC4", linewidth=1.2, label="MVREMXTR (equity index)")
ax1.plot(cum_returns.index, cum_returns["SMM_EW_INDEX"],
         color="#F7DC6F", linewidth=1.2, label="SMM Equal Weighted (spot)")
ax1.plot(cum_returns.index, cum_returns["SMM_UW_INDEX"],
         color="#FF6B6B", linewidth=1.2, label="SMM Usage Weighted (spot)")
ax1.axhline(100, color="#444444", linewidth=0.5, linestyle="--")
ax1.set_ylabel("Cumulative Return (base=100)",
               color="#888888", fontsize=9)
ax1.set_title(
    "Cumulative Returns — Three RE Indices\n"
    "Teal = MVREMXTR equity index | "
    "Yellow = SMM equal weighted | Red = SMM usage weighted",
    color="#ffffff", fontsize=11, pad=10
)
ax1.tick_params(colors="#888888")
ax1.spines["bottom"].set_color("#333333")
ax1.spines["left"].set_color("#333333")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
ax1.legend(facecolor="#111111", edgecolor="#333333",
           labelcolor="#aaaaaa", fontsize=8)

# Rolling correlation
ax2 = axes[1]
ax2.set_facecolor("#0a0a0a")
ax2.plot(roll_corr_ew.index, roll_corr_ew.values,
         color="#F7DC6F", linewidth=1.0,
         label="MVREMXTR vs SMM Equal Weighted")
ax2.plot(roll_corr_uw.index, roll_corr_uw.values,
         color="#FF6B6B", linewidth=1.0,
         label="MVREMXTR vs SMM Usage Weighted")
ax2.axhline(0, color="#444444", linewidth=0.5)
ax2.axhline(0.5, color="#666666", linewidth=0.5,
            linestyle="--", alpha=0.5)
ax2.set_ylabel("60-Day Rolling Correlation",
               color="#888888", fontsize=9)
ax2.set_title(
    "Basis Risk — Rolling 60-Day Correlation Between MVREMXTR and SMM Spot Indices\n"
    "Low correlation = high basis risk = MVREMXTR is a poor proxy",
    color="#ffffff", fontsize=11, pad=10
)
ax2.set_ylim(-1, 1)
ax2.tick_params(colors="#888888")
ax2.spines["bottom"].set_color("#333333")
ax2.spines["left"].set_color("#333333")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
ax2.legend(facecolor="#111111", edgecolor="#333333",
           labelcolor="#aaaaaa", fontsize=8)

# Individual metal cumulative returns
ax3 = axes[2]
ax3.set_facecolor("#0a0a0a")
metal_cum = (1 + log_ret_spot).cumprod() * 100

colors_metals = [
    "#FF6B6B", "#F7DC6F", "#4ECDC4", "#A8E6CF",
    "#FFB347", "#DDA0DD", "#87CEEB", "#98FB98",
    "#F0E68C", "#FFA07A", "#20B2AA", "#9370DB"
]

for i, col in enumerate(log_ret_spot.columns):
    name = METAL_NAMES.get(col, col)
    w    = USAGE_WEIGHTS.get(col, 0)
    lw   = 1.5 if w > 0 else 0.5
    alpha = 0.9 if w > 0 else 0.3
    ax3.plot(metal_cum.index, metal_cum[col],
             color=colors_metals[i % len(colors_metals)],
             linewidth=lw, alpha=alpha, label=name)

ax3.axhline(100, color="#444444", linewidth=0.5, linestyle="--")
ax3.set_ylabel("Cumulative Return (base=100)",
               color="#888888", fontsize=9)
ax3.set_xlabel("Date", color="#888888", fontsize=9)
ax3.set_title(
    "Individual Metal Cumulative Returns\n"
    "Bold = included in usage weighted index",
    color="#ffffff", fontsize=11, pad=10
)
ax3.tick_params(colors="#888888")
ax3.spines["bottom"].set_color("#333333")
ax3.spines["left"].set_color("#333333")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
ax3.legend(facecolor="#111111", edgecolor="#333333",
           labelcolor="#aaaaaa", fontsize=7,
           ncol=3, loc="upper left")

plt.tight_layout()
path = "charts/indices/re_index_comparison.png"
plt.savefig(path, dpi=150, bbox_inches="tight",
            facecolor="#0a0a0a")
plt.close()
print(f"  Saved: {path}")

# Chart 2 — Scatter plots MVREMXTR vs SMM indices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0a0a0a")

for ax, col, label, color in [
    (ax1, "SMM_EW_INDEX", "Equal Weighted", "#F7DC6F"),
    (ax2, "SMM_UW_INDEX", "Usage Weighted", "#FF6B6B"),
]:
    ax.set_facecolor("#0a0a0a")
    ax.scatter(
        comparison[col],
        comparison["MVREMXTR"],
        alpha=0.3, s=5, color=color
    )

    # Regression line
    x = comparison[col].values
    y = comparison["MVREMXTR"].values
    mask = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[mask], y[mask], 1)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, m * x_line + b,
            color="#ffffff", linewidth=1.5, alpha=0.8)

    corr_val = comparison[col].corr(comparison["MVREMXTR"])
    ax.set_xlabel(f"SMM {label} Index Return",
                  color="#888888", fontsize=9)
    ax.set_ylabel("MVREMXTR Return", color="#888888", fontsize=9)
    ax.set_title(
        f"MVREMXTR vs SMM {label}\n"
        f"Correlation = {corr_val:.4f}",
        color="#ffffff", fontsize=10, pad=10
    )
    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#1a1a1a", linewidth=0.5)

plt.suptitle(
    "Basis Risk Scatter Plots — MVREMXTR vs SMM Spot Price Indices\n"
    "Low R² = high basis risk = equity index diverges from actual metal prices",
    color="#ffffff", fontsize=10, y=1.02
)
plt.tight_layout()
path = "charts/indices/basis_risk_scatter.png"
plt.savefig(path, dpi=150, bbox_inches="tight",
            facecolor="#0a0a0a")
plt.close()
print(f"  Saved: {path}")

# ── SAVE ──────────────────────────────────────────────────────

comparison.to_csv("re_index_comparison.csv")

# Also save individual metal returns for later use
log_ret_spot.to_csv("re_metal_returns.csv")

# Save index returns separately for pipeline use
index_returns = pd.DataFrame({
    "SMM_EW_INDEX": eq_index,
    "SMM_UW_INDEX": uw_index,
})
index_returns.to_csv("re_index_returns.csv")

print(f"\nSaved: re_index_comparison.csv")
print(f"Saved: re_metal_returns.csv")
print(f"Saved: re_index_returns.csv")

print(f"\n{'='*60}")
print("SUMMARY FOR PAPER")
print("="*60)
print(f"""
The basis risk analysis compares three RE market proxies:

1. MVREMXTR (equity-based): tracks mining company stocks
2. SMM Equal Weighted: average of 12 actual metal spot prices
3. SMM Usage Weighted: weighted by industrial relevance to our
   company sample (Nd, Dy, Ga, Ge at 20% each; Pr, Tb at 10%)

Key finding:
  MVREMXTR vs SMM EW correlation  = {c_ew:.4f}
  MVREMXTR vs SMM UW correlation  = {c_uw:.4f}

{"HIGH basis risk confirmed" if c_ew < 0.5 else "Moderate basis risk"} —
using MVREMXTR as RE proxy introduces measurement error
because mining equity returns reflect firm-specific factors
beyond commodity price movements.

We run our full GARCH-QQR pipeline on all three indices
and compare results to quantify the impact of index choice
on company-level sensitivity estimates.
""")

print("RE index builder complete — ready to run GARCH-QQR on SMM indices")
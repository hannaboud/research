"""
Markov Switching Regression
============================
Identifies hidden market regimes in the MVREMXTR residuals.
Two regimes: normal periods and crisis periods.
These regimes are then used to run QQR separately within each regime.

HOW TO RUN:
    python3 markov_switching.py

REQUIRES:
    garch_residuals.csv

OUTPUT:
    markov_regimes.csv      — regime classification for each day
    markov_params.csv       — model parameters
    charts/markov/          — regime visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("charts/markov", exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────

INPUT_FILE   = "garch_residuals.csv"
REGIMES_FILE = "markov_regimes.csv"
PARAMS_FILE  = "markov_params.csv"
N_REGIMES    = 3    # low vs normal vs crisis
CRISIS_REGIME = None  # will be identified automatically

# ── LOAD DATA ──────────────────────────────────────────────────

print("="*60)
print("Markov Switching — Regime Identification")
print("="*60)

print("\nLoading GARCH residuals...")
residuals = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
print(f"  Shape: {residuals.shape}")

# Use MVREMXTR residuals for regime identification
# We want to find regimes in the RE market itself
re_resid_daily = residuals[".MVREMXTR"].dropna()
print(f"  MVREMXTR daily residuals: {len(re_resid_daily)} observations")

# Aggregate to monthly — use mean of daily residuals within each month
# This smooths out day-to-day noise and focuses on sustained RE movements
re_resid = re_resid_daily.resample("W").mean().dropna()
print(f"  MVREMXTR weekly residuals: {len(re_resid)} observations")
print(f"  Date range: {re_resid.index[0].date()} to {re_resid.index[-1].date()}")

# ── FIT MARKOV SWITCHING MODEL ─────────────────────────────────

print("\nFitting Markov Switching model...")
print(f"  Number of regimes: {N_REGIMES}")
print(f"  Switching variance: YES (different volatility per regime)")

model = MarkovRegression(
    re_resid,
    k_regimes=N_REGIMES,
    trend="c",              # constant mean per regime
    switching_variance=True # different variance per regime
)

result = model.fit(
    search_reps=20,         # multiple starting points for robustness
    search_scale=1.0,
    disp=False
)

print(f"  Converged: YES")
print(f"  Log-likelihood: {result.llf:.4f}")
print(f"  AIC: {result.aic:.4f}")
print(f"  BIC: {result.bic:.4f}")

# ── IDENTIFY WHICH REGIME IS CRISIS ───────────────────────────

# ── IDENTIFY WHICH REGIME IS WHICH ────────────────────────────

print("\nIdentifying regimes by variance...")

means     = []
variances = []

for i in range(N_REGIMES):
    mean_key = f"const[{i}]"
    var_key  = f"sigma2[{i}]"
    m = result.params.get(mean_key, 0)
    v = result.params.get(var_key, 1)
    means.append(m)
    variances.append(v)
    print(f"  Regime {i}: mean={m:.4f}, variance={v:.4f}, "
          f"std={np.sqrt(v):.4f}")

# Sort regimes by variance
# Lowest variance  = normal
# Middle variance  = stress
# Highest variance = crisis
sorted_by_var = np.argsort(variances)
normal_regime = int(sorted_by_var[0])
stress_regime = int(sorted_by_var[1])
crisis_regime = int(sorted_by_var[2])

print(f"\n  Normal regime = Regime {normal_regime} "
      f"(variance={variances[normal_regime]:.4f})")
print(f"  Stress regime = Regime {stress_regime} "
      f"(variance={variances[stress_regime]:.4f})")
print(f"  Crisis regime = Regime {crisis_regime} "
      f"(variance={variances[crisis_regime]:.4f})")

# ── EXTRACT REGIME PROBABILITIES ───────────────────────────────

print("\nExtracting regime probabilities...")

# Smoothed probabilities — probability of being in each regime
# at each point in time, using all available information
smoothed_probs = result.smoothed_marginal_probabilities

crisis_prob = smoothed_probs[crisis_regime]
normal_prob = smoothed_probs[normal_regime]

# Classify each day into a regime
# Using 0.5 threshold — if prob > 0.5, assign to that regime
regime_classification = (crisis_prob > 0.5).astype(int)
regime_map = {
    normal_regime: "normal",
    stress_regime: "stress",
    crisis_regime: "crisis"
}

# Get most likely regime for each day
regime_classification = smoothed_probs.values.argmax(axis=1)
regime_labels = pd.Series(
    [regime_map[r] for r in regime_classification],
    index=re_resid.index
)

# Crisis probability = probability of being in crisis regime
crisis_prob = smoothed_probs[crisis_regime]
normal_prob = smoothed_probs[normal_regime]
stress_prob = smoothed_probs[stress_regime]

regimes_df = pd.DataFrame({
    "crisis_prob"  : crisis_prob,
    "stress_prob"  : stress_prob,
    "normal_prob"  : normal_prob,
    "regime"       : regime_labels,
    "re_residual"  : re_resid,
}, index=re_resid.index)

# Summary statistics
n_crisis = (regime_labels == "crisis").sum()
n_stress = (regime_labels == "stress").sum()
n_normal = (regime_labels == "normal").sum()
pct_crisis = n_crisis / len(regime_labels) * 100
pct_stress = n_stress / len(regime_labels) * 100
pct_normal = n_normal / len(regime_labels) * 100

print(f"\n  Total days     : {len(regime_labels)}")
print(f"  Crisis days    : {n_crisis} ({pct_crisis:.1f}%)")
print(f"  Stress days    : {n_stress} ({pct_stress:.1f}%)")
print(f"  Normal days    : {n_normal} ({pct_normal:.1f}%)")

# ── TRANSITION PROBABILITIES ───────────────────────────────────

print("\nTransition probabilities:")
print("  (probability of staying in or switching between regimes)")

trans_matrix = result.regime_transition

# Handle different shapes of transition matrix
if trans_matrix.ndim == 3:
    trans = trans_matrix[:, :, 0]
else:
    trans = trans_matrix

p_stay_normal  = trans[normal_regime, normal_regime]
p_go_crisis    = trans[crisis_regime, normal_regime]
p_stay_crisis  = trans[crisis_regime, crisis_regime]
p_go_normal    = trans[normal_regime, crisis_regime]

print(f"\n  From Normal  → Stay Normal : {float(p_stay_normal):.4f}")
print(f"  From Normal  → Go Crisis   : {float(p_go_crisis):.4f}")
print(f"  From Crisis  → Stay Crisis : {float(p_stay_crisis):.4f}")
print(f"  From Crisis  → Go Normal   : {float(p_go_normal):.4f}")

dur_normal = 1 / float(p_go_crisis) if float(p_go_crisis) > 0 else 999
dur_crisis = 1 / float(p_go_normal) if float(p_go_normal) > 0 else 999
print(f"\n  Expected duration normal regime : {dur_normal:.1f} days")
print(f"  Expected duration crisis regime : {dur_crisis:.1f} days")

# ── IDENTIFY CRISIS EPISODES ───────────────────────────────────

print("\nCrisis episodes identified:")

in_crisis = False
crisis_episodes = []
start_date = None

for date, regime in regime_labels.items():
    if regime == "crisis" and not in_crisis:
        in_crisis  = True
        start_date = date
    elif regime == "normal" and in_crisis:
        in_crisis = False
        crisis_episodes.append((start_date, date))

if in_crisis:
    crisis_episodes.append((start_date, regime_labels.index[-1]))

for i, (start, end) in enumerate(crisis_episodes):
    duration = (end - start).days
    if duration > 14:  # only show episodes longer than 5 days
        print(f"  Episode {i+1}: {start.date()} to {end.date()} "
              f"({duration} days)")

# ── CHARTS ────────────────────────────────────────────────────

print("\nBuilding charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.patch.set_facecolor("#0a0a0a")

# Chart 1 — MVREMXTR residuals with regime coloring
ax1 = axes[0]
ax1.set_facecolor("#0a0a0a")

colors = regimes_df["regime"].map(
    {"crisis": "#FF6B6B", "stress": "#F7DC6F", "normal": "#4ECDC4"}
)

ax1.bar(
    regimes_df.index,
    regimes_df["re_residual"],
    color=colors,
    width=1,
    alpha=0.7
)
ax1.axhline(0, color="#444444", linewidth=0.5)
ax1.set_ylabel("MVREMXTR Std Residual",
               color="#888888", fontsize=9)
ax1.set_title(
    "MVREMXTR Standardized Residuals — Crisis vs Normal Regimes\n"
    "Red = Crisis regime | Teal = Normal regime",
    color="#ffffff", fontsize=11, pad=10
)
ax1.tick_params(colors="#888888")
ax1.spines["bottom"].set_color("#333333")
ax1.spines["left"].set_color("#333333")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)

# Chart 2 — Crisis probability over time
ax2 = axes[1]
ax2.set_facecolor("#0a0a0a")
ax2.plot(
    regimes_df.index,
    regimes_df["crisis_prob"],
    color="#FF6B6B",
    linewidth=0.8,
    alpha=0.9
)
ax2.axhline(0.5, color="#ffffff", linewidth=0.8,
            linestyle="--", alpha=0.5, label="50% threshold")
ax2.fill_between(
    regimes_df.index,
    regimes_df["crisis_prob"],
    0.5,
    where=regimes_df["crisis_prob"] > 0.5,
    alpha=0.3,
    color="#FF6B6B"
)
ax2.set_ylabel("Crisis Probability",
               color="#888888", fontsize=9)
ax2.set_title(
    "Smoothed Crisis Regime Probability",
    color="#ffffff", fontsize=11, pad=10
)
ax2.set_ylim(0, 1)
ax2.tick_params(colors="#888888")
ax2.spines["bottom"].set_color("#333333")
ax2.spines["left"].set_color("#333333")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
ax2.legend(facecolor="#111111", edgecolor="#333333",
           labelcolor="#aaaaaa", fontsize=8)

# Chart 3 — Regime distribution over time
ax3 = axes[2]
ax3.set_facecolor("#0a0a0a")

# Rolling 60-day crisis probability
rolling_crisis = regimes_df["crisis_prob"].rolling(60).mean()
ax3.plot(
    rolling_crisis.index,
    rolling_crisis.values,
    color="#FF6B6B",
    linewidth=1.2,
    label="60-day rolling crisis prob"
)
ax3.axhline(0.5, color="#ffffff", linewidth=0.8,
            linestyle="--", alpha=0.5)
ax3.set_ylabel("Rolling Crisis Probability",
               color="#888888", fontsize=9)
ax3.set_xlabel("Date", color="#888888", fontsize=9)
ax3.set_title(
    "60-Day Rolling Average Crisis Probability",
    color="#ffffff", fontsize=11, pad=10
)
ax3.set_ylim(0, 1)
ax3.tick_params(colors="#888888")
ax3.spines["bottom"].set_color("#333333")
ax3.spines["left"].set_color("#333333")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
ax3.legend(facecolor="#111111", edgecolor="#333333",
           labelcolor="#aaaaaa", fontsize=8)

plt.tight_layout()
path = "charts/markov/regime_identification.png"
plt.savefig(path, dpi=150, bbox_inches="tight",
            facecolor="#0a0a0a")
plt.show()
print(f"  Saved: {path}")
plt.close()

# ── SAVE ──────────────────────────────────────────────────────

regimes_df.to_csv(REGIMES_FILE)

params_summary = pd.DataFrame({
    "regime"    : ["normal", "stress", "crisis"],
    "mean"      : [means[normal_regime], means[stress_regime], 
                   means[crisis_regime]],
    "variance"  : [variances[normal_regime], variances[stress_regime],
                   variances[crisis_regime]],
    "std"       : [np.sqrt(variances[normal_regime]),
                   np.sqrt(variances[stress_regime]),
                   np.sqrt(variances[crisis_regime])],
    "n_days"    : [n_normal, n_stress, n_crisis],
    "pct_days"  : [pct_normal, pct_stress, pct_crisis],
})
params_summary.to_csv(PARAMS_FILE, index=False)

print(f"\nSaved: {REGIMES_FILE}")
print(f"Saved: {PARAMS_FILE}")
print("\nMarkov Switching complete — ready for QQR")
"""
GJR-GARCH Filter
================
Fits a GJR-GARCH(1,1) model to each return series to remove
volatility clustering before QQR estimation.

WHY GJR-GARCH:
    Standard returns exhibit volatility clustering — calm periods
    cluster with calm periods, volatile periods cluster together.
    This contaminates QQR extreme quantiles with general market
    volatility rather than genuine RE price sensitivity.

    GJR-GARCH extends standard GARCH by capturing the leverage
    effect — negative shocks increase future volatility MORE than
    positive shocks of the same magnitude. This asymmetry is
    well documented in equity markets.

MODEL:
    Return:    r_t = mu + epsilon_t
    Variance:  sigma²_t = omega
                        + alpha  * epsilon²_{t-1}
                        + gamma  * epsilon²_{t-1} * I(epsilon_{t-1} < 0)
                        + beta   * sigma²_{t-1}

    Where I(epsilon < 0) = 1 if previous shock was negative, 0 otherwise
    gamma captures the extra volatility from negative shocks (leverage effect)

OUTPUT:
    Standardized residual = epsilon_t / sigma_t
    These are the cleaned returns — on a comparable scale across
    all time periods regardless of market volatility environment.

HOW TO RUN:
    python3 garch_filter.py

REQUIRES:
    log_returns.csv    — from data_prep.py

OUTPUT FILES:
    garch_residuals.csv   — standardized residuals (cleaned returns)
    garch_params.csv      — fitted parameters for each series
    garch_volatility.csv  — conditional volatility sigma_t per day
"""

import pandas as pd
import numpy as np
from arch import arch_model
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("charts/garch", exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────

# Scale returns by 100 for numerical stability in GARCH estimation
# (GARCH works better with returns expressed as percentages)
SCALE = 100

# ── LOAD DATA ──────────────────────────────────────────────────

print("="*60)
print("GJR-GARCH Filter")
print("="*60)

print("\nLoading log returns...")
log_returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
print(f"  Shape: {log_returns.shape}")
print(f"  Series: {list(log_returns.columns)}")

# ── FIT GARCH MODELS ───────────────────────────────────────────

print("\nFitting GJR-GARCH(1,1) models...")
print(f"  {'Series':<15} {'omega':>8} {'alpha':>8} {'gamma':>8} "
      f"{'beta':>8} {'persist':>8} {'LB p-val':>10}")
print("  " + "-"*70)

residuals   = {}
volatility  = {}
params_list = []

for col in log_returns.columns:

    series = log_returns[col].dropna() * SCALE

    try:
        # Fit GJR-GARCH(1,1) with normal distribution
        # p=1, q=1 means one lag for both ARCH and GARCH terms
        # o=1 means one asymmetric (leverage) term
        model = arch_model(
            series,
            vol="Garch",
            p=1,          # ARCH lag
            o=1,          # Asymmetric/leverage term (makes it GJR)
            q=1,          # GARCH lag
            dist="normal",
            mean="Constant"
        )

        result = model.fit(disp="off", show_warning=False)

        # Extract parameters
        omega = result.params.get("omega", np.nan)
        alpha = result.params.get("alpha[1]", np.nan)
        gamma = result.params.get("gamma[1]", np.nan)
        beta  = result.params.get("beta[1]", np.nan)

        # Persistence = alpha + gamma/2 + beta
        # If persistence > 1, volatility is explosive (bad)
        # Should be < 1 for stable model
        persistence = alpha + gamma/2 + beta

        # Extract standardized residuals
        std_resid = result.resid / result.conditional_volatility

        # Extract conditional volatility (unscaled)
        cond_vol  = result.conditional_volatility / SCALE

        # Ljung-Box test on squared residuals
        # Tests whether volatility clustering has been removed
        # p-value > 0.05 means no remaining clustering — good
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
        lb_pval   = lb_result["lb_pvalue"].iloc[-1]

        residuals[col]  = std_resid
        volatility[col] = cond_vol

        params_list.append({
            "ticker"     : col,
            "omega"      : round(omega, 6),
            "alpha"      : round(alpha, 4),
            "gamma"      : round(gamma, 4),
            "beta"       : round(beta, 4),
            "persistence": round(persistence, 4),
            "lb_pval"    : round(lb_pval, 4),
            "lb_clean"   : "YES" if lb_pval > 0.05 else "NO"
        })

        print(f"  {col:<15} {omega:>8.4f} {alpha:>8.4f} {gamma:>8.4f} "
              f"{beta:>8.4f} {persistence:>8.4f} {lb_pval:>10.4f}")

    except Exception as e:
        print(f"  {col:<15} ERROR: {e}")
        continue

# ── BUILD OUTPUT DATAFRAMES ────────────────────────────────────

residuals_df  = pd.DataFrame(residuals, index=log_returns.index)
volatility_df = pd.DataFrame(volatility, index=log_returns.index)
params_df     = pd.DataFrame(params_list)

# ── QUALITY CHECK ──────────────────────────────────────────────

print("\n" + "="*60)
print("QUALITY CHECK — Standardized Residuals")
print("="*60)
print("\nIf GARCH worked properly:")
print("  Mean should be close to 0")
print("  Std should be close to 1")
print("  Kurtosis should be closer to 3 (was 5-17 before)")
print("  Skewness should be closer to 0")
print()
print(f"  {'Series':<15} {'Mean':>8} {'Std':>8} "
      f"{'Skew':>8} {'Kurt':>8} {'LB Clean':>10}")
print("  " + "-"*60)

for col in residuals_df.columns:
    s    = residuals_df[col].dropna()
    mean = s.mean()
    std  = s.std()
    skew = s.skew()
    kurt = s.kurtosis()
    lb   = params_df[params_df["ticker"]==col]["lb_clean"].values[0] \
           if len(params_df[params_df["ticker"]==col]) > 0 else "N/A"

    print(f"  {col:<15} {mean:>8.4f} {std:>8.4f} "
          f"{skew:>8.4f} {kurt:>8.4f} {lb:>10}")

# ── GARCH PARAMETERS SUMMARY ───────────────────────────────────

print("\n" + "="*60)
print("GARCH PARAMETERS SUMMARY")
print("="*60)
print(params_df.to_string(index=False))

# Persistence warning
high_persist = params_df[params_df["persistence"] > 0.99]
if len(high_persist) > 0:
    print(f"\nWARNING: High persistence (>0.99) in:")
    print(high_persist[["ticker","persistence"]].to_string(index=False))
    print("These series have very long-lasting volatility shocks")

# Leverage effect summary
print(f"\nLeverage effect (gamma > 0) — negative shocks increase volatility more:")
leverage = params_df[params_df["gamma"] > 0]
print(f"  {len(leverage)}/{len(params_df)} series show leverage effect")

# LB test summary
clean = params_df[params_df["lb_clean"] == "YES"]
print(f"\nLjung-Box test — volatility clustering removed:")
print(f"  {len(clean)}/{len(params_df)} series pass (no remaining clustering)")

# ── PLOT CONDITIONAL VOLATILITY ────────────────────────────────

print("\nBuilding volatility charts...")

# Plot volatility for key companies
key_tickers = ["NVDA.O", "AAPL.O", "RTX.N", "FCX.N", ".MVREMXTR"]
key_tickers = [t for t in key_tickers if t in volatility_df.columns]

fig, axes = plt.subplots(len(key_tickers), 1,
                          figsize=(14, 3*len(key_tickers)))
fig.patch.set_facecolor("#0a0a0a")

for i, ticker in enumerate(key_tickers):
    ax = axes[i]
    ax.set_facecolor("#0a0a0a")

    vol = volatility_df[ticker].dropna() * 100  # as percentage

    ax.fill_between(vol.index, vol.values,
                    alpha=0.4, color="#4ECDC4")
    ax.plot(vol.index, vol.values,
            color="#4ECDC4", linewidth=0.8)

    ax.set_ylabel("Daily Vol %", color="#888", fontsize=8)
    ax.set_title(f"{ticker} — Conditional Volatility",
                 color="#ffffff", fontsize=10)
    ax.tick_params(colors="#888888", labelsize=7)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    # Mark COVID crash and other key events
    events = {
        "COVID\nCrash": "2020-03-20",
        "Russia\nUkraine": "2022-02-24",
        "RE\nRestrictions": "2023-07-03",
    }
    for label, date in events.items():
        try:
            ax.axvline(pd.Timestamp(date),
                      color="#FF6B6B", linewidth=0.8,
                      linestyle="--", alpha=0.6)
        except:
            pass

plt.tight_layout()
path = "charts/garch/conditional_volatility.png"
plt.savefig(path, dpi=150, bbox_inches="tight",
            facecolor="#0a0a0a")
plt.close()
print(f"  Saved: {path}")

# ── SAVE OUTPUTS ───────────────────────────────────────────────

residuals_df.to_csv("garch_residuals.csv")
volatility_df.to_csv("garch_volatility.csv")
params_df.to_csv("garch_params.csv", index=False)

print(f"\nSaved: garch_residuals.csv   — standardized residuals")
print(f"Saved: garch_volatility.csv  — conditional volatility")
print(f"Saved: garch_params.csv      — model parameters")
print(f"\nShape of residuals: {residuals_df.shape}")
print(f"\nGARCH filtering complete — ready for QQR")
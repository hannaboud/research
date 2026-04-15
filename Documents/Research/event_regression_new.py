"""
Event-Based Regression — Dual Index (MVREMXTR + CSI 930598)
=============================================================
Runs event regression using both RE indices simultaneously
and produces side by side comparison of results.
 
PRIMARY INDEX   : CSI_RE  (CSI Rare Earth Industry Index)
ROBUSTNESS CHECK: .MVREMXTR (MVIS Global RE Index)
 
HOW TO RUN:
    python3 event_regression_new.py
 
REQUIRES:
    garch_residuals.csv
    market_caps.csv
    event_study_sample.csv
 
OUTPUT:
    shock_days_CSI.csv
    shock_days_MVREMXTR.csv
    event_delta_results_CSI.csv
    event_delta_results_MVREMXTR.csv
    event_delta_comparison.csv
    charts/event/
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
 
os.makedirs("charts/event", exist_ok=True)
 
ROLLING_WINDOW    = 60
SHOCK_THRESHOLD   = 2.0
WINDOW_EXPANSION  = 2
P_VALUE_THRESHOLD = 0.05
MIN_OBSERVATIONS  = 30
 
RE_INDICES = {
    "CSI"     : "CSI_RE",
    "MVREMXTR": ".MVREMXTR",
}
 
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
 
print("="*60)
print("Event Regression — Dual Index (CSI + MVREMXTR)")
print("="*60)
 
residuals = pd.read_csv(
    "garch_residuals.csv", index_col=0, parse_dates=True
)
mkt_resid       = residuals["MARKET"]
stock_residuals = residuals[TICKERS]
 
try:
    mcaps = pd.read_csv("market_caps.csv").rename(columns={
        "Instrument":"ticker",
        "Company Market Cap":"market_cap_usd"
    })[["ticker","market_cap_usd"]].copy()
    mcaps["market_cap_usd"] = pd.to_numeric(
        mcaps["market_cap_usd"], errors="coerce"
    )
    sample = pd.read_csv("event_study_sample.csv")
    sample["ticker"] = sample["Symbol"].apply(
        lambda x: f"{x}.O" if x in NASDAQ else f"{x}.N"
    )
    has_meta = True
except FileNotFoundError:
    has_meta = False
 
all_results = {}
 
for index_name, index_col in RE_INDICES.items():
 
    print(f"\n{'='*60}")
    print(f"RE Index: {index_name} ({index_col})")
    print("="*60)
 
    re_resid = residuals[index_col].dropna()
 
    rolling_mean      = re_resid.shift(1).rolling(ROLLING_WINDOW).mean()
    rolling_std       = re_resid.shift(1).rolling(ROLLING_WINDOW).std()
    z_score           = (re_resid - rolling_mean) / rolling_std
    is_shock          = z_score.abs() > SHOCK_THRESHOLD
    is_shock_expanded = is_shock.copy()
 
    for i in range(1, WINDOW_EXPANSION + 1):
        is_shock_expanded = (is_shock_expanded |
                             is_shock.shift(i) |
                             is_shock.shift(-i))
    is_shock_expanded = is_shock_expanded.fillna(False)
 
    shock_df = pd.DataFrame({
        "re_residual" : re_resid,
        "z_score"     : z_score,
        "is_shock"    : is_shock,
        "is_shock_exp": is_shock_expanded,
        "direction"   : np.where(re_resid > 0, "up", "down"),
    })
 
    n_shock     = is_shock.sum()
    n_shock_exp = is_shock_expanded.sum()
    n_total     = len(is_shock_expanded)
 
    print(f"  Core shock days     : {n_shock} ({n_shock/n_total*100:.1f}%)")
    print(f"  Expanded shock days : {n_shock_exp} ({n_shock_exp/n_total*100:.1f}%)")
    print(f"  Normal days         : {n_total-n_shock_exp} ({(n_total-n_shock_exp)/n_total*100:.1f}%)")
 
    shock_df.to_csv(f"shock_days_{index_name}.csv")
 
    shock_idx = is_shock_expanded[is_shock_expanded].index
    re_shock  = re_resid.loc[shock_idx]
    mkt_shock = mkt_resid.loc[shock_idx]
 
    results_list = []
 
    for ticker in TICKERS:
        if ticker not in stock_residuals.columns:
            continue
        stock = stock_residuals[ticker].loc[
            stock_residuals[ticker].index.intersection(shock_idx)
        ]
        df = pd.concat([stock, re_shock, mkt_shock], axis=1).dropna()
        df.columns = ["stock","RE","market"]
 
        if len(df) < MIN_OBSERVATIONS:
            continue
 
        try:
            X     = sm.add_constant(df[["RE","market"]])
            model = sm.OLS(df["stock"], X).fit()
            results_list.append({
                "ticker"     : ticker,
                "delta_RE"   : round(model.params["RE"], 6),
                "p_value"    : round(model.pvalues["RE"], 4),
                "significant": model.pvalues["RE"] < P_VALUE_THRESHOLD,
                "r2"         : round(model.rsquared, 4),
                "n_obs"      : len(df),
            })
        except Exception:
            continue
 
    results_df = pd.DataFrame(results_list)
    sig = results_df["significant"].sum()
 
    print(f"\n  Companies significant : {sig}/{len(results_df)} ({sig/len(results_df)*100:.0f}%)")
    print(f"  Average R²            : {results_df['r2'].mean():.4f}")
    print(f"  Negative delta        : {(results_df['delta_RE']<0).sum()} ({(results_df['delta_RE']<0).mean():.0f}%)")
 
    if has_meta:
        results_df = results_df.merge(mcaps, on="ticker", how="left")
        results_df = results_df.merge(
            sample[["ticker","Security","tier","GICS Sub-Industry"]],
            on="ticker", how="left"
        )
        re_vol = re_resid.std() * np.sqrt(252)
        results_df["dollar_exposure_usd"] = (
            results_df["delta_RE"].abs() * re_vol * results_df["market_cap_usd"]
        )
        results_df["exposure_billions"] = results_df["dollar_exposure_usd"] / 1e9
 
    print(f"\n  {'Ticker':<10} {'Company':<25} {'Delta':>8} {'Sig':>4} {'R²':>6}")
    print("  " + "-"*58)
    for _, row in results_df.sort_values("delta_RE").iterrows():
        sig_mark = " *" if row["significant"] else "  "
        name = str(row.get("Security", row["ticker"]))[:24]
        print(f"  {row['ticker']:<10} {name:<25} "
              f"{row['delta_RE']:>8.4f}{sig_mark} {row['r2']:>6.4f}")
 
    results_df.to_csv(f"event_delta_results_{index_name}.csv", index=False)
    all_results[index_name] = results_df
 
# ── SIDE BY SIDE COMPARISON ────────────────────────────────────
 
print(f"\n{'='*60}")
print("SIDE BY SIDE COMPARISON — CSI vs MVREMXTR")
print("="*60)
 
if "CSI" in all_results and "MVREMXTR" in all_results:
    csi_res = all_results["CSI"][["ticker","delta_RE","significant","r2"]]\
        .rename(columns={"delta_RE":"delta_CSI",
                         "significant":"sig_CSI","r2":"r2_CSI"})
    mv_res = all_results["MVREMXTR"][["ticker","delta_RE","significant","r2"]]\
        .rename(columns={"delta_RE":"delta_MVREMXTR",
                         "significant":"sig_MVREMXTR","r2":"r2_MVREMXTR"})
 
    comparison = csi_res.merge(mv_res, on="ticker", how="outer")
 
    if has_meta:
        comparison = comparison.merge(
            sample[["ticker","Security","tier"]], on="ticker", how="left"
        )
 
    comparison["same_sign"] = (
        np.sign(comparison["delta_CSI"]) ==
        np.sign(comparison["delta_MVREMXTR"])
    )
 
    print(f"\n{'Ticker':<10} {'Company':<22} {'δ CSI':>8} {'δ MVREMXTR':>11} {'Sign OK':>8} {'R²CSI':>7} {'R²MV':>7}")
    print("-"*78)
 
    for _, row in comparison.sort_values("delta_CSI", na_position="last").iterrows():
        name  = str(row.get("Security", row["ticker"]))[:21]
        agree = "✓" if row["same_sign"] else "✗"
        dc    = f"{row['delta_CSI']:.4f}" if pd.notna(row["delta_CSI"]) else "  nan"
        dm    = f"{row['delta_MVREMXTR']:.4f}" if pd.notna(row["delta_MVREMXTR"]) else "  nan"
        rc    = f"{row['r2_CSI']:.4f}" if pd.notna(row["r2_CSI"]) else " nan"
        rm    = f"{row['r2_MVREMXTR']:.4f}" if pd.notna(row["r2_MVREMXTR"]) else " nan"
        print(f"{row['ticker']:<10} {name:<22} {dc:>8} {dm:>11} {agree:>8} {rc:>7} {rm:>7}")
 
    n_agree = comparison["same_sign"].sum()
    n_total = comparison["same_sign"].notna().sum()
    print(f"\nSign agreement  : {n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)")
    print(f"Avg R² CSI      : {comparison['r2_CSI'].mean():.4f}")
    print(f"Avg R² MVREMXTR : {comparison['r2_MVREMXTR'].mean():.4f}")
 
    comparison.to_csv("event_delta_comparison.csv", index=False)
    print(f"Saved: event_delta_comparison.csv")
 
    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0a0a0a")
 
    for ax, (name, res) in zip(axes, all_results.items()):
        ax.set_facecolor("#0a0a0a")
        sorted_df = res.sort_values("delta_RE")
        labels = sorted_df["Security"].fillna(sorted_df["ticker"]).astype(str)
        colors = ["#FF6B6B" if v < 0 else "#4ECDC4" for v in sorted_df["delta_RE"]]
        ax.barh(labels, sorted_df["delta_RE"], color=colors,
                alpha=0.85, edgecolor="#0a0a0a", linewidth=0.5)
        ax.axvline(0, color="#444444", linewidth=1.0)
        ax.set_xlabel("Delta RE", color="#888888", fontsize=9)
        ax.set_title(
            f"RE Index: {name}\n"
            f"Sig: {sorted_df['significant'].sum()}/29 | "
            f"Avg R²: {sorted_df['r2'].mean():.3f}",
            color="#ffffff", fontsize=10, pad=10
        )
        ax.tick_params(colors="#888888", labelsize=7)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)
 
    plt.suptitle(
        "Event Regression Delta — CSI 930598 vs MVREMXTR\n"
        "Red = hurt by RE rises | Teal = benefits",
        color="#ffffff", fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig("charts/event/delta_comparison_dual.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print("Saved: charts/event/delta_comparison_dual.png")
 
print("\nEvent regression complete — ready for QQR")
 
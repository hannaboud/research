"""
Quantile-on-Quantile Regression — Dual Index (MVREMXTR + CSI)
==============================================================
Runs QQR using both RE indices simultaneously and produces
side by side surface comparison.

PRIMARY INDEX   : MVREMXTR — better statistical identification
ROBUSTNESS CHECK: CSI_RE   — better correlation with actual RE prices

HOW TO RUN:
    python3 qqr.py

REQUIRES:
    garch_residuals.csv
    shock_days_MVREMXTR.csv
    shock_days_CSI.csv

OUTPUT:
    qqr_results/            — surface CSVs per company per index
    charts/qqr/             — 3D surface plots
    qqr_summary_MVREMXTR.csv
    qqr_summary_CSI.csv
    qqr_comparison.csv      — side by side key cells
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.regression.quantile_regression import QuantReg
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("qqr_results", exist_ok=True)
os.makedirs("charts/qqr",  exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────

QUANTILES   = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
BANDWIDTH   = 0.15
MIN_EFF_OBS = 25

TICKERS = [
    "NVDA.O","AAPL.O","GOOG.O","GOOGL.O","AVGO.O",
    "TSLA.O","MU.O","AMD.O","AMAT.O","LRCX.O",
    "INTC.O","KLAC.O","TXN.O","ADI.O","QCOM.O",
    "RTX.N","BA.N","LMT.N","NOC.N","GD.N",
    "CAT.N","GE.N","ETN.N","PH.N","HWM.N",
    "FCX.N","WDC.O","STX.O","HON.O",
]

# Both RE indices
RE_INDICES = {
    "MVREMXTR": ".MVREMXTR",
    "CSI"     : "CSI_RE",
}

NASDAQ = [
    "NVDA","AAPL","GOOG","GOOGL","AVGO","TSLA","MU","AMD",
    "AMAT","LRCX","INTC","KLAC","TXN","ADI","QCOM",
    "WDC","STX","HON",
]

# ── LOAD DATA ──────────────────────────────────────────────────

print("="*60)
print("QQR — Dual Index (MVREMXTR + CSI)")
print("Solver: statsmodels QuantReg")
print("="*60)

residuals = pd.read_csv(
    "garch_residuals.csv", index_col=0, parse_dates=True
)
print(f"Residuals shape: {residuals.shape}")

# Load shock days for both indices
shock_days = {}
normal_days = {}

for name in RE_INDICES:
    sd = pd.read_csv(
        f"shock_days_{name}.csv", index_col=0, parse_dates=True
    )
    shock_days[name]  = sd[sd["is_shock_exp"]].index
    normal_days[name] = sd[~sd["is_shock_exp"]].index
    print(f"{name} — shock: {len(shock_days[name])} | "
          f"normal: {len(normal_days[name])}")

# Company info
try:
    sample = pd.read_csv("event_study_sample.csv")
    sample["ticker"] = sample["Symbol"].apply(
        lambda x: f"{x}.O" if x in NASDAQ else f"{x}.N"
    )
    has_meta = True
except FileNotFoundError:
    has_meta = False

# ── QQR CORE FUNCTIONS ─────────────────────────────────────────

def qqr_beta(y_vals, x_vals, tau, theta, bandwidth=BANDWIDTH):
    mask = np.isfinite(y_vals) & np.isfinite(x_vals)
    y    = y_vals[mask]
    x    = x_vals[mask]

    if len(y) < MIN_EFF_OBS * 2:
        return np.nan

    x_theta = np.quantile(x, theta)
    x_std   = x.std()
    if x_std < 1e-8:
        return np.nan

    u       = (x - x_theta) / (x_std * bandwidth)
    weights = np.exp(-0.5 * u ** 2)
    w_sum   = weights.sum()
    if w_sum < 1e-10:
        return np.nan
    weights = weights / w_sum

    n_eff = 1.0 / (weights ** 2).sum()
    if n_eff < MIN_EFF_OBS:
        return np.nan

    w_scaled = np.round(weights * 1000).astype(int)
    w_scaled = np.maximum(w_scaled, 0)
    y_exp    = np.repeat(y, w_scaled)
    x_exp    = np.repeat(x, w_scaled)

    if len(y_exp) < MIN_EFF_OBS:
        return np.nan

    X_mat = np.column_stack([np.ones(len(x_exp)), x_exp])

    try:
        result = QuantReg(y_exp, X_mat).fit(
            q=tau, max_iter=2000, p_tol=1e-6, vcov="robust"
        )
        beta = result.params[1]
        if np.abs(beta) > 15:
            return np.nan
        return float(beta)
    except Exception:
        return np.nan


def compute_surface(y_series, x_series):
    y_vals  = y_series.values
    x_vals  = x_series.values
    n       = len(QUANTILES)
    surface = np.full((n, n), np.nan)
    for i, tau in enumerate(QUANTILES):
        for j, theta in enumerate(QUANTILES):
            surface[i, j] = qqr_beta(y_vals, x_vals, tau, theta)
    return surface


def plot_surfaces(ticker, surfaces_dict):
    """Plot 3D surfaces for both indices side by side."""
    n_panels = len(surfaces_dict) * 3  # full/shock/normal per index
    fig = plt.figure(figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor("#0a0a0a")
    theta_g, tau_g = np.meshgrid(QUANTILES, QUANTILES)

    panel = 1
    for idx_name, (sf, ss, sn) in surfaces_dict.items():
        for title, surf in [
            (f"{idx_name} Full",   sf),
            (f"{idx_name} Shock",  ss),
            (f"{idx_name} Normal", sn),
        ]:
            ax = fig.add_subplot(1, n_panels, panel, projection="3d")
            ax.set_facecolor("#111111")
            sp   = np.where(np.isnan(surf), 0, surf)
            vmax = min(
                float(np.nanpercentile(np.abs(sp[sp!=0]), 95))
                if np.any(sp!=0) else 1, 5
            )
            ax.plot_surface(
                theta_g, tau_g, np.clip(sp, -vmax, vmax),
                cmap="RdYlGn", alpha=0.9, edgecolor="none",
                vmin=-vmax, vmax=vmax
            )
            ax.scatter(
                [QUANTILES[-1]], [QUANTILES[0]], [sp[0,-1]],
                color="#FF0000", s=40, zorder=5
            )
            ax.set_xlabel("θ RE", color="#aaaaaa", fontsize=6, labelpad=2)
            ax.set_ylabel("τ Stock", color="#aaaaaa", fontsize=6, labelpad=2)
            ax.set_zlabel("β", color="#aaaaaa", fontsize=6, labelpad=2)
            ax.set_title(
                f"{title}\nβ(0.05,0.95)={surf[0,-1]:.3f}",
                color="#ffffff", fontsize=7, pad=5
            )
            ax.tick_params(colors="#888888", labelsize=5)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            panel += 1

    plt.suptitle(
        f"QQR — {ticker} | Red dot = worst case β(0.05,0.95)",
        color="#ffffff", fontsize=9, y=1.01
    )
    plt.tight_layout()
    plt.savefig(f"charts/qqr/{ticker}_dual.png",
                dpi=100, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()


# ── RUN QQR FOR EACH INDEX ─────────────────────────────────────

all_summaries = {}

for index_name, index_col in RE_INDICES.items():

    print(f"\n{'='*60}")
    print(f"QQR with RE Index: {index_name} ({index_col})")
    print("="*60)

    if index_col not in residuals.columns:
        print(f"  SKIPPED — {index_col} not in residuals")
        continue

    re_resid = residuals[index_col]
    s_days   = shock_days[index_name]
    n_days   = normal_days[index_name]

    summary_list = []

    for ticker in TICKERS:
        if ticker not in residuals.columns:
            continue

        print(f"  {ticker}...", end=" ", flush=True)

        y = residuals[ticker]
        x = re_resid
        common = y.index.intersection(x.index)
        y = y.loc[common].dropna()
        x = x.loc[common].dropna()
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]

        # Full sample surface
        sf = compute_surface(y, x)
        print("F", end="", flush=True)

        # Shock days surface
        si = common.intersection(s_days)
        ss = compute_surface(y.loc[si], x.loc[si]) \
             if len(si) >= MIN_EFF_OBS * 2 \
             else np.full((len(QUANTILES), len(QUANTILES)), np.nan)
        print("S", end="", flush=True)

        # Normal days surface
        ni = common.intersection(n_days)
        sn = compute_surface(y.loc[ni], x.loc[ni]) \
             if len(ni) >= MIN_EFF_OBS * 2 \
             else np.full((len(QUANTILES), len(QUANTILES)), np.nan)
        print("N", end=" ", flush=True)

        # Save CSVs
        for surf, label in [
            (sf, f"{index_name}_full"),
            (ss, f"{index_name}_shock"),
            (sn, f"{index_name}_normal"),
        ]:
            pd.DataFrame(
                surf,
                index=np.round(QUANTILES,2),
                columns=np.round(QUANTILES,2)
            ).to_csv(f"qqr_results/{ticker}_{label}.csv")

        # Key cells
        b_worst_f  = sf[0, -1]
        b_worst_s  = ss[0, -1]
        b_worst_n  = sn[0, -1]
        b_best_f   = sf[-1, 0]
        b_best_s   = ss[-1, 0]
        b_med_f    = sf[3, 3]
        b_med_s    = ss[3, 3]

        print(f"β(0.05,0.95) full={b_worst_f:.3f} shock={b_worst_s:.3f}")

        summary_list.append({
            "ticker"           : ticker,
            "re_index"         : index_name,
            "beta_worst_full"  : round(b_worst_f, 4),
            "beta_worst_shock" : round(b_worst_s, 4),
            "beta_worst_normal": round(b_worst_n, 4),
            "beta_best_full"   : round(b_best_f,  4),
            "beta_best_shock"  : round(b_best_s,  4),
            "beta_med_full"    : round(b_med_f,   4),
            "beta_med_shock"   : round(b_med_s,   4),
            "avg_full"         : round(np.nanmean(sf), 4),
            "avg_shock"        : round(np.nanmean(ss), 4),
            "n_shock"          : len(si),
            "n_normal"         : len(ni),
        })

        # Store surfaces for plotting
        if ticker not in [d.get("ticker") for d in summary_list[:-1]]:
            pass  # will plot after both indices done

    summary_df = pd.DataFrame(summary_list)

    if has_meta:
        summary_df = summary_df.merge(
            sample[["ticker","Security","tier"]],
            on="ticker", how="left"
        )

    summary_df.to_csv(f"qqr_summary_{index_name}.csv", index=False)
    all_summaries[index_name] = summary_df

    # Print summary table
    print(f"\n{'='*60}")
    print(f"QQR SUMMARY — {index_name}")
    print("="*60)
    print(f"\n{'Ticker':<10} {'Company':<24} "
          f"{'β(0.05,0.95)':<13} {'β(0.05,0.95)':<13} {'β(0.50,0.50)'}")
    print(f"{'':10} {'':24} {'Full':<13} {'Shock':<13} {'Full'}")
    print("-"*72)

    for _, row in summary_df.sort_values(
        "beta_worst_shock", na_position="last"
    ).iterrows():
        name = str(row.get("Security", row["ticker"]))[:23]
        wf = f"{row['beta_worst_full']:.3f}" \
             if pd.notna(row["beta_worst_full"]) else " nan"
        ws = f"{row['beta_worst_shock']:.3f}" \
             if pd.notna(row["beta_worst_shock"]) else " nan"
        wm = f"{row['beta_med_full']:.3f}" \
             if pd.notna(row["beta_med_full"]) else " nan"
        print(f"{row['ticker']:<10} {name:<24} {wf:<13} {ws:<13} {wm}")

    neg_f = (summary_df["beta_worst_full"]  < 0).sum()
    neg_s = (summary_df["beta_worst_shock"] < 0).sum()
    print(f"\nNeg β(0.05,0.95) full  : {neg_f}/29")
    print(f"Neg β(0.05,0.95) shock : {neg_s}/29")

# ── SIDE BY SIDE COMPARISON ────────────────────────────────────

if len(all_summaries) == 2:
    print(f"\n{'='*60}")
    print("CROSS-INDEX COMPARISON — MVREMXTR vs CSI")
    print("="*60)

    mv = all_summaries["MVREMXTR"][
        ["ticker","beta_worst_full","beta_worst_shock","beta_med_full"]
    ].rename(columns={
        "beta_worst_full" : "MV_worst_full",
        "beta_worst_shock": "MV_worst_shock",
        "beta_med_full"   : "MV_med_full",
    })

    cs = all_summaries["CSI"][
        ["ticker","beta_worst_full","beta_worst_shock","beta_med_full"]
    ].rename(columns={
        "beta_worst_full" : "CSI_worst_full",
        "beta_worst_shock": "CSI_worst_shock",
        "beta_med_full"   : "CSI_med_full",
    })

    comp = mv.merge(cs, on="ticker", how="outer")

    if has_meta:
        comp = comp.merge(
            sample[["ticker","Security","tier"]],
            on="ticker", how="left"
        )

    # Sign agreement on worst case shock cell
    comp["sign_agree_shock"] = (
        np.sign(comp["MV_worst_shock"]) ==
        np.sign(comp["CSI_worst_shock"])
    )

    print(f"\n{'Ticker':<10} {'Company':<22} "
          f"{'MV shock':>9} {'CSI shock':>10} {'Agree':>7}")
    print("-"*62)

    for _, row in comp.sort_values(
        "MV_worst_shock", na_position="last"
    ).iterrows():
        name  = str(row.get("Security", row["ticker"]))[:21]
        mv_s  = f"{row['MV_worst_shock']:.3f}" \
                if pd.notna(row["MV_worst_shock"]) else "  nan"
        cs_s  = f"{row['CSI_worst_shock']:.3f}" \
                if pd.notna(row["CSI_worst_shock"]) else "  nan"
        agree = "✓" if row["sign_agree_shock"] else "✗"
        print(f"{row['ticker']:<10} {name:<22} "
              f"{mv_s:>9} {cs_s:>10} {agree:>7}")

    n_agree = comp["sign_agree_shock"].sum()
    n_total = comp["sign_agree_shock"].notna().sum()
    print(f"\nSign agreement β(0.05,0.95) shock: "
          f"{n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)")

    comp.to_csv("qqr_comparison.csv", index=False)
    print("Saved: qqr_comparison.csv")

    # Comparison chart — worst case beta both indices
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0a0a0a")

    for ax, (idx_name, summary) in zip(axes, all_summaries.items()):
        ax.set_facecolor("#0a0a0a")
        sorted_df = summary.sort_values("beta_worst_shock")

        if has_meta:
            labels = sorted_df["Security"].fillna(
                sorted_df["ticker"]
            ).astype(str)
        else:
            labels = sorted_df["ticker"]

        colors = ["#FF6B6B" if pd.notna(v) and v < 0 else "#4ECDC4"
                  for v in sorted_df["beta_worst_shock"]]

        vals = sorted_df["beta_worst_shock"].fillna(0)
        ax.barh(labels, vals, color=colors,
                alpha=0.85, edgecolor="#0a0a0a", linewidth=0.5)
        ax.axvline(0, color="#444444", linewidth=1.0)
        ax.set_xlabel("β(0.05, 0.95) — Worst Case",
                      color="#888888", fontsize=9)
        ax.set_title(
            f"RE Index: {idx_name}\n"
            f"Neg: {(sorted_df['beta_worst_shock']<0).sum()}/29",
            color="#ffffff", fontsize=10, pad=10
        )
        ax.tick_params(colors="#888888", labelsize=7)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    plt.suptitle(
        "QQR Worst Case β(0.05,0.95) — MVREMXTR vs CSI\n"
        "Red = hurt when RE spikes + stock falls | Teal = benefits",
        color="#ffffff", fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig("charts/qqr/worst_case_comparison.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print("Saved: charts/qqr/worst_case_comparison.png")

print(f"\nQQR complete — both indices done")
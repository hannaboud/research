"""
China RE PPI — Monthly Multi-Factor OLS Robustness Check (Ultimate Version)
Includes side-by-side comparison + sign agreement + Excel export
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import lseg.data as ld

warnings.filterwarnings('ignore')

# ========================= CONFIG =========================

P_VALUE_THRESHOLD = 0.05
MIN_OBSERVATIONS = 20

TICKERS = [
    "NVDA.O", "AAPL.O", "GOOG.O", "GOOGL.O", "AVGO.O", "TSLA.O", "MU.O", "AMD.O",
    "AMAT.O", "LRCX.O", "INTC.O", "KLAC.O", "TXN.O", "ADI.O", "QCOM.O",
    "RTX.N", "BA.N", "LMT.N", "NOC.N", "GD.N", "CAT.N", "GE.N", "ETN.N",
    "PH.N", "HWM.N", "FCX.N", "WDC.O", "STX.O", "HON.O",
]

MACRO_FACTORS = {
    "VIX_PROXY": "VXX",
    "US10Y": "US10YT=RR",
    "DXY": ".DXY",
    "WTI": "CLc1",
    "COPPER": "HGc1",
}

RE_PPI_RIC = "aCNCNHVGWM"
START_DATE = "2016-01-01"
END_DATE = "2026-04-01"

CHARTS_DIR = Path("charts/ppi")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================= DATA FETCH =======================

ld.open_session()

# PPI
ppi_raw = ld.get_history([RE_PPI_RIC], fields=None, interval="monthly", start=START_DATE, end=END_DATE)
if ppi_raw.empty:
    ppi_raw = ld.get_history([RE_PPI_RIC], fields=["VALUE"], interval="monthly", start=START_DATE, end=END_DATE)

ppi_prices = ppi_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
ppi_returns = np.log(ppi_prices / ppi_prices.shift(1)).dropna()
ppi_returns.name = "PPI_RE"

# Stocks + Market
all_universe = TICKERS + ["IVV"]
prices_daily = ld.get_history(all_universe, fields=["TRDPRC_1"], interval="daily",
                              start=START_DATE, end=END_DATE, adjustments=["exchangeCorrection", "manualCorrection"])

monthly_returns = np.log(prices_daily.resample("ME").last() / prices_daily.resample("ME").last().shift(1)).dropna()
market_monthly = monthly_returns["IVV"]

# Macros
macro_dfs = []
for name, ric in MACRO_FACTORS.items():
    if name == "US10Y":
        df = ld.get_history([ric], fields=None, interval="daily", start=START_DATE, end=END_DATE)
        if not df.empty:
            for col in ['MID_YLD_1', 'A_YLD_1', 'B_YLD_1', 'YLDTOMAT']:
                if col in df.columns:
                    rets = df[col].diff().dropna()
                    rets.name = name
                    macro_dfs.append(rets)
                    break
    else:
        df = ld.get_history([ric], fields=["TRDPRC_1"], interval="daily", start=START_DATE, end=END_DATE)
        if not df.empty:
            rets = np.log(df.iloc[:, 0] / df.iloc[:, 0].shift(1)).dropna()
            rets.name = name
            macro_dfs.append(rets)

macro_monthly = pd.concat(macro_dfs, axis=1).resample("ME").last().ffill()

# ======================= REGRESSIONS =======================

ppi_lagged = ppi_returns.shift(1)
common_idx = ppi_lagged.index.intersection(monthly_returns.index).intersection(macro_monthly.index)

results_list = []
for ticker in TICKERS:
    if ticker not in monthly_returns.columns:
        continue

    stock = monthly_returns[ticker]
    df_reg = pd.concat([stock, ppi_lagged, market_monthly, macro_monthly], axis=1).loc[common_idx].dropna()

    if len(df_reg) < MIN_OBSERVATIONS:
        continue

    df_reg = df_reg.apply(pd.to_numeric, errors='coerce').dropna()
    df_reg.columns = ["stock", "PPI", "Market"] + list(macro_monthly.columns)

    X = sm.add_constant(df_reg.drop(columns=["stock"]))
    model = sm.OLS(df_reg["stock"], X.astype(float)).fit(cov_type='HC3')   # Robust SE

    results_list.append({
        "ticker": ticker,
        "delta_PPI": round(model.params["PPI"], 6),
        "p_value_PPI": round(model.pvalues["PPI"], 4),
        "significant_PPI": model.pvalues["PPI"] < P_VALUE_THRESHOLD,
        "r2_PPI": round(model.rsquared, 4),
    })

results_df = pd.DataFrame(results_list)

# ======================= COMPARISON =======================

for file, prefix in [("event_delta_results_MVREMXTR.csv", "MVREMXTR"),
                     ("event_delta_results_CSI.csv", "CSI")]:
    try:
        df = pd.read_csv(file)[["ticker", "delta_RE", "significant", "r2"]]
        df = df.rename(columns={
            "delta_RE": f"delta_{prefix}",
            "significant": f"sig_{prefix}",
            "r2": f"r2_{prefix}"
        })
        results_df = results_df.merge(df, on="ticker", how="left")
    except:
        print(f"Could not load {file}")

# Add sign agreement
results_df["PPI_MV_agree"] = np.sign(results_df["delta_PPI"]) == np.sign(results_df.get("delta_MVREMXTR", np.nan))
results_df["PPI_CSI_agree"] = np.sign(results_df["delta_PPI"]) == np.sign(results_df.get("delta_CSI", np.nan))

results_df = results_df.sort_values("delta_PPI")

# ======================= OUTPUT =======================

print("="*100)
print("FINAL RESULTS — Rare Earth Price Sensitivity Analysis")
print("="*100)
print(f"Period                    : {common_idx[0].date()} to {common_idx[-1].date()}")
print(f"Monthly observations      : {len(common_idx)}")
print(f"Companies                 : {len(results_df)}")
print(f"Significant PPI deltas    : {results_df['significant_PPI'].sum()} ({results_df['significant_PPI'].mean():.1%})")
print(f"Negative PPI delta        : {(results_df['delta_PPI'] < 0).sum()}")
print("-"*100)

# Display key columns
key_cols = ["ticker", "delta_PPI", "p_value_PPI", "significant_PPI",
            "delta_MVREMXTR", "sig_MVREMXTR", "delta_CSI", "sig_CSI",
            "PPI_MV_agree", "PPI_CSI_agree"]

print(results_df[key_cols].round(4).to_string(index=False))

# Excel export with formatting
with pd.ExcelWriter("ppi_robustness_full_comparison.xlsx", engine='xlsxwriter') as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = writer.sheets["Results"]

    # Format negative deltas in red
    red_format = workbook.add_format({'font_color': 'red'})
    for row in range(1, len(results_df)+1):
        if results_df.iloc[row-1]["delta_PPI"] < 0:
            worksheet.write(row, 1, results_df.iloc[row-1]["delta_PPI"], red_format)  # delta_PPI column

print("\nFull results + comparison exported to ppi_robustness_full_comparison.xlsx")

# Chart
colors = np.where(results_df["delta_PPI"] < 0, "#FF6B6B", "#4ECDC4")
results_df.plot(kind="barh", x="ticker", y="delta_PPI", figsize=(13, 15),
                color=colors, edgecolor="black", linewidth=0.6)

plt.title("Stock Sensitivity to China Rare Earth PPI\n"
          "(Multi-Factor OLS with Macro Controls)", fontsize=15, pad=20)
plt.xlabel("PPI Beta (Positive = Benefits from Rising RE Prices)")
plt.axvline(0, color="gray", linestyle="--", lw=1.2)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "ppi_delta_multi_factor.png", dpi=300, bbox_inches="tight")
plt.close()

print("High-resolution chart saved.")
print("\nAnalysis complete! ✅")
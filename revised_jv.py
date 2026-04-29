"""
China RE PPI — Full S&P 500 Multi-Factor OLS Robustness Check
With Side-by-Side Comparison + Advanced Visualizations
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

RE_PPI_RIC = "aCNCNHVGWM"
START_DATE = "2016-01-01"
END_DATE = "2026-04-01"

CHARTS_DIR = Path("charts/ppi")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

MACRO_FACTORS = {
    "VIX_PROXY": "VXX",
    "US10Y": "US10YT=RR",
    "DXY": ".DXY",
    "WTI": "CLc1",
    "COPPER": "HGc1"
}

# ======================= FETCH DATA =======================

ld.open_session()

TICKERS = ["NVDA.O", "AAPL.O", "GOOG.O", "GOOGL.O", "AVGO.O", "TSLA.O",
           "MU.O", "AMD.O",
           "AMAT.O", "LRCX.O", "INTC.O", "KLAC.O", "TXN.O", "ADI.O",
           "QCOM.O",
           "RTX.N", "BA.N", "LMT.N", "NOC.N", "GD.N", "CAT.N", "GE.N",
           "ETN.N",
           "PH.N", "HWM.N", "FCX.N", "WDC.O", "STX.O", "HON.O"]


# uncomment the below to do every stock in the sp500
# TICKERS = ld.get_data(
#     universe=['0#.SPX'],
#     fields=['TR.RIC']
# )['Instrument'].tolist()

print("LSEG session opened.\n")

print("=" * 90)
print("China RE PPI — Full S&P 500 Multi-Factor Analysis")
print("=" * 90)

# PPI
ppi_raw = ld.get_history([RE_PPI_RIC], fields=None, interval="monthly",
                         start=START_DATE, end=END_DATE)
if ppi_raw.empty:
    ppi_raw = ld.get_history([RE_PPI_RIC], fields=["VALUE"], interval="monthly",
                             start=START_DATE, end=END_DATE)

ppi_prices = ppi_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
ppi_returns = np.log(ppi_prices / ppi_prices.shift(1)).dropna()
ppi_returns.name = "PPI_RE"

# Stocks + Market
print(f"Fetching {len(TICKERS)} stocks + IVV...")
all_universe = TICKERS + ["IVV"]
prices_daily = ld.get_history(all_universe, fields=["TRDPRC_1"],
                              interval="daily",
                              start=START_DATE, end=END_DATE,
                              adjustments=["exchangeCorrection",
                                           "manualCorrection"])

monthly_returns = np.log(
    prices_daily.resample("ME").last() / prices_daily.resample(
        "ME").last().shift(1)).dropna()
market_monthly = monthly_returns["IVV"]

# Macro factors
macro_dfs = []
for name, ric in MACRO_FACTORS.items():
    print(f"  → {name} ({ric})")
    if name == "US10Y":
        df = ld.get_history([ric], fields=None, interval="daily",
                            start=START_DATE, end=END_DATE)
        if not df.empty:
            for col in ['MID_YLD_1', 'A_YLD_1', 'B_YLD_1', 'YLDTOMAT']:
                if col in df.columns:
                    rets = df[col].diff().dropna()
                    rets.name = name
                    macro_dfs.append(rets)
                    break
    else:
        df = ld.get_history([ric], fields=["TRDPRC_1"], interval="daily",
                            start=START_DATE, end=END_DATE)
        if not df.empty:
            rets = np.log(df.iloc[:, 0] / df.iloc[:, 0].shift(1)).dropna()
            rets.name = name
            macro_dfs.append(rets)

macro_monthly = pd.concat(macro_dfs, axis=1).resample("ME").last().ffill()

# ======================= REGRESSIONS =======================

ppi_lagged = ppi_returns.shift(1)
common_idx = ppi_lagged.index.intersection(monthly_returns.index).intersection(
    macro_monthly.index)

results_list = []
for ticker in TICKERS:
    if ticker not in monthly_returns.columns:
        continue

    stock = monthly_returns[ticker]
    df_reg = \
    pd.concat([stock, ppi_lagged, market_monthly, macro_monthly], axis=1).loc[
        common_idx].dropna()

    if len(df_reg) < MIN_OBSERVATIONS:
        continue

    df_reg = df_reg.apply(pd.to_numeric, errors='coerce').dropna()
    df_reg.columns = ["stock", "PPI", "Market"] + list(macro_monthly.columns)

    X = sm.add_constant(df_reg.drop(columns=["stock"]))
    model = sm.OLS(df_reg["stock"], X.astype(float)).fit(cov_type='HC3')

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
        df = df.rename(columns={"delta_RE": f"delta_{prefix}",
                                "significant": f"sig_{prefix}",
                                "r2": f"r2_{prefix}"})
        results_df = results_df.merge(df, on="ticker", how="left")
    except:
        pass

results_df["PPI_MV_agree"] = np.sign(results_df["delta_PPI"]) == np.sign(
    results_df.get("delta_MVREMXTR", np.nan))
results_df["PPI_CSI_agree"] = np.sign(results_df["delta_PPI"]) == np.sign(
    results_df.get("delta_CSI", np.nan))

results_df = results_df.sort_values("delta_PPI")

# ======================= SUMMARY =======================

sig_count = results_df["significant_PPI"].sum()
neg_count = (results_df["delta_PPI"] < 0).sum()

print("=" * 90)
print("FINAL RESULTS")
print("=" * 90)
print(
    f"Period                    : {common_idx[0].date()} to {common_idx[-1].date()}")
print(f"Companies analyzed        : {len(results_df)}")
print(
    f"Significant PPI deltas    : {sig_count} ({sig_count / len(results_df):.1%})")
print(f"Negative PPI delta        : {neg_count}")
print("-" * 90)

# ======================= ADVANCED VISUALIZATIONS =======================

significant_df = results_df[results_df['significant_PPI'] == True].copy()

if not significant_df.empty:
    print(
        f"\nGenerating detailed charts for {len(significant_df)} significant stocks...")

    # 1. Bar Chart
    plt.figure(figsize=(12, 10))
    colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in
              significant_df['delta_PPI']]
    plt.barh(significant_df['ticker'], significant_df['delta_PPI'],
             color=colors, edgecolor='black')
    plt.title('Significant Sensitivity to China RE PPI (Full S&P 500)',
              fontsize=16, pad=20)
    plt.xlabel('PPI Beta Coefficient')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "ppi_significant_bars.png", dpi=300,
                bbox_inches='tight')
    plt.close()

    # 2. Scatter Plots
    scat_dir = CHARTS_DIR / "scatters"
    scat_dir.mkdir(exist_ok=True)
    for _, row in significant_df.iterrows():
        ticker = row['ticker']
        df_s = pd.concat([monthly_returns[ticker], ppi_lagged], axis=1).dropna()
        df_s.columns = ['stock', 'PPI']

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(df_s['PPI'], df_s['stock'], alpha=0.7, s=50)
        z = np.polyfit(df_s['PPI'], df_s['stock'], 1)
        p = np.poly1d(z)
        ax.plot(df_s['PPI'], p(df_s['PPI']), "r--", lw=2.5,
                label=f'β = {row["delta_PPI"]:.4f}')
        ax.set_title(f'{ticker} — RE PPI Sensitivity')
        ax.legend()
        plt.savefig(scat_dir / f"{ticker}_scatter.png", dpi=280,
                    bbox_inches='tight')
        plt.close()

    # 3. Rolling Beta
    roll_dir = CHARTS_DIR / "rolling_beta"
    roll_dir.mkdir(exist_ok=True)
    for _, row in significant_df.iterrows():
        ticker = row['ticker']
        df_r = pd.concat([monthly_returns[ticker], ppi_lagged], axis=1).dropna()
        rolling = df_r.iloc[:, 0].rolling(24).cov(df_r.iloc[:, 1]) / df_r.iloc[
            :, 1].rolling(24).var()
        rolling.plot(figsize=(11, 6),
                     title=f'{ticker} — Rolling 24m Beta to RE PPI')
        plt.axhline(0, color='gray', linestyle='--')
        plt.savefig(roll_dir / f"{ticker}_rolling_beta.png", dpi=280,
                    bbox_inches='tight')
        plt.close()

    # 4. Cumulative Split
    cum_dir = CHARTS_DIR / "cumulative"
    cum_dir.mkdir(exist_ok=True)
    for _, row in significant_df.iterrows():
        ticker = row['ticker']
        df_c = pd.concat([monthly_returns[ticker], ppi_lagged], axis=1).dropna()
        high = df_c.iloc[:, 1] > df_c.iloc[:, 1].median()
        (1 + df_c.loc[high].iloc[:, 0]).cumprod().plot(label='High RE PPI',
                                                       color='#FF6B6B')
        (1 + df_c.loc[~high].iloc[:, 0]).cumprod().plot(label='Low RE PPI',
                                                        color='#4ECDC4')
        plt.title(f'{ticker} — High vs Low RE Price Periods')
        plt.legend()
        plt.savefig(cum_dir / f"{ticker}_cum_split.png", dpi=280,
                    bbox_inches='tight')
        plt.close()

print(
    "\nAll done! Check charts/ppi/ folder and ppi_robustness_full_comparison.xlsx")
results_df.to_excel("ppi_robustness_full_comparison.xlsx", index=False)
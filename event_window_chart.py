"""
Event Window Chart
==================
For each major rare earth price shock, shows how the top 15 most
exposed companies' stock prices moved in the 15 days before and
30 days after the shock.

This chart tests the inverse correlation hypothesis:
    RE prices go UP → exposed company stocks go DOWN

Each chart is normalized to 100 at the event day so all companies
can be compared on the same scale regardless of absolute price level.

HOW TO RUN:
    python3 event_window_chart.py

REQUIRES:
    daily_prices_top15.csv   — daily prices for top 15 companies + REMX
    re_prices_daily.csv      — daily SMM rare earth spot prices
    event_months.csv         — event months from event_regression.py
    dollar_exposure_events.csv — to identify top 15 companies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs("charts/event_windows", exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

DAYS_BEFORE = 15   # trading days before event
DAYS_AFTER  = 30   # trading days after event


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    # Daily stock prices for top 15
    stocks = pd.read_csv("daily_prices_top15.csv",
                         index_col=0, parse_dates=True)
    # Flatten multi-level columns if needed
    if isinstance(stocks.columns, pd.MultiIndex):
        stocks.columns = stocks.columns.get_level_values(-1)
    print(f"  Daily stocks   : {stocks.shape}")

    # Daily RE prices
    re_daily = pd.read_csv("re_prices_daily.csv",
                           index_col=0, parse_dates=True)
    if isinstance(re_daily.columns, pd.MultiIndex):
        re_daily.columns = re_daily.columns.get_level_values(-1)
    print(f"  Daily RE prices: {re_daily.shape}")

    # Event months
    event_months = pd.read_csv("event_months.csv",
                               index_col=0, parse_dates=True)
    events = event_months[event_months["is_event"]].copy()
    print(f"  Event months   : {len(events)}")

    # Load exposure to get company names
    try:
        exposure = pd.read_csv("dollar_exposure_events.csv")
    except FileNotFoundError:
        exposure = pd.read_csv("dollar_exposure.csv")

    return stocks, re_daily, events, exposure


# ── BUILD RE INDEX (daily) ─────────────────────────────────────────────────────

def build_daily_re_index(re_daily):
    """
    Builds a composite RE index from daily SMM prices.
    Normalizes each metal to 100 at start then averages.
    """
    # Rename columns for clarity
    rename = {
        "SMM-REM-NDM": "neodymium",
        "SMM-REO-DXO": "dysprosium",
        "SMM-MIN-GAL": "gallium",
        "SMM-MIN-GMN": "germanium",
        "SMM-REM-RCB": "cobalt",
    }
    re = re_daily.rename(columns=rename)

    # Use available metals
    metals = [m for m in rename.values() if m in re.columns]

    # Normalize each to 100 at first available date
    re_norm = re[metals].div(re[metals].iloc[0]) * 100

    # Equal weighted composite
    re_index = re_norm.mean(axis=1)
    re_index.name = "RE_composite"

    print(f"  Daily RE index built from: {metals}")
    return re_index


# ── GET TOP 15 COMPANIES ──────────────────────────────────────────────────────

def get_top15(exposure, stocks):
    """Gets top 15 companies by dollar exposure that have daily price data."""
    top = (exposure
           .dropna(subset=["Security", "dollar_exposure_usd"])
           .sort_values("dollar_exposure_usd", ascending=False))

    result = []
    for _, row in top.iterrows():
        ticker = row["ticker"]
        if ticker in stocks.columns:
            result.append({
                "ticker"  : ticker,
                "name"    : str(row["Security"])[:20],
                "tier"    : row.get("tier", ""),
                "exposure": row["exposure_billions"],
            })
        if len(result) == 15:
            break

    print(f"\nTop 15 companies found in daily data: {len(result)}")
    for r in result:
        print(f"  {r['ticker']:<10} {r['name']:<22} "
              f"${r['exposure']:.1f}B")

    return result


# ── FIND BIGGEST SHOCKS ───────────────────────────────────────────────────────

def get_biggest_shocks(events, n=6):
    """
    Returns the n events with the largest absolute z-score.
    These are the most extreme RE price shocks.
    """
    biggest = (events
               .reindex(events["z_score"].abs().sort_values(ascending=False).index)
               .head(n))

    print(f"\nTop {n} biggest shocks selected:")
    for date, row in biggest.iterrows():
        direction = "UP  " if row["re_return"] > 0 else "DOWN"
        print(f"  {date.strftime('%Y-%m-%d')}  {direction}  "
              f"z={row['z_score']:+.2f}  "
              f"RE return={row['re_return']:+.1%}")

    return biggest


# ── BUILD EVENT WINDOW ────────────────────────────────────────────────────────

def get_event_window(prices, event_date, days_before, days_after):
    """
    Extracts a window of prices around an event date.
    Returns prices normalized to 100 at the event date.

    Parameters:
        prices     : Series or DataFrame of daily prices
        event_date : the event date
        days_before: trading days before event
        days_after : trading days after event

    Returns:
        DataFrame indexed by trading day offset (-15 to +30)
    """
    # Find the event date index
    idx = prices.index.searchsorted(event_date)

    if idx >= len(prices):
        return None

    start = max(0, idx - days_before)
    end   = min(len(prices), idx + days_after + 1)

    window = prices.iloc[start:end].copy()

    # Normalize to 100 at event date
    event_value = prices.iloc[idx]
    if isinstance(event_value, pd.Series):
        window = window.div(event_value) * 100
    else:
        window = window / event_value * 100

    # Reindex to trading day offset
    event_pos = idx - start
    window.index = range(-event_pos, len(window) - event_pos)

    return window


# ── PLOT EVENT WINDOW ─────────────────────────────────────────────────────────

def plot_event_window(stocks_daily, re_index, event_date,
                      event_row, top15, days_before, days_after):
    """
    Plots the event window chart for one shock event.
    Shows:
        - Thin colored lines: top 15 company stock prices
        - Thick white line: RE composite index
        - Vertical red line: event day (day 0)
        - Horizontal gray line: 100 (baseline)
    """
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#A9CCE3"
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    # Plot each company
    for i, company in enumerate(top15):
        ticker = company["ticker"]
        if ticker not in stocks_daily.columns:
            continue

        window = get_event_window(
            stocks_daily[ticker].dropna(),
            event_date, days_before, days_after
        )

        if window is None or len(window) < 5:
            continue

        ax.plot(
            window.index,
            window.values,
            color=colors[i % len(colors)],
            linewidth=0.9,
            alpha=0.7,
            label=company["name"],
            zorder=2
        )

    # Plot RE index — thick white line
    re_window = get_event_window(
        re_index.dropna(),
        event_date, days_before, days_after
    )

    if re_window is not None and len(re_window) > 5:
        ax.plot(
            re_window.index,
            re_window.values,
            color="#FFFFFF",
            linewidth=3.5,
            label="RE Index",
            zorder=5
        )

    # Reference lines
    ax.axvline(0, color="#FF4444", linewidth=1.5,
               linestyle="--", zorder=4, alpha=0.8, label="Event day")
    ax.axhline(100, color="#444444", linewidth=0.8,
               linestyle="-", zorder=1)

    # Shade pre-event and post-event regions
    ax.axvspan(-days_before, 0, alpha=0.03, color="#ffffff")
    ax.axvspan(0, days_after, alpha=0.05, color="#FF4444")

    # Labels
    direction = "PRICE SPIKE" if event_row["re_return"] > 0 else "PRICE CRASH"
    z         = event_row["z_score"]
    re_ret    = event_row["re_return"]

    ax.set_xlabel("Trading days relative to event (0 = event day)",
                  color="#888888", fontsize=10)
    ax.set_ylabel("Normalized Price (base = 100 at event day)",
                  color="#888888", fontsize=10)
    ax.set_title(
        f"RE {direction} — {event_date.strftime('%B %Y')}\n"
        f"RE Index return: {re_ret:+.1%}  |  Z-score: {z:+.2f}  |  "
        f"Window: {days_before} days before → {days_after} days after",
        color="#ffffff", fontsize=12, pad=15
    )

    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#1a1a1a", linewidth=0.5)
    ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.5)

    ax.set_xlim(-days_before - 1, days_after + 1)

    ax.legend(
        loc="upper left", fontsize=7, ncol=2,
        facecolor="#111111", edgecolor="#333333",
        labelcolor="#aaaaaa", framealpha=0.85,
    )

    plt.tight_layout()

    direction_label = "up" if event_row["re_return"] > 0 else "down"
    path = (f"charts/event_windows/"
            f"event_{event_date.strftime('%Y%m')}_{direction_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.show()
    print(f"  Saved: {path}")
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("Event Window Chart")
    print("="*60)
    print(f"Window: {DAYS_BEFORE} days before → {DAYS_AFTER} days after")

    # Load
    stocks_daily, re_daily, events, exposure = load_data()

    # Build daily RE index
    re_index = build_daily_re_index(re_daily)

    # Get top 15 companies
    top15 = get_top15(exposure, stocks_daily)

    if not top15:
        print("ERROR: No top 15 companies found in daily price data")
        exit(1)

    # Get biggest shocks
    biggest_shocks = get_biggest_shocks(events, n=6)

    # Plot one chart per shock
    print(f"\nBuilding event window charts...")
    print("="*60)

    for event_date, event_row in biggest_shocks.iterrows():
        direction = "UP" if event_row["re_return"] > 0 else "DOWN"
        print(f"\nEvent: {event_date.strftime('%Y-%m-%d')} "
              f"({direction}, z={event_row['z_score']:+.2f})")

        plot_event_window(
            stocks_daily, re_index,
            event_date, event_row,
            top15,
            DAYS_BEFORE, DAYS_AFTER
        )

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nCharts saved to: charts/event_windows/")
    print(f"  One chart per event — {len(biggest_shocks)} total")
    print(f"\nWhat to look for:")
    print(f"  If inverse correlation exists:")
    print(f"    RE spike events → company lines trend DOWN after day 0")
    print(f"    RE crash events → company lines trend UP after day 0")
    print(f"  If lines move randomly → no clear causal relationship")

from datetime import date
from jake_folder.utils import get_trading_days
from jake_folder.summarize_missing_dates import summarize_missing_dates
import lseg.data as ld
import time
import os
import pandas as pd


def refresh_and_load_daily_historical_data(
        universe,
        daily_data_filename = "daily_data.csv",
        batch_size = 100,
        START_DATE = "2016-01-01",
        END_DATE = date.today()
):
    ld.open_session()
    universe = ld.get_data(
        universe=['0#.SPX'],
        fields=['TR.RIC']
    )['Instrument'].tolist() + ["IVV"]
    daily_data_filename = "daily_data.csv"
    batch_size = 100
    START_DATE = "2016-01-01"
    END_DATE = date.today()

    universe.sort()

    if os.path.exists(daily_data_filename):
        daily_data = pd.read_csv(daily_data_filename)
        print(f"✅ loaded '{daily_data_filename}'")

    else:
        data_list = []

        for i in range(0, len(universe), batch_size):
            batch = universe[i:i + batch_size]
            print(
                f"  Fetching batch {i // batch_size + 1} / {
                len(universe) // batch_size + 1} ({len(batch)} tickers)"
            )

            try:
                df = ld.get_history(
                    universe=batch,
                    fields=["TR.PriceClose"],
                    interval="daily",
                    start=START_DATE,
                    end=END_DATE
                )
                data_list.append(df)
                # Small pause to be nice to the API
                time.sleep(0.8)
            except Exception as e:
                print(f"    Error on batch: {e}")
                time.sleep(2)

        daily_data = pd.concat(data_list, axis=1)


    summarize_missing_dates(daily_data)

    daily_data.to_csv("daily_data.csv", index=False)
    return daily_data
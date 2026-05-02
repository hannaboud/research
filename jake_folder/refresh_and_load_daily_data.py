from datetime import date
from jake_folder.utils import get_trading_days
from jake_folder.summarize_missing_dates import summarize_missing_dates
import lseg.data as ld
import time
import os
import pandas as pd

# Creates daily_data.csv if it doesn't exist

def refresh_and_load_daily_data(
        universe,
        daily_data_filename = "daily_data.csv",
        batch_size = 5,
        START_DATE = "2016-01-01",
        END_DATE = date.today(),
        pkg_root = "jake_folder"
):
    universe.sort()

    daily_data_path = os.path.join(pkg_root, daily_data_filename)

    if os.path.exists(daily_data_path):
        daily_data = pd.read_csv(daily_data_path)
        print(f"✅ loaded '{daily_data_filename}'")

    else:
        data_list = []
        for i in range(0, len(universe), batch_size):
            batch = universe[i:i + batch_size]
            print("  Fetching data for: " + ", ".join(batch))
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
                time.sleep(1)
            except Exception as e:
                print(f"    Error on batch: {e}")
                time.sleep(2)

        daily_data = pd.concat(data_list, axis=1)
        daily_data.to_csv(daily_data_path)

    summarize_missing_dates(daily_data)

    return daily_data
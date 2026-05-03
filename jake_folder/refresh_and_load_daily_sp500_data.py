from datetime import date
from jake_folder.utils import get_trading_days, refinitiv_batch_fetch
from jake_folder.summarize_missing_dates import summarize_missing_dates
import lseg.data as ld
import pandas as pd
import os

# Creates daily_data.csv if it doesn't exist
# Feches daily data for everything in the SP500 plus what the user specifies in
#  the "additional_RICs" parameter
def refresh_and_load_daily_sp500_data(
        additional_RICs,
        daily_data_filename = "daily_data.csv",
        START_DATE = "2016-01-01",
        END_DATE = str(date.today()),
        pkg_root = "jake_folder"
):
    ld.open_session()
    sp500_RICs = ld.get_data(
        universe=['0#.SPX'],
        fields=['TR.RIC']
    )['Instrument'].tolist()
    additional_RICs = ["IVV"]
    universe = sp500_RICs + additional_RICs
    daily_data_filename = "daily_data.csv"
    START_DATE = "2016-01-01"
    END_DATE = str(date.today())
    pkg_root = "jake_folder"
    universe = universe[:15]

    daily_data_path = os.path.join(pkg_root, daily_data_filename)
    trading_days = pd.to_datetime(get_trading_days(START_DATE, END_DATE))

    if os.path.exists(daily_data_path):
        daily_data = pd.read_csv(
            daily_data_path,
            index_col='Date',
            parse_dates=True
        )
        print(f"Found existing '{daily_data_filename}'")

        missing = trading_days[~trading_days.isin(daily_data.index)]

        if len(missing) > 0:
            start_dt = missing.min().strftime('%Y-%m-%d')
            end_dt = missing.max().strftime('%Y-%m-%d')
            print("Missing dates found!")
            print(f'Fetching new data from {start_dt} to {end_dt}!')
            data_list = refinitiv_batch_fetch(
                universe=universe,
                batch_size=5,
                start_dt=START_DATE,
                end_dt=END_DATE,
                fields=["TR.PriceClose"],
                interval="daily",
                sleep_time=0.5
            )
        else:
            print("  -> daily_prices.csv is up-to-date")
            return daily_data

    else:
        data_list = refinitiv_batch_fetch(
            universe=universe,
            batch_size=5,
            start_dt=START_DATE,
            end_dt=END_DATE,
            fields=["TR.PriceClose"],
            interval="daily",
            sleep_time=0.5
        )

    ld.close_session()
    daily_data = pd.concat(data_list, axis=1)
    daily_data.to_csv(daily_data_path)

    return daily_data
import lseg.data as ld
from datetime import datetime
import time
import textwrap

def get_trading_days(start_date: str, end_date: str, calendar: str = "USA"):
    """
    Returns a list of trading days between two dates for a given calendar.

    calendar examples: "USA", "GBL" (London), "EUR" (Eurozone), "IND", "JPN", etc.
    """
    ld.open_session()
    response = ld.dates_and_calendars.date_schedule(
        start_date=start_date,
        end_date=end_date,
        frequency="daily",
        calendars=["USA"]
    )
    ld.close_session()
    return response


def refinitiv_batch_fetch(
        universe: list[str],
        batch_size: int,
        start_dt: str, end_dt: str,
        fields: list[str],
        interval: str,
        sleep_time: float
):
    ld.open_session()
    universe.sort()
    data_list = []
    for i in range(0, len(universe), batch_size):
        batch = universe[i:i + batch_size]
        print("  Fetching data for: " + ", ".join(batch), end=" ")
        try:
            df = ld.get_history(
                universe=batch,
                fields=fields,
                interval=interval,
                start=start_dt,
                end=end_dt
            )
            data_list.append(df)
            print("✔")
            # Small pause to be nice to the API
            time.sleep(sleep_time)
        except Exception as e:
            print(f"    Error on batch: {e}")
            time.sleep(2)
    ld.close_session()
    return data_list

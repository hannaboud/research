import lseg.data as ld
from datetime import datetime

def get_trading_days(start_date: str, end_date: str, calendar: str = "USA"):
    """
    Returns a list of trading days between two dates for a given calendar.

    calendar examples: "USA", "GBL" (London), "EUR" (Eurozone), "IND", "JPN", etc.
    """
    response = ld.dates_and_calendars.date_schedule(
        start=start_date,
        end=end_date,
        frequency="Daily",
        calendars=[calendar],  # exchange / country calendar
        date_adjustment="Following"  # or "Preceding", "ModifiedFollowing", etc.
    )

    # The response contains the adjusted trading days
    trading_days = [d['Date'] for d in response.data.raw['dates']]
    return trading_days

import lseg.data as ld

def refresh_and_load_ppi(START_DATE: str, END_DATE: str):
    ld.open_session()
    ppi_raw = ld.get_history(
        ["aCNCNHVGWM"],
        fields=None,
        interval="monthly",
        start=START_DATE,
        end=END_DATE
    )
    ld.close_session()

    ppi_prices = ppi_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
    ppi_returns = np.log(ppi_prices / ppi_prices.shift(1)).dropna()
    ppi_returns.name = "PPI_RE"

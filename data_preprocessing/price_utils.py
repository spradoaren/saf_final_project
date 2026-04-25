def extract_adjusted_close(df, ticker):
    """Return adjusted close series for ticker from MultiIndex (Price, Ticker) df."""
    try:
        return df["Close"][ticker].dropna()
    except KeyError:
        raise KeyError(
            f"Ticker {ticker} not found. Available: "
            f"{df.columns.get_level_values(1).unique().tolist()}"
        )

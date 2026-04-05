import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Union, Iterable, Optional
from enum import Enum
from abc import ABC, abstractmethod

import pandas as pd
import yfinance as yf

from curl_cffi import requests
session = requests.Session(impersonate="chrome")
ticker = yf.Ticker('...', session=session)

class DataAdapter(ABC):
    """
    Abstract Base Class defining the standard interface for all data sources.
    Includes shared caching logic.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            os.makedirs(self.cache_dir)

    @abstractmethod
    def get_data(
        self,
        tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        pass

    def _get_current_date(self) -> str:
        return datetime.today().strftime("%Y-%m-%d")

    def _generate_cache_key(
        self,
        tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
        start_date: str,
        end_date: str,
    ) -> str:

        if isinstance(tickers, (str, Enum)):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)

        normalized_tickers = []

        for t in ticker_list:
            if isinstance(t, Enum):
                normalized_tickers.append(str(t.value))
            else:
                normalized_tickers.append(str(t))

        normalized_tickers.sort()

        payload = {
            "tickers": normalized_tickers,
            "start": start_date,
            "end": end_date,
        }

        payload_str = json.dumps(payload, sort_keys=True)

        return hashlib.md5(payload_str.encode("utf-8")).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:

        return self.cache_dir / f"{cache_key}.csv"

    def clear_cache(self):

        for file in self.cache_dir.glob("*.csv"):
            file.unlink()

class YFinanceAdapter(DataAdapter):

    """
    Adapter for Yahoo Finance data.
    Implements caching and remote query.
    """

    def __init__(self, cache_dir: str = "data/yfinance_cache"):

        super().__init__(cache_dir)

    def get_data(
        self,
        tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:

        # ---------- Step 1: normalize end_date ----------

        if end_date is None:

            end_date = self._get_current_date()

        # ---------- Step 2: validate date range ----------

        today = datetime.today().date()

        if datetime.strptime(end_date, "%Y-%m-%d").date() > today:

            raise ValueError(
                f"End date {end_date} exceeds today's date {today}"
            )

        # ---------- Step 3: generate cache key ----------

        cache_key = self._generate_cache_key(
            tickers,
            start_date,
            end_date,
        )

        cache_path = self._get_cache_path(cache_key)

        # ---------- Step 4: load cache ----------

        if cache_path.exists() and not force_refresh:

            print("Loading data from cache...")

            df = pd.read_csv(
                cache_path,
                header=[0,1],     
                index_col=0,     
                parse_dates=True  
            )

            df.index.name = "Date"

            return df

        # ---------- Step 5: normalize ticker list ----------

        if isinstance(tickers, (str, Enum)):

            ticker_list = [tickers]

        else:

            ticker_list = list(tickers)

        normalized = []

        for t in ticker_list:

            if isinstance(t, Enum):

                normalized.append(str(t.value))

            else:

                normalized.append(str(t))

        # ---------- Step 6: fetch from yfinance ----------

        print("Fetching data from Yahoo Finance...")

        df = yf.download(

            tickers=normalized,

            start=start_date,

            end=end_date,

            auto_adjust=False,

            progress=False,

        )

        if df.empty: # type: ignore

            raise ValueError("No data returned from Yahoo Finance")

        # ---------- Step 7: save cache ----------

        df.to_csv(cache_path) # type: ignore

        print(f"Cache saved at {cache_path}")

        return df  # type: ignore


if __name__ == "__main__":

    adapter = YFinanceAdapter()

    df = adapter.get_data(

        tickers=["AAPL", "MSFT"],

        start_date="2024-01-01",

        end_date="2024-02-01",

    )

    print(df.head())

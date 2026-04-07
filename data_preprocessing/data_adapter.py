from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Iterable, Optional, List
from enum import Enum
from datetime import datetime, date, timedelta
import hashlib
import time
import random
import warnings
import logging

import pandas as pd
import yfinance as yf

TickerInput = Union[str, Enum, Iterable[str], Iterable[Enum]]

logger = logging.getLogger(__name__)


class DataAdapter(ABC):
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_data(
        self,
        tickers: TickerInput,
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _today_str(self) -> str:
        return date.today().strftime("%Y-%m-%d")

    def _parse_date(self, s: str) -> date:
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date()
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {s!r}") from e

    def _normalize_tickers(self, tickers: TickerInput) -> List[str]:
        if isinstance(tickers, (str, Enum)):
            raw = [tickers]
        else:
            raw = list(tickers)

        out: List[str] = []
        for t in raw:
            value = str(t.value) if isinstance(t, Enum) else str(t)
            value = value.strip().upper()
            if value:
                out.append(value)

        if not out:
            raise ValueError("No tickers provided.")

        return sorted(set(out))

    def _cache_path_for(self, ticker: str, start_date: str, end_date: str) -> Path:
        key = hashlib.md5(f"{ticker}|{start_date}|{end_date}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{ticker}_{key}.parquet"

    def clear_cache(self) -> None:
        for pattern in ("*.pkl", "*.csv", "*.parquet"):
            for file in self.cache_dir.glob(pattern):
                try:
                    file.unlink()
                except FileNotFoundError:
                    pass


class YFinanceAdapter(DataAdapter):
    def __init__(self, cache_dir: str = "data/yfinance_cache"):
        super().__init__(cache_dir)

    def get_data(
        self,
        tickers: TickerInput,
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
        delay_between: float = 0.5,
    ) -> pd.DataFrame:
        today_d = date.today()

        start_d = self._parse_date(start_date)

        if end_date is None:
            end_date = self._today_str()
        end_d = self._parse_date(end_date)

        if end_d > today_d:
            warnings.warn(
                f"end_date clamped from {end_date} to today {today_d.isoformat()}",
                UserWarning,
            )
            end_d = today_d
            end_date = today_d.strftime("%Y-%m-%d")

        if start_d > end_d:
            raise ValueError(f"start_date {start_date} must be <= end_date {end_date}.")

        tickers_list = self._normalize_tickers(tickers)

        # yfinance end date is exclusive, so add one day
        yf_end = (end_d + timedelta(days=1)).strftime("%Y-%m-%d")

        dfs: List[pd.DataFrame] = []
        failed: List[str] = []

        for i, ticker in enumerate(tickers_list):
            cache_path = self._cache_path_for(ticker, start_date, end_date)

            if cache_path.exists() and not force_refresh:
                try:
                    cached = pd.read_parquet(cache_path)
                    cached.index = pd.to_datetime(cached.index)
                    if not isinstance(cached.columns, pd.MultiIndex):
                        raise ValueError("Cached dataframe does not have MultiIndex columns.")
                    cached.columns.names = ["Price", "Ticker"]
                    dfs.append(cached)
                    continue
                except Exception as e:
                    logger.warning(
                        "Failed reading cache for %s at %s: %s. Refetching.",
                        ticker,
                        cache_path,
                        e,
                    )

            try:
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=yf_end,
                    auto_adjust=False,
                    progress=False,
                    timeout=20,
                    group_by="column",
                    threads=False,
                )

                if df is None or df.empty:
                    raise RuntimeError("Empty response — ticker may be invalid or delisted.")

                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"

                # Single-ticker downloads often come back with plain columns like
                # Open, High, Low, Close, Adj Close, Volume. Normalize to a
                # consistent (Price, Ticker) MultiIndex.
                if isinstance(df.columns, pd.MultiIndex):
                    if df.columns.nlevels != 2:
                        raise ValueError(
                            f"Unexpected column shape for {ticker}: {df.columns.nlevels} levels"
                        )

                    level0 = list(df.columns.get_level_values(0))
                    level1 = list(df.columns.get_level_values(1))

                    if all(str(x).upper() == ticker for x in level1):
                        df.columns = pd.MultiIndex.from_arrays(
                            [level0, level1],
                            names=["Price", "Ticker"],
                        )
                    elif all(str(x) in {"Open", "High", "Low", "Close", "Adj Close", "Volume"} for x in level1):
                        # In some cases the order comes back reversed.
                        df.columns = pd.MultiIndex.from_arrays(
                            [level1, level0],
                            names=["Price", "Ticker"],
                        )
                    else:
                        # Force names even if already structurally correct.
                        df.columns.names = ["Price", "Ticker"]
                else:
                    df.columns = pd.MultiIndex.from_product(
                        [df.columns, [ticker]],
                        names=["Price", "Ticker"],
                    )

                # Sort columns for deterministic output/cache.
                df = df.sort_index(axis=1)

                df.to_parquet(cache_path)
                dfs.append(df)

            except Exception as e:
                logger.warning("Ticker %s failed: %s", ticker, e)
                failed.append(ticker)

            if i < len(tickers_list) - 1 and delay_between > 0:
                sleep_time = delay_between + random.uniform(0, min(1.5, delay_between + 0.5))
                time.sleep(sleep_time)

        if not dfs:
            raise RuntimeError(f"No data retrieved. Failed tickers: {failed}")

        if failed:
            warnings.warn(
                f"These tickers failed and are excluded: {failed}",
                UserWarning,
            )

        result = pd.concat(dfs, axis=1).sort_index(axis=1)
        result = result.loc[:, ~result.columns.duplicated()]

        return result
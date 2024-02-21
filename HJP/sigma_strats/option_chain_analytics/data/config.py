"""
define api and data specific conversions of options and futures tickers
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from enum import Enum

TIME_FMT = '%Y%m%d%H%M%S'
EXPIRY_DATE_FORMAT = '%d%b%Y'
SECONDS_PER_YEAR = 365*24*60*60  # days, hours, minute, seconds


class NearestStrikeSelection(Enum):
    # nearest strike from the strikes grid, for put ratios must be strictly below or above
    NEAREST = 0
    MAX_OI = 1
    ABOVE = 2
    BELOW = 3


class DeltaType(Enum):
    MARKED = 1
    NORMAL = 2


class OptionTickerConfig(Enum):
    """
    enumerate ticker methodology for each exchange or data provider
    we need to extract 3 data from the ticker: maturity, strike, type
    """
    CMS = 1   # deribit-ETH-28OCT22-2000-C-option
    DERIBIT = 2  # ETH-13JAN23-1100-P
    YAHOO = 3  # SPY230117C00353000


def get_ttm_from_dates(mat_date: pd.Timestamp, value_time: pd.Timestamp, is_floor_at_zero: bool = True) -> float:
    ttm = (mat_date - value_time).total_seconds() / SECONDS_PER_YEAR
    if is_floor_at_zero:
        ttm = np.maximum(ttm, 0.0)
    return ttm


def split_option_contract_ticker(contract: str = 'deribit-ETH-28OCT22-2000-C-option',
                                 option_ticker_config: OptionTickerConfig = OptionTickerConfig.CMS
                                 ) -> Tuple[str, float, str]:
    """
    split ticker as follows
    deribit-SOL-28OCT22-10-C-option
    """
    if option_ticker_config == OptionTickerConfig.CMS:
        parts = contract.split('-')
        if len(parts) < 5:
            raise ValueError(f"invalid contract = {contract}")
        mat_str = parts[2]
        strike = float(parts[3])
        option_type = parts[4]
    elif option_ticker_config == OptionTickerConfig.DERIBIT:
        parts = contract.split('-')
        mat_str = parts[1]
        strike = float(parts[2])
        option_type = parts[3]
    else:
        raise NotImplementedError

    return mat_str, strike, option_type


def get_option_data_from_contract(value_time: pd.Timestamp,
                                  contract: str = 'deribit-ETH-28OCT22-2000-C-option',
                                  option_ticker_config: OptionTickerConfig = OptionTickerConfig.CMS
                                  ) -> Tuple[str, pd.Timestamp, float, str, float]:
    """
    split ticker as follows
    deribit-SOL-28OCT22-10-C-option
    """
    mat_str, strike, option_type = split_option_contract_ticker(contract=contract,
                                                                option_ticker_config=option_ticker_config)
    mat_date = mat_to_timestamp(mat_str)
    ttm = get_ttm_from_dates(mat_date=mat_date, value_time=value_time)
    return mat_str, mat_date, strike, option_type, ttm


class FutureTickerConfig(Tuple, Enum):
    """
    enumertate ticker methodology for each exchange or data provider
    we need to extract 3 data from the ticker: maturity, strike, type
    """
    CMS = (1, "%d%b%y")  # deribit-ETH-28OCT22-future
    DERIBIT = (2, "%d%b%y")  # ETH-13JAN23
    BINANCEUSDM = (3, "%y%m%d")  # ETH/USDT, ETHUSDT_230331
    BYBIT = (4, "%y%m%d")  # ETH/USD:ETH-230630
    OKX = (5, "%y%m%d")  # ETH-USDT-SWAP
    HUOBI = (6, "%y%m%d")  # ETH-USD, ETH-USDT-230127


def split_future_contract_ticker(contract: str = 'deribit-ETH-30DEC22-future',
                                 future_ticker_config: FutureTickerConfig = FutureTickerConfig.CMS
                                 ) -> Tuple[str, Optional[pd.Timestamp]]:
    """
    split ticker as follows
    deribit-SOL-28OCT22-10-C-option
    """
    if future_ticker_config == FutureTickerConfig.CMS:
        parts = contract.split('-')
        mat_str = parts[2]
        if mat_str == 'PERPETUAL':
            mat_ts = None
        else:
            mat_ts = mat_to_timestamp(mat_str, date_format=future_ticker_config[1])

    elif future_ticker_config == FutureTickerConfig.DERIBIT:
        parts = contract.split('-')
        mat_str = parts[-1]
        if mat_str == 'PERPETUAL':
            mat_ts = None
        else:
            mat_ts = mat_to_timestamp(mat_str, date_format=future_ticker_config[1])

    elif future_ticker_config == FutureTickerConfig.BINANCEUSDM:
        parts = contract.split('_')
        mat_str = parts[-1]
        if len(parts) == 1:
            mat_ts = None
        else:
            mat_ts = mat_to_timestamp(mat_str, date_format=future_ticker_config[1])

    elif future_ticker_config == FutureTickerConfig.BYBIT:
        parts = contract.split('-')
        mat_str = parts[-1]
        if len(parts) == 1:  # perp is /USD:USDC
            mat_ts = None
        else:
            mat_ts = mat_to_timestamp(mat_str, date_format=future_ticker_config[1])

    elif future_ticker_config == FutureTickerConfig.OKX:
        parts = contract.split('-')
        mat_str = parts[-1]
        if mat_str == 'SWAP':
            mat_ts = None
        else:
            mat_ts = mat_to_timestamp(mat_str, date_format=future_ticker_config[1])

    elif future_ticker_config == FutureTickerConfig.HUOBI:
        parts = contract.split('-')
        mat_str = parts[-1]
        if not len(parts) == 3:
            mat_ts = None
        else:
            mat_str = re.findall(r'\d+', contract)[0]  # get list of nums in string
            mat_ts = mat_to_timestamp(mat_str, date_format=future_ticker_config[1])
    else:
        raise NotImplementedError(f"future_ticker_config={future_ticker_config}")

    return mat_str, mat_ts


def get_ttm_from_future_ticker(value_time: pd.Timestamp,
                               contract: str = 'deribit-ETH-30DEC22-future',
                               future_ticker_config: FutureTickerConfig = FutureTickerConfig.CMS,
                               is_floor_at_zero: bool = True
                               ) -> float:
    mat_str, mat_date = split_future_contract_ticker(contract=contract, future_ticker_config=future_ticker_config)
    if mat_date is not None:
        ttm = get_ttm_from_dates(mat_date=mat_date, value_time=value_time, is_floor_at_zero=is_floor_at_zero)
    else:
        if is_floor_at_zero:  # so no confusion
            ttm = 0.0
        else:
            ttm = np.inf
    return ttm


def mat_to_timestamp(mat: str,
                     date_format: str = "%d%b%y"  # deribit is default
                     ) -> pd.Timestamp:
    """
    can be generalized to exchange
    """
    return pd.Timestamp(pd.to_datetime(mat, format=date_format), tz='UTC') + pd.to_timedelta(8.00, unit='h')


def get_file_name(ticker: str, freq: Optional[str], hour_offset: Optional[int]) -> str:
    if freq is not None:
        if freq == 'H':
            file_name = f"{ticker}_freq_{freq}"
        else:
            if hour_offset is not None:
                file_name = f"{ticker}_freq_{freq}_hour_{hour_offset}"
            else:
                file_name = f"{ticker}_freq_{freq}"
    else:
        file_name = ticker
    return file_name


class UnitTests(Enum):
    OPTIONS_TICKER = 1
    FUTURES_TICKER = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.OPTIONS_TICKER:
        mat_str, strike, option_type = split_option_contract_ticker()
        print(mat_str)
        print(strike)
        print(option_type)

    elif unit_test == UnitTests.FUTURES_TICKER:
        mat_str, mat_ts = split_future_contract_ticker(contract='deribit-BTC-18MAR22-future')
        print(f"mat_str={mat_str}")
        print(f"mat_ts={mat_ts}")

        mat_str, mat_ts = split_future_contract_ticker(contract='ETH/USD:USDC',
                                                       future_ticker_config=FutureTickerConfig.BYBIT)
        print(f"mat_str={mat_str}")
        print(f"mat_ts={mat_ts}")

        mat_str, mat_ts = split_future_contract_ticker(contract='ETHUSDT_230331',
                                                       future_ticker_config=FutureTickerConfig.BINANCEUSDM)
        print(f"mat_str={mat_str}")
        print(f"mat_ts={mat_ts}")


if __name__ == '__main__':

    unit_test = UnitTests.FUTURES_TICKER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

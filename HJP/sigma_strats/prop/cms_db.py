"""
create a local copy of data from AWS
"""
import pandas as pd
import qis
from qis import TimePeriod
from typing import Dict, Optional, Union
from enum import Enum

# oca
import sigma_strats.local_path as local_path
import sigma_strats.option_chain_analytics.data.config as gu
from sigma_strats.option_chain_analytics.option_chain import SliceColumn


def get_start_end_date() -> TimePeriod:
    time_period = TimePeriod('2021-09-02 08:00:00+00:00', '2023-02-14 8:00:00+00:00', tz='UTC')
    return time_period


# will serve as columns for value data
CMS_SLICE_COLUMNS = [SliceColumn.CONTRACT,
                     SliceColumn.MARK_PRICE,
                     SliceColumn.UNDERLYING_PRICE,
                     SliceColumn.BID_PRICE,
                     SliceColumn.ASK_PRICE,
                     SliceColumn.MARK_IV,
                     SliceColumn.BID_IV,
                     SliceColumn.ASK_IV,
                     SliceColumn.OPEN_INTEREST,
                     SliceColumn.DELTA,
                     SliceColumn.VEGA,
                     SliceColumn.THETA,
                     SliceColumn.GAMMA,
                     SliceColumn.EXCHANGE_TIME,
                     SliceColumn.USD_MULTIPLIER] # inserted as column


@qis.timer
def parse_df_all_to_dict(df_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # pivot to columns
    all_columns = list([x.value for x in CMS_SLICE_COLUMNS])
    all_columns.remove('contract')
    all_columns.remove('exchange_time')  # cannot be aggregated by pivot
    df_all = df_all.drop('exchange_time', axis=1)
    this = pd.pivot_table(df_all, values=all_columns, index='index', columns='contract', dropna=False)
    dfs = {x: this[x].sort_index() for x in all_columns}
    return dfs


def load_dfs_from_local(ticker: str = 'BTC', freq: str = 'D', hour_offset: int = 8) -> Dict[str, pd.DataFrame]:
    table_name = gu.get_file_name(ticker=ticker, freq=freq, hour_offset=hour_offset)
    try:
        df_all = qis.load_df_from_feather(file_name=table_name, local_path=f"{local_path.get_resource_path()}\\deribit\\")
    except:
        df_all = qis.load_df_from_feather(file_name=table_name, local_path=f"{local_path.get_resource_path()}/deribit/")

    dfs = parse_df_all_to_dict(df_all=df_all)
    return dfs


@qis.timer
def load_contract_ts_data_db(ticker: str = 'BTC',
                             freq: str = 'D',
                             hour_offset: Optional[int] = 8
                             ) -> Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame]]:

    data_dict = load_dfs_from_local(ticker=ticker, freq=freq, hour_offset=hour_offset)
    try:
        spot_data = qis.load_df_from_feather(file_name=f"{ticker}-spot", local_path=f"{local_path.get_resource_path()}\\deribit\\")
    except:
        spot_data = qis.load_df_from_feather(file_name=f"{ticker}-spot", local_path=f"{local_path.get_resource_path()}/deribit/")

    option_ticker_config = gu.OptionTickerConfig.CMS
    return dict(data_dict=data_dict, spot_data=spot_data, ticker=ticker, option_ticker_config=option_ticker_config)


@qis.timer
def read_sql_table(ticker: str = 'BTC', freq: str = 'D', hour_offset: int = 8):
    engine = local_path.get_aws_engine()
    table_name = gu.get_file_name(ticker=ticker, freq=freq, hour_offset=hour_offset)
    return pd.read_sql_table(table_name=table_name, con=engine, schema='deribit')


def save_from_db_to_local(ticker: str = 'BTC', freq: str = 'D', hour_offset: int = 8) -> None:
    df_all = read_sql_table(ticker=ticker, freq=freq, hour_offset=hour_offset)
    table_name = gu.get_file_name(ticker=ticker, freq=freq, hour_offset=hour_offset)
    qis.save_df_to_feather(df=df_all, file_name=table_name, local_path=f"{local_path.get_resource_path()}\\deribit\\")
    df_spot = pd.read_sql_table(table_name=f"{ticker}-spot", con=local_path.get_aws_engine(), schema='deribit')
    qis.save_df_to_feather(df=df_spot, file_name=f"{ticker}-spot", local_path=f"{local_path.get_resource_path()}\\deribit\\")
    print(f"saved {table_name}")


class UnitTests(Enum):
    SAVE_FROM_DB_TO_LOCAL = 1
    LOAD_CONTRACTS_DATA = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.SAVE_FROM_DB_TO_LOCAL:
        tickers = ['BTC', 'ETH']
        freqs = ['D', 'H']
        freqs = ['D']
        for ticker in tickers:
            for freq in freqs:
                save_from_db_to_local(ticker=ticker, freq=freq)

    elif unit_test == UnitTests.LOAD_CONTRACTS_DATA:
        from sigma_strats.option_chain_analytics.ts_data import OptionsDataDFs
        ts_data = OptionsDataDFs(**load_contract_ts_data_db(ticker='ETH', freq='D'))
        ts_data.print()


if __name__ == '__main__':

    unit_test = UnitTests.LOAD_CONTRACTS_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

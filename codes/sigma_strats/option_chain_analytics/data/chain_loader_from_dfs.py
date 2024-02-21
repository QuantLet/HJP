"""
create chain data object with options using time series options data in OptionsDataDFs
"""
# built in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
from enum import Enum
import qis
from qis import timer, TimePeriod

# analytics
from sigma_strats.option_chain_analytics.data import config as gu
from sigma_strats.option_chain_analytics.option_chain import SliceColumn, UnderlyingColumn, ExpirySlice, SlicesChain
from sigma_strats.option_chain_analytics.ts_data import OptionsDataDFs


# @timer
def create_chain_from_from_options_dfs(options_data_dfs: OptionsDataDFs,
                                       value_time: pd.Timestamp,
                                       option_ticker_config: gu.OptionTickerConfig = gu.OptionTickerConfig.CMS
                                       ) -> Optional[SlicesChain]:

    options_df = options_data_dfs.get_time_slice(timestamp=value_time)
    if not options_df.empty:

        if option_ticker_config == gu.OptionTickerConfig.CMS:  # explicit use of CMS data

            options_df[SliceColumn.CONTRACT.value] = options_df.index.to_list()
            options_df[['mat_id', SliceColumn.EXPIRY.value, SliceColumn.STRIKE.value, SliceColumn.OPTION_TYPE.value, SliceColumn.TTM.value]] \
                = [gu.get_option_data_from_contract(value_time=value_time,
                                                    contract=x,
                                                    option_ticker_config=option_ticker_config
                                                    ) for x in options_df[SliceColumn.CONTRACT.value].to_list()]
            options_df[SliceColumn.STRIKE.value] = options_df[SliceColumn.STRIKE.value].apply(pd.to_numeric, errors='ignore')
            options_df[SliceColumn.TTM.value] = options_df[SliceColumn.TTM.value].apply(pd.to_numeric, errors='ignore')

            mat_slice = options_df.groupby('mat_id')
            expiry_slices, undelying_datas = {}, {}
            for mat, df in mat_slice:
                forward = qis.np_nonan_weighted_avg(a=df[SliceColumn.UNDERLYING_PRICE], weights=df[SliceColumn.OPEN_INTEREST])   # forward is contract weighted avs
                if not np.isnan(forward):
                    undelying_data = {UnderlyingColumn.EXPIRY_ID: str(mat),
                                      UnderlyingColumn.VALUE_TIME: value_time,
                                      UnderlyingColumn.EXPIRY: df[SliceColumn.EXPIRY].iloc[0],
                                      UnderlyingColumn.SPOT_PRICE: forward,  # to do
                                      UnderlyingColumn.FUTURE: str(mat),  # to do
                                      UnderlyingColumn.FUTURE_PRICE: forward,  # to do
                                      UnderlyingColumn.IR_RATE: 0.0,
                                      UnderlyingColumn.TTM: df[SliceColumn.TTM].iloc[0]}
                    undelying_data = pd.Series(undelying_data)
                    expiry_slices[str(mat)] = ExpirySlice(options=df, undelying_data=undelying_data)
                    undelying_datas[str(mat)] = undelying_data
            undelying_df = pd.DataFrame.from_dict(undelying_datas, orient='index')
            chain = SlicesChain(options_df=options_df.set_index(SliceColumn.CONTRACT),
                                undelying_df=undelying_df,
                                expiry_slices=expiry_slices,
                                value_time=value_time)

        else:
            raise NotImplementedError(f"{option_ticker_config}")
    else:
        chain = None
    return chain


@timer
def create_chain_timeseries(options_data_dfs: OptionsDataDFs,
                            dates_schedule: pd.DatetimeIndex = None,
                            time_period: TimePeriod = None,
                            freq: str = 'W-FRI',
                            hour_offset: int = 8
                            ) -> Dict[pd.Timestamp, SlicesChain]:

    if dates_schedule is None:
        if time_period is None:
            raise ValueError(f"time_period={time_period} must be non Nons")
        dates_schedule = qis.generate_dates_schedule(time_period=time_period,
                                                     freq=freq,
                                                     hour_offset=hour_offset)

    chain_data = {}
    for timestamp in dates_schedule:
        chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=timestamp)
        if chain is not None:
            chain_data[timestamp] = chain
    return chain_data


@timer
def generate_vol_delta_ts(options_data_dfs: OptionsDataDFs,
                          days_map: Dict[str, int] = None,
                          deltas: List[float] = (-0.10, -0.25, 0.50, 0.25, 0.10),
                          freq: str = 'D',
                          hour_offset: Optional[int] = 8,
                          time_period: TimePeriod = None
                          ) -> Tuple[pd.DataFrame, ...]:

    chain_data = create_chain_timeseries(options_data_dfs=options_data_dfs,
                                         time_period=time_period,
                                         freq=freq,
                                         hour_offset=hour_offset)
    if days_map is None:
        days_map = {'1d': 1, '2d': 2, '1w': 7, '2w': 10, '1m': 28, '2m': 56, '3m': 84, '2Q': 168}

    vols, strikes, options, index_prices = {}, {}, {}, {}
    for date, chain in chain_data.items():
        delta_vol_matrix = chain.generate_delta_vol_matrix(value_time=date, days_map=days_map, deltas=deltas)
        if delta_vol_matrix is not None:
            vols_, strikes_, options_, index_prices_ = delta_vol_matrix.get_melted_matrix()
            vols[date] = vols_['vols'].rename(date)
            strikes[date] = strikes_['strikes'].rename(date)
            options[date] = options_['options'].rename(date)
            index_prices[date] = index_prices_['index_prices'].rename(date)
    vols = pd.DataFrame.from_dict(vols, orient='index')
    strikes = pd.DataFrame.from_dict(strikes, orient='index')
    options = pd.DataFrame.from_dict(options, orient='index').applymap(lambda x: x.replace('deribit-', ''))
    index_prices = pd.DataFrame.from_dict(index_prices, orient='index')
    return vols, strikes, options, index_prices


class UnitTests(Enum):
    CREATE_CHAIN_AT_TS = 1
    CREATE_TS_CHAIN_DATA = 2
    CREATE_WEEKLY_ROLLS = 3


def run_unit_test(unit_test: UnitTests):

    from sigma_strats.prop import load_contract_ts_data_db, get_start_end_date

    if unit_test == UnitTests.CREATE_CHAIN_AT_TS:
        date = pd.Timestamp('2022-09-16 08:00:00+00:00')
        # date = pd.Timestamp('2021-09-29 08:00:00+00:00') # start date for eth
        date = pd.Timestamp('2021-10-28 08:00:00+00:00')
        date = pd.Timestamp('2022-09-19 08:00:00+00:00')
        ticker = 'ETH'
        options_data_dfs = OptionsDataDFs(**load_contract_ts_data_db(ticker=ticker, hour_offset=8))
        chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=date)
        chain.save_joint_slices(file_name=f"{ticker}_{date.strftime('%Y%m%d_%H_%M_%S')}")

        next_date = date+pd.DateOffset(days=30)
        slice_id = chain.get_next_slice_after_date(mat_date=next_date)
        print(f"{date}: {slice_id}")
        print(f"{chain.get_atm_put_id(slice_id=slice_id)}, {chain.get_atm_call_id(slice_id=slice_id)}")
        print(f"{chain.get_put_delta_option_id(slice_id=slice_id, delta=-0.25)}")
        print(f"{chain.get_call_delta_option_id(slice_id=slice_id, delta=0.25)}")

        # days_map = {'1d': 1, '2d': 2, '1w': 7, '2w': 14, '1m': 30, '2m': 60, '3m': 90, '2Q': 120,  '3Q': 180}
        # days_map = {'1d': 1, '2d': 2, '1w': 7, '2w': 14, '1m': 30, '2m': 60, '3m': 90, '2Q': 120}
        days_map = {'2d': 2, '1w': 7, '2w': 14, '1m': 30, '2m': 60, '3m': 90}
        delta_vol_matrix = chain.generate_delta_vol_matrix(value_time=date, days_map=days_map)
        delta_vol_matrix.print()

        vols, strikes, options, index_prices = delta_vol_matrix.get_melted_matrix()
        print(vols)
        print(strikes)
        delta_vol_matrix.plot_vol_in_strike()

    elif unit_test == UnitTests.CREATE_TS_CHAIN_DATA:
        options_data_dfs = OptionsDataDFs(**load_contract_ts_data_db(ticker='SOL', hour_offset=8, freq='D'))
        chain_data = create_chain_timeseries(options_data_dfs=options_data_dfs,
                                             time_period=get_start_end_date(ticker='BTC'),
                                             freq='W-FRI',
                                             hour_offset=8)
        for key, chain in chain_data.items():
            print(f"{key}, {chain.expiry_slices.keys()}")

    elif unit_test == UnitTests.CREATE_WEEKLY_ROLLS:
        weekly_fridays_rolls = qis.generate_dates_schedule(TimePeriod(pd.Timestamp('2022-05-06 00:00:00+00:00'),
                                                                      qis.get_current_time_with_tz(tz='UTC', days_offset=7)),
                                                           freq='W-FRI',
                                                           hour_offset=8)
        weekly_fridays_rolls = weekly_fridays_rolls[:-1]

        options_data_dfs = OptionsDataDFs(**load_contract_ts_data_db(ticker='ETH', hour_offset=8, freq='D'))
        chain_data = create_chain_timeseries(options_data_dfs=options_data_dfs,
                                             dates_schedule=weekly_fridays_rolls[:-1])

        for date, next_date in zip(weekly_fridays_rolls[:-1], weekly_fridays_rolls[1:]):
            print(f"{date}: {chain_data[date].get_next_slice_after_date(mat_date=next_date)}, "
                  f"{chain_data[date].get_atm_put_id(mat_date=next_date)}")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CREATE_WEEKLY_ROLLS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

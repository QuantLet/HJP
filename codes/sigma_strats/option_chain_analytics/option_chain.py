"""
iinterface for gen vol gen vol slices data
contract convention is 'deribit-SOL-28OCT22-10-C-option'
"""
from __future__ import annotations  # to allow class method annotations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Dict, Tuple, Optional
from numba.typed import List
from enum import Enum
import qis

# analytics
from sigma_strats.option_chain_analytics.data.config import NearestStrikeSelection, get_ttm_from_dates

IS_FORCE_ALL_COLUMNS_DATA = False


class UnderlyingColumn(str, Enum):
    """
    mandatory field for underlying data
    """
    EXPIRY_ID = 'expiry_id'
    VALUE_TIME = 'value_time'
    EXPIRY = 'expiry'
    SPOT_PRICE = 'spot_price'
    FUTURE = 'future'
    FUTURE_PRICE = 'future_price'
    IR_RATE = 'ir_rate'
    TTM = 'ttm'


class SliceColumn(str, Enum):
    """
    mandatory columns for calls and puts dataframes
    for crypto inverse options, mark_price is in units of underlying and usd_multiplier = underlying_price
    """
    # comes from data set
    MARK_PRICE = 'mark_price'  # option mark
    UNDERLYING_PRICE = 'underlying_price'  # price of underlying asset
    USD_MULTIPLIER = 'usd_multiplier'  # usd_multiplier = underlying_price or 1.0
    BID_PRICE = 'bid_price'  # best bid
    ASK_PRICE = 'ask_price'   # best ask
    BID_SIZE = 'bid_size'
    ASK_SIZE = 'ask_size'
    MARK_IV = 'mark_iv'  # float
    BID_IV = 'bid_iv'  # float
    ASK_IV = 'ask_iv'  # float
    OPEN_INTEREST = 'open_interest'  # number of open contratcs,  float
    VOLUME = 'volume'  # float
    DELTA = 'delta'  # float
    VEGA = 'vega'  # float
    THETA = 'theta'  # float
    GAMMA = 'gamma'  # float
    EXCHANGE_TIME = 'exchange_time'  # pd.Timestamp  record time at exchange
    # added to slices
    CONTRACT = 'contract'  # string
    STRIKE = 'strike'  # float
    OPTION_TYPE = 'optiontype'  # {'C', 'P'}
    EXPIRY = 'expiry'  # pd.Timestamp
    TTM = 'ttm'  # float


def get_clean_slice(df: pd.DataFrame) -> pd.DataFrame:
    cond = (df[SliceColumn.OPEN_INTEREST] > 0.0) & (df[SliceColumn.MARK_PRICE] > 0.0) \
           & (df[SliceColumn.BID_PRICE] > 0.0) & (df[SliceColumn.ASK_PRICE] > 0.0)  \
           & (df[SliceColumn.UNDERLYING_PRICE] > 0.0)
    clean_df = df[cond]
    return clean_df


@dataclass
class ExpirySlice:
    """
    slice_t of call and puts options data per expiry
    """
    undelying_data: pd.Series
    options: pd.DataFrame

    def __post_init__(self):

        if IS_FORCE_ALL_COLUMNS_DATA:
            ALL_COLUMNS = list(x.value for x in SliceColumn)
            all_options_col = np.all(np.in1d(self.options.columns, ALL_COLUMNS, assume_unique=True))
            if not all_options_col:
                raise ValueError(f"missing column data "
                                 f"{self.options.columns[np.in1d(self.options.columns, ALL_COLUMNS, assume_unique=True)]}")

        self.expiry_id = self.undelying_data[UnderlyingColumn.EXPIRY_ID]
        self.value_time = self.undelying_data[UnderlyingColumn.VALUE_TIME]
        self.expiry_time = self.undelying_data[UnderlyingColumn.EXPIRY]
        self.future_price = self.undelying_data[UnderlyingColumn.FUTURE_PRICE]
        self.ttm = self.undelying_data[UnderlyingColumn.TTM]
        # for easy of searching contracts by strike, reindex call and puts using strikes
        self.calls = self.options.loc[self.options[SliceColumn.OPTION_TYPE.value] == 'C', :].sort_values(
            by=SliceColumn.STRIKE).set_index(SliceColumn.STRIKE, drop=False)
        self.puts = self.options.loc[self.options[SliceColumn.OPTION_TYPE.value] == 'P', :].sort_values(
            by=SliceColumn.STRIKE).set_index(SliceColumn.STRIKE, drop=False)

    def print(self) -> None:
        print(f"expiry_id={self.expiry_id},\n"
              f"future_price={self.future_price},\n"
              f"value_time={self.value_time},\n"
              f"expiry_time={self.expiry_time}\n")
        print(f"calls=\n{self.calls}")
        print(f"puts=\n{self.puts}")

    def get_future_price(self) -> float:
        return self.future_price

    def get_ttm(self) -> float:
        return self.ttm

    def get_usd_mark(self) -> pd.Series:
        """
        for coined options mark_price is in BTC/ETH, multiply by underlying future price
        """
        return self.options[SliceColumn.MARK_PRICE].multiply(self.options[SliceColumn.USD_MULTIPLIER])

    def get_call_slice(self,
                       is_filtered: bool = True,
                       min_strike: float = None,
                       index_by_contract: bool = False
                       ) -> pd.DataFrame:
        if is_filtered:
            df = get_clean_slice(df=self.calls)
        else:
            df = self.calls
        if min_strike is not None:
            df = df.loc[df[SliceColumn.STRIKE] >= min_strike, :]
        if df.empty:
            print(f"empty slice call data at {self.expiry_id} on {self.value_time}")
        if not df.empty and index_by_contract is True:
            df = df.set_index(SliceColumn.CONTRACT, drop=False)
        return df

    def get_put_slice(self,
                      is_filtered: bool = True,
                      max_strike: float = None,
                      index_by_contract: bool = False
                      ) -> pd.DataFrame:
        if is_filtered:
            df = get_clean_slice(df=self.puts)
        else:
            df = self.puts
        if max_strike is not None:
            df = df.loc[df[SliceColumn.STRIKE] <= max_strike, :]
        if df.empty:
            print(f"empty slice put data at {self.expiry_id} on {self.value_time}")
        if not df.empty and index_by_contract is True:
            df = df.set_index(SliceColumn.CONTRACT, drop=False)
        return df

    def get_joint_slice(self,
                        delta_bounds: Tuple[Optional[float], Optional[float]] = None,
                        is_filtered: bool = True
                        ) -> pd.DataFrame:
        """
        get put/call ivols slice_t
        """
        put_wing = self.get_put_slice(is_filtered=is_filtered)
        call_wing = self.get_call_slice(is_filtered=is_filtered)
        if put_wing.empty or call_wing.empty:
            print(f"empty joint slice at {self.expiry_id} on {self.value_time}")
            joint_slice = pd.DataFrame()
        else:
            put_wing = put_wing.loc[put_wing[SliceColumn.STRIKE] < self.future_price, :]
            if delta_bounds is not None and delta_bounds[0] is not None:
                put_wing = put_wing.loc[put_wing[SliceColumn.DELTA] < delta_bounds[0], :]
            call_wing = call_wing.loc[call_wing[SliceColumn.STRIKE] >= self.future_price, :]
            if delta_bounds is not None and delta_bounds[1] is not None:
                call_wing = call_wing.loc[call_wing[SliceColumn.DELTA] > delta_bounds[1], :]
            joint_slice = pd.concat([put_wing, call_wing], axis=0)  # stitch together
        return joint_slice

    def get_bid_mark_ask_vols(self,
                              is_delta_space: bool = False,
                              delta_bounds: Tuple[Optional[float], Optional[float]] = None,
                              is_filtered: bool = True,
                              drop_same_deltas: bool = True  # in delta space deep oot options end up with same delta
                              ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.get_joint_slice(delta_bounds=delta_bounds, is_filtered=is_filtered)
        if is_delta_space:
            df1 = df[[SliceColumn.DELTA, SliceColumn.BID_IV, SliceColumn.MARK_IV, SliceColumn.ASK_IV, SliceColumn.STRIKE]].set_index(
                SliceColumn.DELTA)
            if drop_same_deltas:
                df1.index = np.around(df1.index, decimals=4)
                df1 = df1[~df1.index.duplicated(keep='first')]
        else:
            df1 = df[[SliceColumn.STRIKE, SliceColumn.BID_IV, SliceColumn.MARK_IV, SliceColumn.ASK_IV]].set_index(
                SliceColumn.STRIKE, drop=False)

        strikes = df1[SliceColumn.STRIKE]
        df1 = df1.drop(SliceColumn.STRIKE, axis=1)
        return df1, strikes

    def get_atm_option_strike(self, nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI) -> float:
        joint_slice = self.get_joint_slice(is_filtered=True)
        a = joint_slice[SliceColumn.STRIKE].to_numpy()
        if nearest_strike_selection == NearestStrikeSelection.MAX_OI:
            weight = joint_slice[SliceColumn.OPEN_INTEREST].to_numpy()
        else:
            weight = None
        idx = find_idx_nearest_element(value=self.future_price, a=a, weight=weight, nearest_strike_selection=nearest_strike_selection)
        return joint_slice.index[idx]

    def get_atm_call_id(self, nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI) -> str:
        strike = self.get_atm_option_strike(nearest_strike_selection=nearest_strike_selection)
        return self.calls.loc[strike, SliceColumn.CONTRACT]

    def get_atm_put_id(self, nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI) -> str:
        strike = self.get_atm_option_strike(nearest_strike_selection=nearest_strike_selection)
        return self.puts.loc[strike, SliceColumn.CONTRACT]

    def get_put_delta_strike(self,
                             delta: float = -0.25,
                             nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                             ) -> float:
        put_wing = self.get_put_slice(is_filtered=True)
        a = put_wing[SliceColumn.DELTA].to_numpy()
        if nearest_strike_selection == NearestStrikeSelection.MAX_OI:
            weight = put_wing[SliceColumn.OPEN_INTEREST].to_numpy()
        else:
            weight = None
        idx = find_idx_nearest_element(value=delta, a=a, weight=weight, nearest_strike_selection=nearest_strike_selection)
        return put_wing.index[idx]

    def get_put_delta_option_id(self,
                                delta: float = -0.25,
                                nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                                ) -> str:
        strike = self.get_put_delta_strike(delta=delta, nearest_strike_selection=nearest_strike_selection)
        return self.puts.loc[strike, SliceColumn.CONTRACT]

    def get_call_delta_strike(self,
                              delta: float = 0.25,
                              nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                              ) -> float:
        call_wing = self.get_call_slice(is_filtered=True)
        a = call_wing[SliceColumn.DELTA].to_numpy()
        if nearest_strike_selection == NearestStrikeSelection.MAX_OI:
            weight = call_wing[SliceColumn.OPEN_INTEREST].to_numpy()
        else:
            weight = None
        idx = find_idx_nearest_element(value=delta, a=a, weight=weight, nearest_strike_selection=nearest_strike_selection)
        return call_wing.index[idx]

    def get_call_delta_option_id(self, delta: float = 0.25,
                                 nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                                 ) -> str:
        strike = self.get_call_delta_strike(delta=delta, nearest_strike_selection=nearest_strike_selection)
        return self.calls.loc[strike, SliceColumn.CONTRACT]

    def get_slice_open_interest(self) -> pd.DataFrame:
        calls = self.calls[SliceColumn.OPEN_INTEREST].rename('C')
        puts = self.puts[SliceColumn.OPEN_INTEREST].rename('P')
        df = pd.concat([puts, calls], axis=1)
        return df


@dataclass
class SlicesChain:
    """
    collections of slices in time
    """
    options_df: pd.DataFrame  #  all options data index = all contracts, columns = SliceColumn
    undelying_df: pd.DataFrame  # spot, rate ttm index= slice ids, columns = UnderlyingColumn
    expiry_slices: Dict[str, ExpirySlice]
    value_time: pd.Timestamp

    def __post_init__(self):
        # sort by maturities
        mats = {key: eslice.get_ttm() for key, eslice in self.expiry_slices.items()}
        sorted_dds_dict = dict(sorted(mats.items(), key=lambda item: item[1]))
        self.expiry_slices = {k: self.expiry_slices[k] for k in sorted_dds_dict.keys()}  # sorded {Slice_id: ExpirySlice}
        self.expiry_maturities = {k: v.get_ttm() for k, v in self.expiry_slices.items()}  # {Slice_id: ttm}
        self.expiry_times = {v.expiry_time: k for k, v in self.expiry_slices.items()}  # {expiry time: slice id}  # useful to find sices by maturity dates

    def print_slices_id(self) -> None:
        for key in self.expiry_slices.keys():
            print(key)

    def get_slice_id_for_mat_date(self, mat_date: pd.Timestamp, is_normalized: bool = False) -> Optional[str]:
        if is_normalized:  # can find maturity without hour and tz info
            mat_date = mat_date.normalize().tz_convert(None)
            expiry_times = {k.normalize().tz_convert(None): v for k, v in self.expiry_times.items()}  # {expiry time: slice id}  # useful to find sices by maturity dates
        else:
            expiry_times = self.expiry_times
        if mat_date in expiry_times.keys():
            return expiry_times[mat_date]
        else:
            return None

    def get_slice_id_for_mat_dates(self, mat_dates: Union[List[pd.Timestamp], pd.DatetimeIndex],
                                   is_apply_open_interest_filter: bool = False,
                                   is_normalized: bool = True  # does not depend on expiry date
                                   ) -> List[str]:
        slice_ids = []
        for mat_date in mat_dates:
            slice_id = self.get_slice_id_for_mat_date(mat_date=mat_date, is_normalized=is_normalized)
            # by default remove slices with less than 1000 - typical for just rolled contracts
            if slice_id is not None and is_apply_open_interest_filter:
                slice_t = self.expiry_slices[slice_id].get_joint_slice(is_filtered=True)
                if slice_t.empty:
                    slice_id = None
                else:
                    if np.nansum(slice_t[SliceColumn.OPEN_INTEREST]) < 1000.0:
                        slice_id = None
            if slice_id is not None:
                slice_ids.append(slice_id)
        return slice_ids

    def get_next_slice_after_date(self, mat_date: pd.Timestamp) -> str:
        ttm = get_ttm_from_dates(mat_date=mat_date, value_time=self.value_time)
        a = np.array(list(self.expiry_maturities.values()))
        idx = np.searchsorted(a=a, v=ttm, side='left')
        return list(self.expiry_maturities.keys())[idx]

    def get_atm_call_id(self,
                        mat_date: pd.Timestamp = None,
                        slice_id: str = None,
                        nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                        ) -> str:
        if slice_id is None and mat_date is None:
            raise ValueError(f"provide slice_id or mat_date")
        if slice_id is None:
            slice_id = self.get_slice_id_for_mat_date(mat_date)
        slice_t = self.expiry_slices[slice_id]
        return slice_t.get_atm_call_id(nearest_strike_selection=nearest_strike_selection)

    def get_atm_put_id(self,
                       mat_date: pd.Timestamp = None,
                       slice_id: str = None,
                       nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                       ) -> str:
        if slice_id is None and mat_date is None:
            raise ValueError(f"provide slice_id or mat_date")
        if slice_id is None:
            slice_id = self.get_slice_id_for_mat_date(mat_date)
        slice_t = self.expiry_slices[slice_id]
        return slice_t.get_atm_put_id(nearest_strike_selection=nearest_strike_selection)

    def get_put_delta_option_id(self,
                                mat_date: pd.Timestamp = None,
                                slice_id: str = None,
                                delta: float = -0.25,
                                nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                                ) -> str:
        if slice_id is None and mat_date is None:
            raise ValueError(f"provide slice_id or mat_date")
        if slice_id is None:
            slice_id = self.get_slice_id_for_mat_date(mat_date)
        return self.expiry_slices[slice_id].get_put_delta_option_id(delta=delta, nearest_strike_selection=nearest_strike_selection)

    def get_call_delta_option_id(self, mat_date: pd.Timestamp = None, slice_id: str = None, delta: float = 0.25,
                                 nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                                 ) -> str:
        if slice_id is None and mat_date is None:
            raise ValueError(f"provide slice_id or mat_date")
        if slice_id is None:
            slice_id = self.get_slice_id_for_mat_date(mat_date)
        return self.expiry_slices[slice_id].get_call_delta_option_id(delta=delta, nearest_strike_selection=nearest_strike_selection)

    def get_contract_data(self, contract: str) -> pd.Series:
        contract_data = self.options_df.loc[contract, :]
        return contract_data

    def get_contract_execution_price(self, contract: str,
                                     size: float,
                                     is_trade_at_bid_ask: bool = True,
                                     is_usd: bool = False
                                     ) -> Tuple[float, float]:
        contract_data = self.options_df.loc[contract, :]
        mid_price = contract_data[SliceColumn.MARK_PRICE]
        if is_trade_at_bid_ask:
            if size > 0.0:
                execution_price = contract_data[SliceColumn.ASK_PRICE]
            else:
                execution_price = contract_data[SliceColumn.BID_PRICE]
        else:
            execution_price = mid_price
        slippage = np.abs(execution_price-mid_price)

        if is_usd:
            execution_price *= contract_data[SliceColumn.USD_MULTIPLIER]
            slippage *= contract_data[SliceColumn.USD_MULTIPLIER]

        return execution_price, slippage

    def generate_delta_vol_matrix(self,
                                  value_time: pd.Timestamp,
                                  days_map: Dict[str, int] = {'1w': 7, '1m': 30},
                                  deltas: List[float] = (-0.10, -0.25, 0.50, 0.25, 0.10),
                                  nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.ABOVE,
                                  ) -> Optional[DeltaVolMatrix]:
        """
        generate table snapshot of vol data: index= days, columns = deltas
        """
        strikes_matrix, option_ids_matrix, vols_matrix, deltas_matrix, underlying_prices_matrix = {}, {}, {}, {}, {}
        labels = {}
        for label, day in days_map.items():
            next_date = value_time + pd.DateOffset(days=day)   # if overlapping next date will be last avilable maturity
            slice_date = self.get_next_slice_after_date(mat_date=next_date)
            slice_t = self.expiry_slices[slice_date]

            df = slice_t.get_joint_slice(is_filtered=True, delta_bounds=(-0.1, 0.1))

            if len(df.index) >= len(deltas):  # filter out slices with no data
                labels[label] = pd.Series(label, name=next_date)
                vols, option_ids, strikes, strike_deltas, index_prices = [], [], [], [], []
                for delta in deltas:
                    if delta < 0.0:
                        strike = slice_t.get_put_delta_strike(delta=delta, nearest_strike_selection=nearest_strike_selection)
                        slice_df = slice_t.get_put_slice()
                    else:
                        strike = slice_t.get_call_delta_strike(delta=delta, nearest_strike_selection=nearest_strike_selection)
                        slice_df = slice_t.get_call_slice()
                    vols.append(slice_df.loc[strike, SliceColumn.MARK_IV])
                    option_ids.append(slice_df.loc[strike, SliceColumn.CONTRACT])
                    strikes.append(strike)
                    strike_deltas.append(slice_df.loc[strike, SliceColumn.DELTA])
                    index_prices.append(slice_df.loc[strike, SliceColumn.UNDERLYING_PRICE])

                vols_matrix[label] = pd.Series(vols, index=deltas, name=next_date)
                strikes_matrix[label] = pd.Series(strikes, index=deltas, name=next_date)
                option_ids_matrix[label] = pd.Series(option_ids, index=deltas, name=next_date)
                deltas_matrix[label] = pd.Series(strike_deltas, index=deltas, name=next_date)
                underlying_prices_matrix[label] = pd.Series(index_prices, index=deltas, name=next_date)
        vols_matrix = pd.DataFrame.from_dict(vols_matrix, orient='index')
        option_ids_matrix = pd.DataFrame.from_dict(option_ids_matrix, orient='index')
        strikes_matrix = pd.DataFrame.from_dict(strikes_matrix, orient='index')
        deltas_matrix = pd.DataFrame.from_dict(deltas_matrix, orient='index')
        underlying_prices_matrix = pd.DataFrame.from_dict(underlying_prices_matrix, orient='index')

        labels = pd.DataFrame.from_dict(labels, orient='index')
        if not labels.empty:
            labels = labels.rename({0: 'tenor'}, axis=1)
            labels['days'] = labels.iloc[:, 0].map(days_map)
            labels = labels.sort_values('days')

            delta_vol_matrix = DeltaVolMatrix(value_time=value_time,
                                              labels=labels,
                                              vols_matrix=vols_matrix,
                                              option_ids_matrix=option_ids_matrix,
                                              strikes_matrix=strikes_matrix,
                                              deltas_matrix=deltas_matrix,
                                              underlying_prices_matrix=underlying_prices_matrix)
        else:
            delta_vol_matrix = None
        return delta_vol_matrix

    def plot_total_oi_in_maturity(self,
                                  title: str = None,
                                  is_usd: bool = True,
                                  ax: plt.Subplot = None,
                                  **kwargs
                                  ) -> None:
        totals_by_mat = []
        for slice_id in self.expiry_slices.keys():
            eslice = self.expiry_slices[slice_id]
            df = eslice.get_slice_open_interest()
            if is_usd:
                df = df*eslice.future_price
            totals_by_mat.append(df.sum(0).rename(slice_id))

        totals_by_mat = pd.concat(totals_by_mat, axis=1).T
        totals_by_mat_all = totals_by_mat.sum(axis=0)
        totals_by_mat = totals_by_mat.rename({'P': f'Puts total = {totals_by_mat_all[0]:,.0f}',
                                              'C': f'Calls total = {totals_by_mat_all[1]:,.0f}'},
                                             axis=1)

        qis.plot_bars(totals_by_mat,
                  stacked=True,
                  yvar_format='{:,.0f}',
                  colors=['orangered', 'green'],
                  legend_loc='upper center',
                  title=title,
                  x_rotation=90,
                  ax=ax,
                  **kwargs)

    def plot_total_oi_in_strikes(self,
                                 title: str = None,
                                 is_usd: bool = True,
                                 ax: plt.Subplot = None,
                                 **kwargs
                                 ) -> None:
        calls_by_strike = {}
        puts_by_strike = {}
        for slice_id in self.expiry_slices.keys():
            eslice = self.expiry_slices[slice_id]
            df = eslice.get_slice_open_interest()
            if is_usd:
                df = eslice.future_price * df
            for idx, row in df.iterrows():
                if idx in calls_by_strike.keys():
                    calls_by_strike[idx] += row['C']
                    puts_by_strike[idx] += row['P']
                else:
                    calls_by_strike[idx] = row['C']
                    puts_by_strike[idx] = row['P']

        calls_by_strike = pd.Series(calls_by_strike, name='C')
        puts_by_strike = pd.Series(puts_by_strike, name='P')
        totals_by_strike = pd.concat([puts_by_strike, calls_by_strike], axis=1).sort_index()
        totals_by_mat_all = totals_by_strike.sum(axis=0)
        totals_by_strike = totals_by_strike.rename({'P': f'Puts total = {totals_by_mat_all[0]:,.0f}',
                                                    'C': f'Calls total = {totals_by_mat_all[1]:,.0f}'},
                                                   axis=1)
        qis.plot_bars(totals_by_strike,
                  stacked=True,
                  yvar_format='{:,.0f}',
                  colors=['orangered', 'green'],
                  legend_loc='upper center',
                  title=title,
                  x_rotation=90,
                  ax=ax,
                  **kwargs)

    def save_joint_slices(self, file_name: str) -> None:
        slice_dfs = {}
        for key, slite_t in self.expiry_slices.items():
            slice_dfs[key] = slite_t.get_joint_slice()
        qis.save_df_dict_to_feather(dfs=slice_dfs, file_name=file_name)


@dataclass
class DeltaVolMatrix:
    """
    collections of slices in time and delta
    """
    value_time: pd.Timestamp
    labels: pd.DataFrame  # day
    vols_matrix: pd.DataFrame
    option_ids_matrix: pd.DataFrame
    strikes_matrix: pd.DataFrame
    deltas_matrix: pd.DataFrame
    underlying_prices_matrix: pd.DataFrame

    def print(self):
        print(self.labels)
        print(self.vols_matrix)
        print(self.option_ids_matrix)
        print(self.strikes_matrix)
        print(self.deltas_matrix)
        print(self.underlying_prices_matrix)

    def get_melted_matrix(self,
                          mat_id: str = 'tenor',
                          is_tenor: bool = True
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        datas = {'vols': self.vols_matrix,
                 'options': self.option_ids_matrix,
                 'strikes': self.strikes_matrix,
                 'underlying_prices_matrix': self.underlying_prices_matrix}
        out = {}
        if is_tenor:
            xdata = self.labels[mat_id]
        else:
            xdata = pd.Series(self.labels.index, index=self.labels.index, name=mat_id)
        for key, df in datas.items():
            df1 = qis.melt_scatter_data_with_xdata(df=df,
                                               xdata=xdata,
                                               y_column=key,
                                               hue_name='delta')
            df1.index = [f"{delta:0.2f}d_{day}" for day, delta in zip(df1[mat_id], df1['delta'])]
            out[key] = df1
        return out['vols'], out['strikes'], out['options'], out['underlying_prices_matrix']

    def plot_vol_matrix_table(self, ax: plt.Subplot = None, **kwargs):

        # data_colors = put.compute_heatmap_colors(a=self.vols_matrix.to_numpy(), **kwargs)
        df = qis.df_to_str(df=self.vols_matrix, var_format='{:,.0%}')
        qis.plot_df_table(df=df,
                          first_column_width=None,
                          index_column_name='Tenor/\nDelta',
                          # data_colors=data_colors,
                          # heatmap_rows=[x for x in range(0, len(df.columns)-1)],
                          special_columns_colors=[(0, 'steelblue'), (3, 'lightblue')],
                          ax=ax,
                          **kwargs)

    def plot_vol_in_strike(self,
                           title: str = 'ETH implied vols at strikes for delta grid = {-0.10, -0.25, 0.50, 0.25, 0.10}',
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> None:
        vols, strikes, _, _ = self.get_melted_matrix(is_tenor=False)
        vols['strikes'] = strikes['strikes']
        vols['delta'] = vols['delta'].apply(lambda x: str(x))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        colors = qis.get_cmap_colors(n=len(vols['tenor'].unique()))
        sns.lineplot(data=vols, x='strikes', y='vols', hue='tenor', style='tenor', palette=colors, markers=False,
                     dashes=False, ax=ax)

        markers = qis.get_n_markers(n=len(vols['delta'].unique()))
        sns.scatterplot(data=vols, x="strikes", y="vols", hue="delta", style="delta", markers=markers, ax=ax)

        qis.set_ax_ticks_format(ax=ax, xvar_format='{:,.0f}', yvar_format='{:,.0%}', **kwargs)
        qis.set_title(ax=ax, title=title, **kwargs)

        qis.set_legend(ax=ax, **kwargs)

        h, l = ax.get_legend_handles_labels()
        l1 = ax.legend(h[:int(len(h) / 2)], l[:int(len(l) / 2)], loc='upper left', title='Option Tenor', fontsize=14)
        l2 = ax.legend(h[int(len(h) / 2):], l[int(len(l) / 2):], loc='upper right', title='Deltas', ncol=5, fontsize=14)
        ax.add_artist(l1)   # we need this because the 2nd call to legend() erases the first


def find_idx_nearest_element(value: float, a: np.ndarray, weight: np.ndarray = None,
                             nearest_strike_selection: NearestStrikeSelection = NearestStrikeSelection.MAX_OI
                             ) -> int:
    idx_sorted = np.argsort(a)
    sorted_array = a[idx_sorted]
    if weight is not None:
        weight = weight[idx_sorted]

    idx = np.searchsorted(sorted_array, value, side="left")
    if idx >= len(a):
        idx_nearest = idx_sorted[len(a)-1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if nearest_strike_selection == NearestStrikeSelection.MAX_OI and weight is not None:  # use biggest weight
            if weight[idx-1] > weight[idx]:
                idx_nearest = idx_sorted[idx-1]
            else:
                idx_nearest = idx_sorted[idx]

        elif nearest_strike_selection == NearestStrikeSelection.BELOW:
            if sorted_array[idx-1] > sorted_array[idx]:
                idx_nearest = idx_sorted[idx - 1]
            else:
                idx_nearest = idx_sorted[idx]

        elif nearest_strike_selection == NearestStrikeSelection.ABOVE:
            if sorted_array[idx-1] < sorted_array[idx]:
                idx_nearest = idx_sorted[idx - 1]
            else:
                idx_nearest = idx_sorted[idx]

        else:  # use smallest distance
            if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
                idx_nearest = idx_sorted[idx-1]
            else:
                idx_nearest = idx_sorted[idx]
    return idx_nearest

"""
collector of aligned time series data for options
needed to run backtest
options time series data gen
"""
# built in
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional
from enum import Enum
import qis
from qis import TimePeriod

# analytics
from sigma_strats.option_chain_analytics.option_chain import SliceColumn
import sigma_strats.option_chain_analytics.data.config as gu
from sigma_strats.option_chain_analytics.data.config import OptionTickerConfig, FutureTickerConfig, get_ttm_from_dates, DeltaType

# pricers
import sigma_strats.option_chain_analytics.pricers.normal as nor


@dataclass
class TsData:
    """
    aligned df collection for options and futures time series
    data_dict must contain mark_price
    """
    data_dict: Dict[str, pd.DataFrame]
    spot_data: pd.DataFrame
    ticker: str

    def __post_init__(self):
        self.prices: pd.DataFrame = self.data_dict['mark_price']
        self.spot_data: pd.DataFrame = self.spot_data#.reindex(index=self.prices.index, method='ffill')
        self.ewm_vol: Optional[pd.Series] = None

    def print(self):
        for key, df in self.data_dict.items():
            print(key)
            print(df)

    def get_spot_data(self, time_period: TimePeriod = None) -> pd.DataFrame:
        df = self.spot_data
        if time_period is not None:
            df = time_period.locate(df)
        return df

    def get_time_slice(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """
         get non-nan time slice from series
         return is df(index=contracts, colums=[mark_rpice, mark_vol,...])
         """
        if timestamp in self.prices.index:
            dfs = {}
            for key, df in self.data_dict.items():
                dfs[key] = df.loc[timestamp, :].dropna()
            df = pd.DataFrame.from_dict(dfs, orient='columns')
            df.index.name = 'contract'
        else:
            print(f"no slice data for {timestamp}")
            df = pd.DataFrame()
        return df

    def get_spot_price(self,
                       value_time: pd.Timestamp = None,
                       index: Union[pd.DatetimeIndex, pd.Index] = None
                       ) -> Union[float, pd.Series]:
        value = self.spot_data['spot'].rename(self.ticker)
        if value_time is not None:
            if value_time in value.index:
                value = value[value_time]
            else:
                raise KeyError(f"in get_spot_price {value_time} not in {value.index}")
        elif index is not None:
            value = value.reindex(index=index, method='ffill').fillna(method='ffill')
        return value

    def get_ewm_vol(self,
                    value_time: pd.Timestamp,
                    span: float = 168  # =7*24
                    ) -> float:
        if self.ewm_vol is None:
            returns = np.log(self.spot_data['close']).diff()
            self.ewm_vol = qis.compute_ewm_vol(data=returns, span=span, af=24.0 * 365.0)
        idx = self.ewm_vol.index.get_indexer([value_time], method='ffill')
        vol = self.ewm_vol.iloc[idx].to_numpy()[0]
        return vol

    def get_prices(self,
                   contracts: List[str],
                   time_period: TimePeriod = None,
                   is_ffillna: bool = True
                   ) -> pd.DataFrame:
        price_data = self.prices[contracts]
        if time_period is not None:
            price_data = time_period.locate(price_data)
        if is_ffillna:
            price_data = price_data.fillna(method='ffill')
        return price_data

    def get_contracts_data(self,
                           data: str = 'mark_price',
                           contracts: Union[List[str], pd.Index] = None,
                           time_period: TimePeriod = None,
                           value_time: pd.Timestamp = None,
                           is_ffillna: bool = False
                           ) -> pd.DataFrame:
        contracts_data = self.data_dict[data]
        if contracts is not None:
            contracts_data = contracts_data[contracts]
        if time_period is not None:
            contracts_data = time_period.locate(contracts_data)
        elif value_time is not None:
            contracts_data = contracts_data.loc[value_time, :]
        if is_ffillna:
            contracts_data = contracts_data.fillna(method='ffill')
        return contracts_data


@dataclass
class FuturesDataDFs(TsData):
    """
    implementation of options time series data
    """

    future_ticker_config: FutureTickerConfig = field(default=FutureTickerConfig.CMS, init=True)

    class FuturesDataColumns(str, Enum):
        MARK_PRICE = 'mark_price'
        OPEN = 'open'
        HIGH = 'high'
        LOW = 'low'
        CLOSE = 'close'
        VOLUME = 'usd_volume'
        VOLUME_CONTRACTS = 'contract_count'
        OI_VOLUME = 'oi_value_usd'
        OI_CONTRACTS = 'oi_contract_count'

    def get_term_structure(self,
                           value_time: pd.Timestamp,
                           data: str = 'mark_price'
                           ) -> pd.DataFrame:

        contracts_data = self.get_contracts_data(value_time=value_time, data=data).dropna().rename(data).to_frame()
        ttms = [gu.get_ttm_from_future_ticker(value_time=value_time, contract=x) for x in contracts_data.index]
        contracts_data['ttms'] = ttms
        return contracts_data


@dataclass
class OptionsDataDFs(TsData):
    """
    implementation of options time series data
    fields of SliceColumn are contained in data_dict
    """
    # do not pass to TsData
    option_ticker_config: OptionTickerConfig = field(default=OptionTickerConfig.CMS, init=True)

    def get_contracts_delta(self, contracts: List[str],
                            delta_type: DeltaType = DeltaType.MARKED,
                            time_period: TimePeriod = None
                            ) -> pd.DataFrame:
        if delta_type == DeltaType.MARKED:
            contract_deltas = self.get_contracts_data(contracts=contracts,
                                                      data=SliceColumn.DELTA,
                                                      time_period=time_period,
                                                      is_ffillna=True)
        elif delta_type == DeltaType.NORMAL:
            contract_deltas = {}
            for contract in contracts:
                contract_deltas[contract] = self.compute_normal_vol_delta(contract=contract, time_period=time_period)
            contract_deltas = pd.DataFrame.from_dict(contract_deltas, orient='columns')

        else:
            raise NotImplementedError

        return contract_deltas

    def compute_normal_vol_delta(self,
                                 contract: str,
                                 time_period: TimePeriod = None
                                 ) -> pd.Series:

        mat_str, strike, option_type = gu.split_option_contract_ticker(contract=contract,
                                                                       option_ticker_config=self.option_ticker_config)
        mat_date = gu.mat_to_timestamp(mat_str)

        mark_price = self.get_contracts_data(contracts=[contract], time_period=time_period, data=SliceColumn.MARK_PRICE, is_ffillna=True)
        index_price = self.get_contracts_data(contracts=[contract], time_period=time_period, data=SliceColumn.UNDERLYING_PRICE, is_ffillna=True)
        ttms = [get_ttm_from_dates(mat_date=mat_date, value_time=x) for x in mark_price.index]
        delta = np.zeros(len(index_price.index))
        for idx, (forward, given_price, t) in enumerate(zip(index_price.to_numpy(), mark_price.to_numpy(), ttms)):
            # mark price need to be converted t usd
            if not np.isnan(forward) and not np.isnan(given_price):
                normal_vol = nor.infer_normal_implied_vol(forward=forward, ttm=t, strike=strike,
                                                          given_price=forward * given_price, optiontype=option_type)
                delta[idx] = nor.compute_normal_delta(ttm=t, forward=forward, strike=strike, vol=normal_vol,
                                                      optiontype=option_type)
        normal_delta = pd.Series(delta, index=index_price.index, name='delta')
        return normal_delta

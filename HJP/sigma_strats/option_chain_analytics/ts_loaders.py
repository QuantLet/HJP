"""
add prop data loaders here
the return type is Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame]]: where str is SliceColumn
"""
import pandas as pd
from typing import Dict, Union, Optional, Any
from enum import Enum

from sigma_strats.prop.cms_db import load_contract_ts_data_db


class DataSource(Enum):
    AWS_CMS_LOCAL = 1
    AWS_CMS = 2


def ts_data_loader_wrapper(data_source: DataSource = DataSource.AWS_CMS_LOCAL,
                           ticker: str = 'BTC',
                           freq: str = 'D',
                           hour_offset: Optional[int] = 8,
                           **kwargs
                           ) -> Dict[str, Any]:
    """
    generic wrapper for loading
    """
    if data_source == DataSource.AWS_CMS_LOCAL:
        return load_contract_ts_data_db(ticker=ticker, freq=freq, hour_offset=hour_offset)
    else:
        raise NotImplementedError(f"{data_source}")

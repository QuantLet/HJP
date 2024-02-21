import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis
from typing import Dict
from enum import Enum

# analytics
from sigma_strats.prop.cms_db import load_contract_ts_data_db
from sigma_strats.option_chain_analytics.data.chain_loader_from_dfs import create_chain_from_from_options_dfs
from sigma_strats.option_chain_analytics.ts_data import OptionsDataDFs
import sigma_strats.option_chain_analytics.visuals.slices as vis
from sigma_strats.option_chain_analytics.option_chain import SlicesChain

FIG_SIZE = (8.3, 11.7)  # A4 for portrait


def run_chain_report(chain: SlicesChain) -> Dict[str, plt.Figure]:
    figs = {}
    configs = {'Unrestricted': dict(delta_bounds=None, is_filtered=False),
               'Deltas > 0.1': dict(delta_bounds=(-0.1, 0.1), is_filtered=True)}
    for expiry, eslice in chain.expiry_slices.items():
        if eslice.get_ttm() > 0.0:
            for key, vals in configs.items():
                fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
                fig.suptitle(f"slice id={eslice.expiry_id}, future price={eslice.future_price:,.2f} - {key}",
                             fontweight="bold", fontsize=10, color='blue')
                gs = fig.add_gridspec(nrows=2, ncols=1, wspace=0.0, hspace=0.0)
                with sns.axes_style("darkgrid"):
                    ax = fig.add_subplot(gs[0, 0])
                    vis.plot_slice_vols_with_oi(eslice=eslice, title=f"{eslice.expiry_id} Vols In Strike Space",
                                                is_delta_space=False,
                                                ax=ax, **vals)
                    ax = fig.add_subplot(gs[1, 0])
                    vis.plot_slice_vols_with_oi(eslice=eslice, title=f"{eslice.expiry_id} Vols in Delta Space",
                                                is_delta_space=True,
                                                ax=ax, **vals)
                figs[f"{expiry}_{key}"] = fig

    plt.close('all')

    return figs


class UnitTests(Enum):
    RUN_CHAIN_REPORT = 1


def run_unit_test(unit_test: UnitTests):

    import sigma_strats.local_path as local_path

    ticker = 'BTC'
    options_data_dfs = OptionsDataDFs(**load_contract_ts_data_db(ticker=ticker, hour_offset=8))

    if unit_test == UnitTests.RUN_CHAIN_REPORT:
        value_time = pd.Timestamp('2023-02-06 08:00:00+00:00')
        chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)
        figs = run_chain_report(chain=chain)

        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"chain_report_{value_time:%Y%m%dT%H%M%S}",
                             orientation='landscape',
                             local_path=local_path.get_output_path())

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.RUN_CHAIN_REPORT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

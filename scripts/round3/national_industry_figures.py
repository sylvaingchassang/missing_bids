from scripts.round3.figures_import_helper_r3 import *

# before/after collusion scandal industry comparisons, using national data

list_data_sets = [
    ('Bridges', 'bc_collusion.csv'),
    ('Electric', 'ec_collusion.csv'),
    ('Pre-Stressed Concrete', 'pc_collusion.csv'),
    ('Floods', 'fc_collusion.csv')
]

deviations = all_deviations
for industry, file in list_data_sets:
    print('='*20 + '\n' + industry)
    print('collecting and processing data')
    data = asymptotics.MultistagePIDMeanAuctionData(
        os.path.join(path_data, file))
    data_before = asymptotics.MultistagePIDMeanAuctionData.from_clean_bids(
        data.df_bids.loc[data.data.before == 1])
    data_after = asymptotics.MultistagePIDMeanAuctionData.from_clean_bids(
        data.df_bids.loc[data.data.before.isnull()])
    plot_delta(data=data_before, filename='R3/{}_delta_before'.format(
        industry))
    plot_delta(data=data_after, filename='R3/{}_delta_after'.format(industry))

    print('computing before/after problem solution')
    solutions_before, _ = compute_asymptotic_multistage_solution(
        data_before, deviations)
    solutions_after, _ = compute_asymptotic_multistage_solution(
        data_after, deviations)

    solutions_before_90, _ = compute_asymptotic_multistage_solution_90(
        data_before, deviations)
    solutions_after_90, _ = compute_asymptotic_multistage_solution_90(
        data_after, deviations)
    del data
    del data_before
    del data_after

    print('saving plot\n')
    pretty_plot(
        os.path.join('R3', industry),
        np.array([1 - solutions_before, 1 - solutions_after]),
        np.array(["before investigation", "after investigation"]),
        xlabel='minimum markup',
        xticks=r3_min_markups
    )
    pretty_plot(
        os.path.join('R3', industry + '_90'),
        np.array([1 - solutions_before_90, 1 - solutions_after_90]),
        np.array(["before investigation", "after investigation"]),
        xlabel='minimum markup',
        xticks=r3_min_markups
    )

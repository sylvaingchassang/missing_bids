from scripts.round3.figures_import_helper_r3 import *

# before/after collusion scandal industry comparisons, using national data

list_data_sets = [
    ('Bridges', 'bc_collusion.csv'),
    ('Electric', 'ec_collusion.csv'),
    ('Pre-Stressed Concrete', 'pc_collusion.csv'),
    ('Floods', 'fc_collusion.csv')
]


for industry, file in list_data_sets:
    print('='*20 + '\n' + industry)
    print('collecting and processing data')
    data = asymptotics.MultistageAsymptoticAuctionData(
        os.path.join(path_data, file))
    data_before = asymptotics.MultistageAsymptoticAuctionData.from_clean_bids(
        data.df_bids.loc[data.data.before == 1])
    data_after = asymptotics.MultistageAsymptoticAuctionData.from_clean_bids(
        data.df_bids.loc[data.data.before.isnull()])
    plot_delta(data=data_before, filename='R4/{}_delta_before'.format(
        industry))
    plot_delta(data=data_after, filename='R4/{}_delta_after'.format(industry))

    for deviations in [all_deviations, all_deviations_small_sample]:
        print('computing before/after problem solution')
        solutions_before, _ = compute_asymptotic_multistage_solution(
            data_before, deviations)
        solutions_after, _ = compute_asymptotic_multistage_solution(
            data_after, deviations)

        solutions_before_10, _ = compute_asymptotic_multistage_solution_10(
            data_before, deviations)
        solutions_after_10, _ = compute_asymptotic_multistage_solution_10(
            data_after, deviations)

        print('saving plot\n')
        pretty_plot(
            os.path.join('R4', industry + '_'.join(map(str, deviations))),
            np.array([1 - solutions_before, 1 - solutions_after]),
            np.array(["before investigation", "after investigation"]),
            xlabel='minimum markup',
            xticks=r3_min_markups
        )
        pretty_plot(
            os.path.join('R4', industry + '_10_' +
                         '_'.join(map(str, deviations))),
            np.array([1 - solutions_before_10, 1 - solutions_after_10]),
            np.array(["before investigation", "after investigation"]),
            xlabel='minimum markup',
            xticks=r3_min_markups
        )

    del data
    del data_before
    del data_after



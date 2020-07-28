from scripts.round3.figures_import_helper_r3 import *

# illustrating impact of different IC constraints, using city data

print('data located at \n\t{}'.format(path_data))
print('plots saved at \n\t{}\n'.format(path_figures))

print('=' * 20 + '\n' + 'Tsuchiura (before min. price)')
print('collecting and processing data')
tsuchiura_data = auction_data.AuctionData(
    os.path.join(path_data, 'tsuchiura_data.csv'))
plot_delta(tsuchiura_data, filename='R3/tsuchiura_delta')

tsuchiura_before_min_price_ = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])

tsuchiura_before_min_price = auction_data.AuctionData.from_clean_bids(
    tsuchiura_before_min_price_.df_bids.loc[
        tsuchiura_before_min_price_.df_bids.norm_bid > .9])

plot_delta(tsuchiura_before_min_price,
           filename='R3/tsuchiura_delta_no_min_price_bids_above_80pct')

tsuchiura_after_min_price_ = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[~tsuchiura_data.data.minprice.isnull()])

tsuchiura_after_min_price = auction_data.AuctionData.from_clean_bids(
    tsuchiura_after_min_price_.df_bids.loc[
        tsuchiura_after_min_price_.df_bids.norm_bid > .9])

plot_delta(tsuchiura_after_min_price,
           filename='R3/tsuchiura_delta_with_min_price_bids_above_80pct')

print('computing solutions for different deviations, no min price')

solutions_all_deviations, share_ties = compute_efficient_solution_parallel(
    tsuchiura_before_min_price, all_deviations_tsuchiura)
solutions_up_deviations, _ = compute_efficient_solution_parallel(
    tsuchiura_before_min_price, up_deviations_tsuchiura)
solutions_down_deviations, _ = compute_efficient_solution_parallel(
    tsuchiura_before_min_price, down_deviations)

share_comp_all_deviations = 1 - solutions_all_deviations
share_comp_up_deviations = 1 - solutions_up_deviations
share_comp_down_deviations = 1 - solutions_down_deviations

solutions_all_devs_with_min_price, share_min_price = \
    compute_efficient_solution_parallel(
        tsuchiura_after_min_price, all_deviations_tsuchiura)

share_comp_min_price = 1 - share_min_price - (
        1 - share_min_price) * solutions_all_devs_with_min_price

share_comp_all_deviations_w_ties = share_comp_all_deviations * (
    1 - share_ties)
share_comp_up_deviations_w_ties = share_comp_up_deviations * (1 - share_ties)
share_comp_up_deviations_wo_ties = share_comp_up_deviations + \
                                  share_ties * solutions_up_deviations

print('saving plot 1\n')
pretty_plot(
    'R3/Tsuchiura -- different deviations -- no minimum price',
    [share_comp_up_deviations_wo_ties,
     share_comp_up_deviations_w_ties,
     share_comp_all_deviations_w_ties],
    ['upward dev.', 'upward dev. and ties',
     'upward, downward deviations and ties'], ['k.:', 'k.--', 'k.-'],
    xlabel='minimum markup',
    xticks=r3_min_markups)


print('saving plot 1b\n')
pretty_plot(
    'R3/Tsuchiura -- different deviations -- no minimum price (b)',
    [share_comp_up_deviations_wo_ties,
     share_comp_up_deviations_w_ties,
     share_comp_down_deviations,
     share_comp_all_deviations_w_ties],
    ['upward dev', 'upward dev and ties', 'downward dev',
     'upward, downward dev and ties'], ['k.:', 'k.--', 'k.-', 'k.-.'],
    xlabel='minimum markup',
    xticks=r3_min_markups)

print('saving plot 2\n')
pretty_plot(
    'R3/Tsuchiura with and without min price',
    [share_comp_all_deviations_w_ties, share_comp_min_price],
    ['without minimum price', 'with minimum price'],
    ['k.-', 'k.:'],
    xlabel='minimum markup',
    xticks=r3_min_markups)

# computing deviation temptation over profits, using city data

print('='*20 + '\n' + 'Tsuchiura, deviation temptation')
print('collecting and processing data')

min_deviation_temptation_solver = ComputeMinimizationSolution(
    solver_cls=solvers.ParallelSolver,
    constraints=r3_constraints,
    metric=analytics.NormalizedDeviationTemptation
)


print('solving for min temptation')
dev_gain, _ = min_deviation_temptation_solver(
    tsuchiura_before_min_price, all_deviations_tsuchiura)

print('saving plot\n')
pretty_plot(
    'R3/Tsuchiura -- Deviation Gain', [dev_gain],
    [None], max_y=.15,
    ylabel='deviation temptation',
    xlabel='minimum markup',
    xticks=r3_min_markups,
    expect_decreasing=False
)

print('saving data\n')
save2frame([dev_gain],
           ['min_m={}'.format(m) for m in r3_min_markups],
           'R3/tsuchiura_deviation_temptation',
           ['deviation gains'])

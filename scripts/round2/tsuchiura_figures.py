from scripts.figures_import_helper import *

# illustrating impact of different IC constraints, using city data

print('data located at \n\t{}'.format(path_data))
print('plots saved at \n\t{}\n'.format(path_figures))

print('='*20 + '\n' + 'Tsuchiura (before min. price)')
print('collecting and processing data')
tsuchiura_data = auction_data.AuctionData(
    os.path.join(path_data, 'tsuchiura_data.csv'))
plot_delta(tsuchiura_data, filename='R2/tsuchiura_delta')

tsuchiura_before_min_price_ = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])

tsuchiura_before_min_price = auction_data.AuctionData.from_clean_bids(
    tsuchiura_before_min_price_.df_bids.loc[
        tsuchiura_before_min_price_.df_bids.norm_bid > .8])

plot_delta(tsuchiura_before_min_price,
           filename='R2/tsuchiura_delta_no_min_price_bids_above_80pct')

tsuchiura_after_min_price_ = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[~tsuchiura_data.data.minprice.isnull()])

tsuchiura_after_min_price = auction_data.AuctionData.from_clean_bids(
    tsuchiura_after_min_price_.df_bids.loc[
        tsuchiura_after_min_price_.df_bids.norm_bid > .8])

plot_delta(tsuchiura_after_min_price,
           filename='R2/tsuchiura_delta_with_min_price_bids_above_80pct')

all_deviations = [-.02, .0, .0008]
up_deviations = [.0, .0008]

print('computing solutions for different deviations, no min price')

solutions_all_deviations, share_ties = compute_solution_parallel(
    tsuchiura_before_min_price, all_deviations)
solutions_up_deviations, _ = compute_solution_parallel(
    tsuchiura_before_min_price, up_deviations)
share_comp_all_deviations = 1 - solutions_all_deviations
share_comp_up_deviations = 1 - solutions_up_deviations

solutions_all_devs_with_min_price, share_min_price = \
    compute_solution_parallel(tsuchiura_after_min_price, all_deviations)

share_comp_min_price = 1 - share_min_price - (
        1 - share_min_price) * solutions_all_devs_with_min_price

share_comp_all_deviations_w_ties = share_comp_all_deviations * (
    1 - share_ties)
share_comp_up_deviations_w_ties = share_comp_up_deviations * (1 - share_ties)
share_comp_up_deviations_wo_ties = share_comp_up_deviations + \
                                  share_ties * solutions_up_deviations

print('saving plot 1\n')
pretty_plot(
    'R2/Tsuchiura -- different deviations -- no minimum price',
    [share_comp_up_deviations_wo_ties,
     share_comp_up_deviations_w_ties,
     share_comp_all_deviations_w_ties],
    ['upward dev.', 'upward dev. and ties',
     'upward, downward deviations and ties'], ['k.:', 'k.--', 'k.-'],
    xlabel='minimum markup',
    xticks=r2_min_mkps)

print('saving plot 2\n')
pretty_plot(
    'R2/Tsuchiura with and without min price',
    [share_comp_all_deviations_w_ties, share_comp_min_price],
    ['without minimum price', 'with minimum price'],
    ['k.-', 'k.:'],
    xlabel='minimum markup',
    xticks=r2_min_mkps)

# computing deviation temptation over profits, using city data

print('='*20 + '\n' + 'Tsuchiura, deviation temptation')
print('collecting and processing data')

min_deviation_temptation_solver = ComputeMinimizationSolution(
    solver_cls=solvers.ParallelSolver,
    constraint_func=round2_constraints,
    metric=analytics.DeviationTemptationOverProfits,
    #project_choices=[True] * len(r2_min_mkps)
)


print('solving for min temptation')
dev_gain, _ = min_deviation_temptation_solver(
    tsuchiura_before_min_price, all_deviations)

print('saving plot\n')
pretty_plot(
    'R2/Tsuchiura -- Deviation Gain', [dev_gain],
    ['before minimum price'], max_y=.05,
    ylabel='deviation temptation as a share of profits',
    xlabel='minimum markup',
    xticks=r2_min_mkps)

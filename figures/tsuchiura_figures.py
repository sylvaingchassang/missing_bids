from figures.figures_import_helper import *

# illustrating impact of different IC constraints, using city data

print('='*20 + '\n' + 'Tsuchiura (before min. price)')
print('collecting and processing data')
tsuchiura_data = auction_data.AuctionData(
    os.path.join(path_data, 'tsuchiura_data.csv'))

tsuchiura_before_min_price = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])

all_deviations = [-.02, .0, .0008]
up_deviations = [.0, .0008]

print('computing different problem solutions')
solutions_all_deviations, share_ties = compute_minimization_solution(
    tsuchiura_before_min_price, all_deviations, .25)
solutions_up_deviations, _ = compute_minimization_solution(
    tsuchiura_before_min_price, up_deviations, .25)

solutions_all_deviations_w_ties = solutions_all_deviations * (1 - share_ties)
solutions_up_deviations_w_ties = solutions_up_deviations * (1 - share_ties)
solutions_up_deviations_wo_ties = solutions_up_deviations + share_ties * (
    1 - solutions_up_deviations)

print('saving plot\n')
pretty_plot(
    'Tsuchiura (before min price)',
    [solutions_up_deviations_wo_ties,
     solutions_up_deviations_w_ties,
     solutions_all_deviations_w_ties],
    ['upward dev.', 'upward dev. and ties',
     'upward, downward deviations and ties'],
    ['k.:', 'k.--', 'k.-']
)

# computing deviation temptation over profits, using city data

print('='*20 + '\n' + 'Tsuchiura, deviation temptation')
print('collecting and processing data')
min_deviation_temptation_solver = ComputeMinimizationSolution(
    metric=analytics.DeviationTemptationOverProfits,
    list_ks=[0.01, 0.5, 1, 1.5],
    project_choices=[True, False, False, False]
)

print('solving for min temptation')
dev_gain, _ = min_deviation_temptation_solver(
    tsuchiura_before_min_price, all_deviations)

print('saving plot')
pretty_plot('Tsuchiura', [dev_gain],
            ['before minimum price'],
            list_ks=[0.01, 0.5, 1, 1.5],
            ylabel='deviation temptation as a share of profits')

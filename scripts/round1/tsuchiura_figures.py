from scripts.figures_import_helper import *

# illustrating impact of different IC constraints, using city data

print('data located at \n\t{}'.format(path_data))
print('plots saved at \n\t{}\n'.format(path_figures))

print('='*20 + '\n' + 'Tsuchiura (before min. price)')
print('collecting and processing data')
tsuchiura_data = auction_data.AuctionData(
    os.path.join(path_data, 'tsuchiura_data.csv'))

tsuchiura_before_min_price = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])

all_deviations = [-.02, .0, .0008]
up_deviations = [.0, .0008]

print('computing different problem solutions')
compute_minimization_solution_tsuchiura = ComputeMinimizationSolution(
    constraint_func=round1_constraints(.25))
solutions_all_deviations, share_ties = compute_minimization_solution_tsuchiura(
    tsuchiura_before_min_price, all_deviations)
solutions_up_deviations, _ = compute_minimization_solution_tsuchiura(
    tsuchiura_before_min_price, up_deviations)
share_comp_all_deviations = 1 - solutions_all_deviations
share_comp_up_deviations = 1 - solutions_up_deviations


share_comp_all_deviations_w_ties = share_comp_all_deviations * (
    1 - share_ties)
share_comp_up_deviations_w_ties = share_comp_up_deviations * (1 - share_ties)
share_comp_up_deviations_wo_ties = share_comp_up_deviations + \
                                  share_ties * solutions_up_deviations

print('saving plot\n')
pretty_plot(
    'R1/Tsuchiura -- Share of Competitive Histories',
    [share_comp_up_deviations_wo_ties,
     share_comp_up_deviations_w_ties,
     share_comp_all_deviations_w_ties],
    ['upward dev.', 'upward dev. and ties',
     'upward, downward deviations and ties'],
    ['k.:', 'k.--', 'k.-']
)

# computing deviation temptation over profits, using city data

print('='*20 + '\n' + 'Tsuchiura, deviation temptation')
print('collecting and processing data')


def tsuchiura_round1_constraints(demands):
    return markup_info_constraints(
        max_markups=(.5,), ks=(0.01, 0.5, 1, 1.5), demands=demands)


min_deviation_temptation_solver = ComputeMinimizationSolution(
    metric=analytics.DeviationTemptationOverProfits,
    constraint_func=tsuchiura_round1_constraints,
    project_choices=[True, False, False, False]
)

print('solving for min temptation')
dev_gain, _ = min_deviation_temptation_solver(
    tsuchiura_before_min_price, all_deviations)

print('saving plot\n')
pretty_plot('R1/Tsuchiura -- Deviation Gain', [dev_gain],
            ['before minimum price'],
            xticks=(0.01, 0.5, 1, 1.5),
            ylabel='deviation temptation as a share of profits',
            max_y=.03)

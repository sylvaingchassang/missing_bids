from scripts.round3.figures_import_helper_r3 import *

# illustrating impact of different IC constraints, using city data

print('data located at \n\t{}'.format(path_data))
print('plots saved at \n\t{}\n'.format(path_figures))

filename = 'tsuchiura_data.csv'
tsuchiura_data = asymptotics.PIDMeanAuctionData(
    os.path.join(path_data, filename))
tsuchiura_data = asymptotics.PIDMeanAuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])

plot_delta(tsuchiura_data)

filename = 'municipal_pub_reserve_no_pricefloor.csv'
other_data = asymptotics.PIDMeanAuctionData(os.path.join(path_data, filename))

# +
s1 = set(other_data.df_bids.pid)
s2 = set(tsuchiura_data.df_bids.pid)

assert len(s2.intersection(s1)) == 0

# +
all_bids = pd.concat((other_data.df_bids, tsuchiura_data.df_bids), axis=0)
data_low = asymptotics.PIDMeanAuctionData.from_clean_bids(
    all_bids.loc[all_bids.norm_bid < .9])
data_high = asymptotics.PIDMeanAuctionData.from_clean_bids(
    all_bids.loc[all_bids.norm_bid > .9])


plot_delta(data_high,
           filename='R3/all_cities_delta_bids above_90pct')


all_deviations = [-.02, .0, .001]
up_deviations = [.0, .001]

print('computing solutions for different deviations, no min price')

solutions_all_deviations, share_ties = compute_asymptotic_solution(
    data_high, all_deviations)
solutions_up_deviations, _ = compute_asymptotic_solution(
    data_high, up_deviations)
share_comp_all_deviations = 1 - solutions_all_deviations
share_comp_up_deviations = 1 - solutions_up_deviations


share_comp_all_deviations_w_ties = share_comp_all_deviations * (
    1 - share_ties)
share_comp_up_deviations_w_ties = share_comp_up_deviations * (1 - share_ties)
share_comp_up_deviations_wo_ties = share_comp_up_deviations + \
                                  share_ties * solutions_up_deviations

print('saving plot 1\n')
pretty_plot(
    'R3/city data -- different deviations -- no minimum price',
    [share_comp_up_deviations_wo_ties,
     share_comp_up_deviations_w_ties,
     share_comp_all_deviations_w_ties],
    ['upward dev.', 'upward dev. and ties',
     'upward, downward deviations and ties'], ['k.:', 'k.--', 'k.-'],
    xlabel='minimum markup',
    xticks=r3_markups)


# computing deviation temptation over profits, using city data

print('='*20 + '\n' + 'all cities, deviation temptation')
print('collecting and processing data')

min_deviation_temptation_solver = ComputeMinimizationSolution(
    solver_cls=asymptotics.ParallelAsymptoticSolver,
    metric=analytics.NormalizedDeviationTemptation
)


print('solving for min temptation')
dev_gain, _ = min_deviation_temptation_solver(
    data_high, all_deviations)

print('saving plot\n')
pretty_plot(
    'R3/All cities -- Deviation Gain', [dev_gain],
    [None], max_y=.15,
    ylabel='deviation temptation',
    xlabel='minimum markup',
    xticks=r3_markups,
    expect_decreasing=False
)

print('saving data\n')
save2frame([dev_gain],
           ['min_m={}'.format(m) for m in r3_markups],
           'R3/all_cities_deviation_temptation',
           ['deviation gains'])

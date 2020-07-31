from scripts.round3.figures_import_helper_r3 import *
# %matplotlib inline

# before/after industry comparisons, using national data

filename = 'municipal_pub_reserve_no_pricefloor.csv'

# +
filter_ties = auction_data.FilterTies(tolerance=.0001)

unfiltered_data = asymptotics.PIDMeanAuctionData(os.path.join(path_data, filename))

data = filter_ties(unfiltered_data)
plot_delta(data)
# -

data.df_bids.head()

num_auctions_by_bidder = unfiltered_data.data.groupby('bidder_id').size()
top30_bidders = num_auctions_by_bidder.sort_values(ascending=False).head(30)
top30_bidders = top30_bidders.index

deviations = all_deviations
list_solutions = []
for i, bidder in enumerate(top30_bidders):    
    print('firm {}'.format(i + 1))
    data_firm = asymptotics.PIDMeanAuctionData.from_clean_bids(
        data.df_bids.loc[unfiltered_data.data.bidder_id == bidder])
    demand_firm = data_firm.assemble_target_moments(deviations)

    constraints = [environments.EmptyConstraint()]
    metric = analytics.EfficientIsNonCompetitive
    metric.min_markup, metric.max_markup = .02, .5

    min_collusion_solver = asymptotics.ParallelAsymptoticSolver(
        data=data_firm,
        deviations=deviations,
        metric=metric,
        plausibility_constraints=constraints,
        num_points=NUM_POINTS,
        seed=0,
        project=False,
        filter_ties=None,
        num_evaluations=NUM_EVAL,
        confidence_level=.95,
        moment_matrix=moment_matrix,
        moment_weights=None,
        enhanced_guesses=True
    )
    
    list_solutions.append([i, bidder, 1 -
                           min_collusion_solver.result.solution])

print('saving data\n')
save2frame(list_solutions,
           ['rank', 'bidder_id', 'share competitive'],
           'R3/city_auctions_individual_firms')



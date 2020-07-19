from scripts.figures_import_helper import *
# %matplotlib inline

# before/after industry comparisons, using national data

filename = 'municipal_pub_reserve_no_pricefloor.csv'

# +
filter_ties = auction_data.FilterTies(tolerance=.0001)

unfiltered_data = auction_data.AuctionData(os.path.join(path_data, filename))

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
    data_firm = auction_data.AuctionData.from_clean_bids(
        data.df_bids.loc[unfiltered_data.data.bidder_id == bidder])
    demand_firm = data_firm.assemble_target_moments(deviations)
    constraints = [environments.MarkupConstraint(
        max_markup=.5, min_markup=.02)]

    min_collusion_solver = solvers.ParallelSolver(
        data=data_firm,
        deviations=deviations,
        metric=analytics.IsNonCompetitive,
        plausibility_constraints=constraints,
        num_points=NUM_POINTS,
        seed=0,
        project=False,
        filter_ties=None,
        num_evaluations=NUM_EVAL,
        confidence_level=1 - .05 / len(deviations),
        moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
        moment_weights=np.identity(len(deviations))
    )
    
    list_solutions.append([i, bidder, 1 -
                           min_collusion_solver.result.solution])

print('saving data\n')
save2frame(list_solutions,
           ['rank', 'bidder_id', 'share competitive'],
           'R2/city_auctions_individual_firms')



import time
from scripts.figures_import_helper import *
from analytics import IsNonCompetitive

# illustrating impact of different IC constraints, using city data

start = time.time()

tsuchiura_data = auction_data.AuctionData(
    os.path.join(path_data, 'tsuchiura_data.csv'))

tsuchiura_before_min_price = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])

deviations = [-.02, .0, .0008]

demands = [tsuchiura_before_min_price.get_counterfactual_demand(rho)
           for rho in deviations]


constraints = [environments.MarkupConstraint(max_markup=.5, min_markup=.02),
               environments.InformationConstraint(k=1, sample_demands=demands)]

this_solver = solvers.MinCollusionIterativeSolver(
    data=tsuchiura_before_min_price,
    deviations=deviations,
    metric=IsNonCompetitive,
    plausibility_constraints=constraints,
    num_points=200,
    seed=0,
    project=True,
    filter_ties=None,
    number_iterations=20,
    confidence_level=.05,
    moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
    moment_weights=np.identity(len(deviations))
)

print(this_solver.result.solution)
end = time.time()
print('compute time {}s'.format(end - start))

from os import path
import auction_data
import analytics
import environments
import numpy as np
import rebidding as rb
import solvers
from scripts.figures_import_helper import path_data

print('\n>>> \tsetting up data\n')

FC_COLLUSION_PATH = path.join(path_data, 'fc_collusion.csv')

tsuchiura_data = auction_data.AuctionData(
    path.join('..', 'tests', 'reference_data', 'tsuchiura_data.csv'))
tsuchiura_before_min_price = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])


filter_ties = auction_data.FilterTies(tolerance=.0001)
share_ties = filter_ties.get_ties(tsuchiura_before_min_price).mean()
print('share of ties in the unfiltered data {}'.format(share_ties))

filtered_data = filter_ties(tsuchiura_before_min_price)
print('share of ties in the filtered data: {}'.format(
    filter_ties.get_ties(filtered_data).mean()))


deviations = [-.03, .0, .0008]
demands = [filtered_data.get_counterfactual_demand(rho) for rho in deviations]
print(demands)


constraints = [environments.MarkupConstraint(max_markup=.5, min_markup=.05),
               environments.InformationConstraint(k=.5, sample_demands=demands)]

print('\n>>> \texample without rebidding\n')

min_collusion_solver = solvers.MinCollusionIterativeSolver(
    data=tsuchiura_before_min_price,
    deviations=deviations,
    metric=analytics.IsNonCompetitive,
    plausibility_constraints=constraints,
    num_points=1000.0,
    seed=0,
    project=True,
    filter_ties=None,
    number_iterations=50,
    confidence_level=.98,
    moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
    moment_weights=np.identity(3)
)

print('solver demands: {}'.format(min_collusion_solver.solver.demands))
print('rough tolerance T^2: {}'.format(4. / tsuchiura_before_min_price.df_bids.shape[0]))
print('bootstrapped tolerance T^2: {}'.format(
    min_collusion_solver.solver.tolerance))
print('joint confidence: {}'.format(
    min_collusion_solver.solver.joint_confidence))

result = min_collusion_solver.result
print('minimum share of collusive auctions: {}'.format(result.solution))

print('\n>>> \texample with rebidding\n')


# this example requires some normalized bids to be greater than 1

multistage_raw_data = tsuchiura_before_min_price.raw_data.copy()
multistage_raw_data.loc[:, 'norm_bid'] = multistage_raw_data.norm_bid * 1.01

multistage_data = rb.MultistageAuctionData(multistage_raw_data)

rb.MultistageIsNonCompetitive.max_win_prob = 1.

min_collusion_solver = solvers.MinCollusionIterativeSolver(
    data=multistage_data,
    deviations=deviations,
    metric=rb.MultistageIsNonCompetitive,
    plausibility_constraints=constraints,
    num_points=1000.0,
    seed=0,
    project=True,
    filter_ties=None,
    number_iterations=50,
    confidence_level=.98,
    moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
    moment_weights=np.identity(3)
)


print('solver demands: {}'.format(min_collusion_solver.solver.demands))
print('rough tolerance T^2: {}'.format(4. / tsuchiura_before_min_price.df_bids.shape[0]))
print('bootstrapped tolerance T^2: {}'.format(
    min_collusion_solver.solver.tolerance))
print('joint confidence: {}'.format(
    min_collusion_solver.solver.joint_confidence))

result = min_collusion_solver.result
print('minimum share of collusive auctions: {}'.format(result.solution))

print('\n>>> \texample rebidding without downward deviations\n')

deviations = [.0, .001]

multistage_fc_data = rb.MultistageAuctionData(FC_COLLUSION_PATH)

multistage_fc_data_before = rb.MultistageAuctionData(
    multistage_fc_data.df_bids.loc[multistage_fc_data.data.before == 1])

multistage_demands_fc_before = [
    multistage_fc_data_before.get_counterfactual_demand(rho) for rho in
    deviations]

constraints = [
    environments.MarkupConstraint(max_markup=.5, min_markup=.05),
    environments.InformationConstraint(
        k=.5, sample_demands=multistage_demands_fc_before)
]

min_collusion_solver = solvers.MinCollusionIterativeSolver(
    data=multistage_fc_data_before,
    deviations=deviations,
    metric=rb.MultistageIsNonCompetitive,
    plausibility_constraints=constraints,
    num_points=10000.0,
    seed=0,
    project=False,
    filter_ties=None,
    number_iterations=50,
    confidence_level=.95,
    moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
    moment_weights=np.identity(2)
)
share = min_collusion_solver.result.solution
print('share non collusive: {}'.format(1 - share))

print('\n>>> \texample rebidding with more refined IC\n')


deviations = [-.02, .0, .001]

multistage_fc_data = rb.RefinedMultistageData(FC_COLLUSION_PATH)

multistage_fc_data_before = rb.RefinedMultistageData(
    multistage_fc_data.df_bids.loc[multistage_fc_data.data.before == 1])

multistage_demands_fc_before = \
    multistage_fc_data_before.assemble_target_moments(deviations)

constraints = [
    environments.MarkupConstraint(max_markup=.5, min_markup=.05),
    environments.InformationConstraint(
        k=.5, sample_demands=multistage_demands_fc_before)
]

min_collusion_solver = rb.IteratedRefinedMultistageSolver(
    data=multistage_fc_data_before,
    deviations=deviations,
    metric=rb.RefinedMultistageIsNonCompetitive,
    plausibility_constraints=constraints,
    num_points=1000.0,
    seed=0,
    project=True,
    filter_ties=None,
    number_iterations=25,
    confidence_level=.95,
    moment_matrix=rb.refined_moment_matrix(),
    moment_weights=np.identity(5)
)
min_collusion_solver.max_best_sol_index = 250
share = min_collusion_solver.result.solution
print('share non collusive: {}'.format(1 - share))

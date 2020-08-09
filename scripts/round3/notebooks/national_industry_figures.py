from scripts.round3.figures_import_helper_r3 import *

file = 'fc_collusion.csv'
data = asymptotics.MultistagePIDMeanAuctionData(
    os.path.join(path_data, file))
data_before = asymptotics.MultistagePIDMeanAuctionData.from_clean_bids(
    data.df_bids.loc[data.data.before == 1])
data_after = asymptotics.MultistagePIDMeanAuctionData.from_clean_bids(
    data.df_bids.loc[data.data.before.isnull()])

empty_constraints = [environments.EmptyConstraint()]

# +
metric = rebidding.EfficientMultistageIsNonCompetitive
metric.min_markup = .05
metric.max_markup = .5

solver = asymptotics.ParallelAsymptoticMultistageSolver(
    data=data_after,
    deviations=all_deviations,
    metric=metric,
    plausibility_constraints=empty_constraints,
    num_points=1000,
    seed=0,
    project=False,
    filter_ties=None,
    num_evaluations=2,
    confidence_level=.95,
    moment_matrix=multistage_moment_matrix,
    moment_weights=None,
    enhanced_guesses=True
)

print('solving for min collusion')

# print('flood min collusion: ', solver.result.solution)

solutions_after, _ = compute_asymptotic_multistage_solution(
        data_after, all_deviations)

print('sim results: ', solutions_after)
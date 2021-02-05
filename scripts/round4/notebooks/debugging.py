from scripts.round3.figures_import_helper_r3 import *

national_data = asymptotics.MultistagePIDMeanAuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

# +
ms_moment_matrix = np.array([
    [-.98, 1, .5, 1, 0],
    [1, -1, -.5, -1, 0],
    [0, 0, 0, 0, -1],
    [0, 0, 0, 1, -1]
])

metric = rebidding.EfficientMultistageIsNonCompetitive
metric.max_markup = .5
metric.min_markup = .025

deviations = [-.02, 0, .001]
solver = asymptotics.ParallelAsymptoticMultistageSolver(
    data=national_data,
    deviations=deviations,
    metric=metric,
    plausibility_constraints=[environments.EmptyConstraint()],
    num_points=500,
    num_evaluations=2,
    seed=0,
    project=False,
    filter_ties=None,
    confidence_level=[.999, .999, .974, .974],
    moment_matrix=ms_moment_matrix,
    enhanced_guesses=True
)

print('demands', solver.get_solver(0).demands)

print('tolerance', solver.get_solver(0).tolerance)

print('solution', solver.result.solution)

# +
ms_moment_matrix = np.array([
    [-.98, 1, .5, 1, 0],
    [1, -1, -.5, -1, 0]
])
metric = rebidding.EfficientMultistageIsNonCompetitive
metric.max_markup = .5
metric.min_markup = .15

deviations = [-.02, 0, 1e-9]
solver = asymptotics.ParallelAsymptoticMultistageSolver(
    data=national_data,
    deviations=deviations,
    metric=metric,
    plausibility_constraints=[environments.EmptyConstraint()],
    num_points=100,
    num_evaluations=2,
    seed=0,
    project=False,
    filter_ties=None,
    confidence_level=.95,
    moment_matrix=ms_moment_matrix,
    enhanced_guesses=True
)

print('demands', solver.get_solver(0).demands)

print('tolerance', solver.get_solver(0).tolerance)

print('solution', solver.result.solution)
# -

argmin = solver.result.argmin

argmin

# +
ms_moment_matrix = np.array([
    [0, 0, 0, 0, -1],
    [0, 0, 0, 1, -1]
])
metric = rebidding.EfficientMultistageIsNonCompetitive
metric.max_markup = .1
metric.min_markup = .025

deviations = [-1e-9, 0, .001]
solver = asymptotics.ParallelAsymptoticMultistageSolver(
    data=national_data,
    deviations=deviations,
    metric=metric,
    plausibility_constraints=[environments.EmptyConstraint()],
    num_points=500,
    num_evaluations=2,
    seed=0,
    project=False,
    filter_ties=None,
    confidence_level=[.999, .951],
    moment_matrix=ms_moment_matrix,
    enhanced_guesses=True
)

print('demands', solver.get_solver(0).demands)

print('tolerance', solver.get_solver(0).tolerance)

print('solution', solver.result.solution)
# -



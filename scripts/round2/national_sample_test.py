from scripts.figures_import_helper import *


# national data

# +
print('='*20 + '\n' + 'National sample')
print('collecting and processing data')
national_data = rebidding.RefinedMultistageData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

filter_ties = auction_data.FilterTies(tolerance=.0001)
filtered_data = filter_ties(national_data)

# +
r2_min_mkps = [.0,  .05, .1]


def round2_constraints(demands):
    return [
        [environments.MarkupConstraint(max_markup=.5, min_markup=min_markup)]
        for min_markup in r2_min_mkps
    ]


# -

list_devs = [
    [-1e-9, 0, .0005] , [-.02, 0, 1e-9], all_deviations]
list_solutions = []

deviations = [-.02, 0, 1e-9]

# +
constraints = [
    environments.MarkupConstraint(max_markup=.5, min_markup=.02)]

min_collusion_solver = rebidding.ParallelRefinedMultistageSolver(
    data=filtered_data,
    deviations=deviations,
    metric=rebidding.RefinedMultistageIsNonCompetitive,
    plausibility_constraints=constraints,
    num_points=3000,
    seed=0,
    project=False,
    filter_ties=None,
    num_evaluations=20,
    confidence_level=1 - .05 / len(deviations),
    moment_matrix=rebidding.refined_moment_matrix(),
    moment_weights=np.identity(5)
)

print('solution', min_collusion_solver.result.solution)
# -

deviations = [-.02, 0, 1e-9]
print('moments', filtered_data.assemble_target_moments(deviations))



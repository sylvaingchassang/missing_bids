from scripts.figures_import_helper import *


# national data

print('='*20 + '\n' + 'National sample')
print('collecting and processing data')
national_data = rebidding.RefinedMultistageData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

# +
r2_min_mkps = [.0,  .05, .1]


def round2_constraints(demands):
    return [
        [environments.MarkupConstraint(max_markup=.5, min_markup=min_markup)]
        for min_markup in r2_min_mkps
    ]


# +
list_devs = [
    [-1e-9, 0, .0005] , [-.02, 0, 1e-9], all_deviations]
list_solutions = []


for deviations in list_devs:
    this_compute_solution = ComputeMinimizationSolution(
        constraint_func=round2_constraints,
        solver_cls=rebidding.ParallelRefinedMultistageSolver,
        metric=rebidding.RefinedMultistageIsNonCompetitive)
    solutions, _ = this_compute_solution(national_data, deviations)
    list_solutions.append(1 - solutions)

print(list_solutions)

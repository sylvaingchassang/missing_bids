from scripts.figures_import_helper import *


# national data

print('='*20 + '\n' + 'National sample')
print('collecting and processing data')
national_data = rebidding.RefinedMultistageData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

plot_delta(national_data, filename='R2/national_data_deltas')

print('computing problem solutions')
deviations = [-.025, .0, .001]
list_coeffs = [.25, .5, .75]
list_solutions = []

for coeff_marginal_info in list_coeffs:
    rebidding.RefinedMultistageIsNonCompetitive.coeff_marginal_info = \
        coeff_marginal_info
    this_compute_solution = ComputeMinimizationSolution(
        constraint_func=round2_constraints,
        solver_cls=rebidding.ParallelRefinedMultistageSolver,
        metric=rebidding.RefinedMultistageIsNonCompetitive)
    solutions, _ = this_compute_solution(
        national_data, deviations)
    list_solutions.append(1 - solutions)

print('saving plot\n')
pretty_plot('R2/national_auctions',
            list_solutions,
            [r"\alpha={}".format(coeff) for coeff in list_coeffs],
            xlabel='m',
            xticks=r2_min_mkps)

print('saving data\n')
save2frame(solutions,
           ['min_m={}'.format(m) for m in [.05, .1, .15, .2]],
           'R2/national_auctions',
           list_coeffs)

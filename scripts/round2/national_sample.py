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
# ComputeMinimizationSolution._NUM_POINTS = 10000
# ComputeMinimizationSolution._NUM_EVAL = 500

RMIsNonComp = rebidding.RefinedMultistageIsNonCompetitive

list_devs = [[-.0001, .001], [-.025, .0001], [-.025, .0, .001]]
list_solutions = []
for deviations in list_devs:
    this_compute_solution = ComputeMinimizationSolution(
        constraint_func=round2_constraints,
        solver_cls=rebidding.ParallelRefinedMultistageSolver,
        metric=RMIsNonComp)
    solutions, _ = this_compute_solution(national_data, deviations)
    list_solutions.append(1 - solutions)


print('saving plot\n')
pretty_plot(
    'R2/national auctions',
    list_solutions,
    [r"deviations {}".format(devs) for devs in list_devs],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r2_min_mkps
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r2_min_mkps],
           'R2/national_auctions')


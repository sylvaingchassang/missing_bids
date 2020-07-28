from scripts.round3.figures_import_helper_r3 import *


# national data

print('='*20 + '\n' + 'National sample')
print('collecting and processing data')
national_data = rebidding.RefinedMultistageData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

plot_delta(national_data, filename='R3/national_data_deltas')

print('computing problem solutions')


list_devs = [
    [-1e-9] + up_deviations, [1e-9] + down_deviations, all_deviations]
list_solutions = []
for deviations in list_devs:
    this_compute_solution = ComputeMinimizationSolution(
        constraints=empty_constraints,
        solver_cls=rebidding.ParallelRefinedMultistageSolver,
        metric=rebidding.EfficientMultistageIsNonCompetitive)
    solutions, _ = this_compute_solution(national_data, deviations)
    list_solutions.append(1 - solutions)


print('saving plot\n')


pretty_plot(
    'R3/national auctions',
    list_solutions,
    [r"deviations {}".format(dev_repr(devs)) for devs in list_devs],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r3_min_markups
)


print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r3_min_markups],
           'R3/national_auctions')


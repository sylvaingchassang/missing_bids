from scripts.figures_import_helper import *


# national data

print('='*20 + '\n' + 'National sample')
print('collecting and processing data')
national_data = asymptotics.MultistageAsymptoticAuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

print('computing problem solutions')


list_devs = [
    [-1e-9] + up_deviations,  down_deviations + [1e-9], all_deviations]
list_solutions = []
for deviations in list_devs:
    this_compute_solution = ComputeMinimizationSolution(
        constraints=empty_constraints,
        solver_cls=asymptotics.ParallelAsymptoticMultistageSolver,
        metric=rebidding.EfficientMultistageIsNonCompetitive)
    solutions, _ = this_compute_solution(national_data, deviations)
    list_solutions.append(1 - solutions)


print('saving plot\n')


pretty_plot(
    'national auctions',
    list_solutions,
    [r"deviations {}".format(dev_repr(devs)) for devs in list_devs],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r3_min_markups
)


print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r3_min_markups],
           'national_auctions')


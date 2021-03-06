from scripts.figures_import_helper import *


# national data

print('='*20 + '\n' + 'National sample')
print('collecting and processing data')
national_data = asymptotics.MultistageAsymptoticAuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))


print('computing problem solutions')
deviations = all_deviations
list_coeffs = [.25, .5, .75]
list_solutions = []

EMIsNonComp = rebidding.EfficientMultistageIsNonCompetitive

for coeff_marginal_info in list_coeffs:
    EMIsNonComp.coeff_marginal_info = coeff_marginal_info
    this_compute_solution = ComputeMinimizationSolution(
        constraints=empty_constraints,
        solver_cls=asymptotics.ParallelAsymptoticMultistageSolver,
        metric=EMIsNonComp)
    print('\t', 'coeff marginal info', EMIsNonComp.coeff_marginal_info)
    solutions, _ = this_compute_solution(
        national_data, deviations)
    list_solutions.append(1 - solutions)

print('saving plot\n')
pretty_plot('national auctions robustness',
            list_solutions,
            [r"$\alpha={}$".format(coeff) for coeff in list_coeffs],
            xlabel='minimum markup',
            mark=np.array(['k.:', 'k.-', 'k.--']),
            xticks=r3_min_markups)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r3_min_markups],
           'national_auctions_robustness',
           list_coeffs)

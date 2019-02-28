import auction_data
import analytics
import environments
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

hist_plot = auction_data.hist_plot
sns.set_style('white')

# getting data directory
path_data = '/home/sylvain/Dropbox/Econ/papiers/gameTheory/missing_bids/' \
            'data/data_for_missing_bids_figures'
# None #path/to/data (if you know it, otherwise, we'll find it below)

if path_data is None:
    name = 'data_for_missing_bids_figures'
    for root, dirs, _ in os.walk('/'):
        if name in dirs:
            path_data = os.path.join(root, name)
            break

print('data located at \n\t{}'.format(path_data))
path_figures = os.path.join(path_data, 'figures')
print('plots saved at \n\t{}'.format(path_figures))

if not os.path.exists(path_figures):
    os.makedirs(path_figures)


# set global optimization parameters
num_points = 30000
number_iterations = 350
confidence_level = .95


class ComputeMinimizationSolution:
    def __init__(self, metric=analytics.IsNonCompetitive,
                 list_ks=np.array([0.5, 1, 1.5, 2]),
                 project_choices=None):
        self.metric = metric
        self.list_ks = list_ks
        if project_choices is None:
            project_choices = [False] * len(list_ks)
        self.project_choices = project_choices

    def __call__(self, data, deviations, max_markup=.5):
        solutions = []
        filter_ties = auction_data.FilterTies(tolerance=.0001)
        filtered_data = filter_ties(data)
        demands = [filtered_data.get_counterfactual_demand(rho)
                   for rho in deviations]
        _share_ties = filter_ties.get_ties(data).mean()

        for k, proj in zip(self.list_ks, self.project_choices):
            constraints = [
                environments.MarkupConstraint(
                    max_markup=max_markup, min_markup=.02),
                environments.InformationConstraint(k=k, sample_demands=demands)
            ]

            this_solver = analytics.MinCollusionIterativeSolver(
                data=filtered_data,
                deviations=deviations,
                metric=self.metric,
                plausibility_constraints=constraints,
                num_points=int(num_points/(1 + 9 * proj)),
                seed=0,
                project=proj,
                filter_ties=None,
                number_iterations=number_iterations,
                confidence_level=confidence_level,
                moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
                moment_weights=np.array((len(deviations)-1) * [0] + [1])
            )

            share_collusive = this_solver.result.solution
            solutions.append(1 - share_collusive)
            del this_solver
        del filtered_data
        return np.array(solutions), _share_ties


compute_minimization_solution = ComputeMinimizationSolution()


def pretty_plot(title, list_solutions, labels, mark=np.array(['k.:', 'k.-']),
                list_ks=np.array([0.5, 1, 1.5, 2]),
                ylabel='share of competitive histories', max_y=1.05):
    plt.title(title)
    for i, (solutions, label) in enumerate(zip(list_solutions, labels)):
        plt.plot(list_ks, solutions, mark[i], label=label)
    plt.legend(loc='lower right')
    plt.axis([list_ks[0], list_ks[-1], 0, max_y])
    plt.xlabel('k')
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(path_figures, '{}.pdf'.format(title)))
    plt.clf()
    # plt.show()

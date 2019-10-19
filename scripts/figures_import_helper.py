import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
from itertools import product

import solvers
import auction_data
import analytics
import environments

hist_plot = auction_data.hist_plot
sns.set_style('white')

# getting data directory
# path/to/data (if you know it, otherwise, we'll find it below)
path_data = None

if path_data is None:
    name = 'data_for_missing_bids_figures'
    for root, dirs, _ in os.walk('/'):
        if name in dirs:
            path_data = os.path.join(root, name)
            break

path_figures = os.path.join(path_data, 'figures',
                            datetime.utcnow().strftime('%Y%m%d'))


def ensure_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


ensure_dir(path_figures)
ensure_dir(os.path.join(path_figures, 'R1'))
ensure_dir(os.path.join(path_figures, 'R2'))


# set global optimization parameters
num_points = 5000 #30000
num_evaluations = 200
number_iterations = 5
number_iterations_individual_firms = 5
confidence_level = .95


def markup_info_constraints(max_markups, ks, demands):
    return [
        [environments.MarkupConstraint(max_markup=max_markup, min_markup=.02),
         environments.InformationConstraint(k=k, sample_demands=demands)]
        for max_markup, k in product(max_markups, ks)
    ]


def round1_constraints(max_markup=.5):
    return lambda demands: markup_info_constraints(
        max_markups=(max_markup,), ks=(0.5, 1, 1.5, 2), demands=demands)


class ComputeMinimizationSolution:
    def __init__(self, metric=analytics.IsNonCompetitive,
                 constraint_func=round1_constraints(),
                 project_choices=None, filtering=True):
        self.metric = metric
        self.constraint_func = constraint_func
        self._project_choices = project_choices
        self.filtering = filtering

    def __call__(self, data, deviations):
        solutions = []
        if self.filtering:
            filter_ties = auction_data.FilterTies(tolerance=.0001)
            this_data = filter_ties(data)
            _share_ties = filter_ties.get_ties(data).mean()
        else:
            this_data = data
            _share_ties = 0
        demands = [this_data.get_counterfactual_demand(rho)
                   for rho in deviations]
        iter_constraints = self.constraint_func(demands)
        project_choices = \
            self._project_choices if self._project_choices is not None else \
            [False] * len(iter_constraints)

        for constraints, proj in zip(iter_constraints, project_choices):
            this_solver = solvers.MinCollusionIterativeSolver(
                data=this_data,
                deviations=deviations,
                metric=self.metric,
                plausibility_constraints=constraints,
                num_points=int(num_points/(1 + 9 * proj)),
                seed=0,
                project=proj,
                filter_ties=None,
                number_iterations=number_iterations,
                confidence_level=1 - .05/len(deviations),
                moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
                moment_weights=np.identity(len(deviations))
            )

            collusion_metric = this_solver.result.solution
            solutions.append(collusion_metric)
            del this_solver
        if self.filtering:
            del this_data
        del data
        return np.array(solutions), _share_ties


compute_minimization_solution = ComputeMinimizationSolution()
compute_minimization_solution_unfiltered = ComputeMinimizationSolution(
    filtering=False)


def pretty_plot(title, list_solutions, labels, mark=np.array(['k.:', 'k.-']),
                xticks=(0.5, 1, 1.5, 2),
                ylabel='share of competitive histories',
                xlabel='k',
                max_y=1.05):
    plt.title(title)
    for i, (solutions, label) in enumerate(zip(list_solutions, labels)):
        plt.plot(xticks, solutions, mark[i], label=label)
    plt.legend(loc='lower right')
    plt.axis([xticks[0], xticks[-1], 0, max_y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(path_figures, '{}.pdf'.format(title)))
    plt.clf()


def save2frame(data, columns, title):
    pd.DataFrame(data=data, columns=columns).to_csv(
        os.path.join(path_figures, '{}.csv'.format(title)), index=False)

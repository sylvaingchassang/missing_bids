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
import rebidding


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
NUM_POINTS = 2000
NUM_EVAL = 200 #2
NUM_ITER_FIRMS = 5
CONFIDENCE_LEVEL = .95


def markup_info_constraints(max_markups, ks, demands):
    return [
        [environments.MarkupConstraint(max_markup=max_markup, min_markup=.02),
         environments.InformationConstraint(k=k, sample_demands=demands)]
        for max_markup, k in product(max_markups, ks)
    ]


def round1_constraints(demands):
    return markup_info_constraints(
        max_markups=(.5,), ks=(0.5, 1, 1.5, 2), demands=demands)


def round2_constraints(demands):
    return [
        [environments.MarkupConstraint(max_markup=.5, min_markup=min_markup)]
        for min_markup in [.05, .1, .15, .2]
    ]


class ComputeMinimizationSolution:
    def __init__(
            self, metric=analytics.IsNonCompetitive,
            constraint_func=round1_constraints, project_choices=None,
            filtering=True, seed=0, solver_cls=None):
        self.metric = metric
        self.constraint_func = constraint_func
        self._project_choices = project_choices
        self.filtering = filtering
        self.seed = seed
        self.solver_cls = solver_cls or solvers.IteratedSolver

    def __call__(self, data, deviations):
        solutions = []
        _share_ties, this_data = self._apply_filter(data)
        demands = this_data.assemble_target_moments(deviations)
        iter_constraints = self.constraint_func(demands)
        project_choices = \
            self._project_choices or [False] * len(iter_constraints)

        for constraints, proj in zip(iter_constraints, project_choices):
            this_solver = self.get_solver(
                this_data, deviations, constraints, proj)
            solutions.append(this_solver.result.solution)
            del this_solver

        del this_data
        del data
        return np.array(solutions), _share_ties

    def _apply_filter(self, data):
        if self.filtering:
            filter_ties = auction_data.FilterTies(tolerance=.0001)
            this_data = filter_ties(data)
            _share_ties = filter_ties.get_ties(data).mean()
        else:
            this_data = data
            _share_ties = 0
        return _share_ties, this_data

    def get_solver(self, this_data, deviations, constraints, proj):
        return self.solver_cls(
            data=this_data,
            deviations=deviations,
            metric=self.metric,
            plausibility_constraints=constraints,
            num_points=int(NUM_POINTS / (1 + 9 * proj)),
            seed=self.seed,
            project=proj,
            filter_ties=None,
            num_evaluations=NUM_EVAL,
            confidence_level=1 - .05 / len(deviations),
            moment_matrix=self._moment_matrix(deviations),
            moment_weights=self._moment_weights(deviations)
        )

    def _moment_matrix(self, deviations):
        if self._is_rebidding():
            return rebidding.refined_moment_matrix()
        return auction_data.moment_matrix(deviations, 'slope')

    def _moment_weights(self, deviations):
        if self._is_rebidding():
            return np.identity(5)
        return np.identity(len(deviations))

    def _is_rebidding(self):
        return self.solver_cls == rebidding.ParallelRefinedMultistageSolver


compute_minimization_solution = ComputeMinimizationSolution()
compute_minimization_solution_unfiltered = ComputeMinimizationSolution(
    filtering=False)

compute_solution_parallel = ComputeMinimizationSolution(
    solver_cls=solvers.ParallelSolver)
compute_solution_parallel_unfiltered = ComputeMinimizationSolution(
    solver_cls=solvers.ParallelSolver, filtering=False)
compute_solution_rebidding = ComputeMinimizationSolution(
    constraint_func=round2_constraints,
    solver_cls=rebidding.ParallelRefinedMultistageSolver,
    metric=rebidding.RefinedMultistageIsNonCompetitive)
compute_solution_rebidding_unfiltered = ComputeMinimizationSolution(
    constraint_func=round2_constraints,
    solver_cls=rebidding.ParallelRefinedMultistageSolver,
    metric=rebidding.RefinedMultistageIsNonCompetitive,
    filtering=False)


def pretty_plot(title, list_solutions, labels, mark=np.array(['k.:', 'k.-']),
                xticks=(0.5, 1, 1.5, 2), max_y=1.05, xlabel='k',
                ylabel='share of competitive histories'):
    plt_title = title.split('/')[1]
    plt.title(plt_title)
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

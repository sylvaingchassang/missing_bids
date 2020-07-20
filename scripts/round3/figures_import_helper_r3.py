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

from matplotlib import rc

# set path/to/data in a config.py file, which you must create
# config.py is included in .gitignore
from .config import path_data

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

hist_plot = auction_data.hist_plot
sns.set_style('white')


path_figures = os.path.join(path_data, 'figures',
                            datetime.utcnow().strftime('%Y%m%d'))


def ensure_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


ensure_dir(path_figures)
for rnd in ['R1', 'R2', 'R3']:
    ensure_dir(os.path.join(path_figures, rnd))

# set global optimization parameters
NUM_POINTS = 5000
NUM_EVAL = 1

all_deviations = [-.02, .0, .001]
up_deviations = [.0, .001]
down_deviations = [-.02, .0]

all_deviations_tsuchiura = [-.02, .0, .0008]
up_deviations_tsuchiura = [.0, .0008]

r3_min_markups = [.0, .025, .05, .1, .2, .4]
r3_markups = list(product(r3_min_markups, [.5]))

r3_constraints = ([environments.MarkupConstraint(
    max_markup=max_markup, min_markup=min_markup)]
    for min_markup, max_markup in r3_markups)

empty_constraints = [[environments.EmptyConstraint()]] * len(r3_markups)


class ComputeMinimizationSolution:
    _NUM_POINTS = NUM_POINTS
    _NUM_EVAL = NUM_EVAL

    def __init__(
            self, metric=analytics.IsNonCompetitive,
            markups=r3_markups, constraints=empty_constraints,
            seed=0, solver_cls=None):
        self.metric = metric
        self.markups_list = markups
        self.constraints_list = constraints
        self.filtering = True
        self.seed = seed
        self.solver_cls = solver_cls or solvers.IteratedSolver

    def __call__(self, data, deviations):
        solutions = []
        deviations = analytics.ordered_deviations(deviations)
        _share_ties, this_data = self._apply_filter(data)

        for mkps, constraints in zip(self.markups_list, self.constraints_list):
            this_solver = self.get_solver(
                this_data, deviations, constraints, mkps)
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

    def get_solver(self, this_data, deviations, constraints, mkps):
        self._update_metric_params(mkps)
        return self.solver_cls(
            data=this_data,
            deviations=deviations,
            metric=self.metric,
            plausibility_constraints=constraints,
            num_points=self._NUM_POINTS,
            seed=self.seed,
            project=True,
            filter_ties=None,
            num_evaluations=self._NUM_EVAL,
            confidence_level=1 - .05 / len(deviations),
            moment_matrix=self._moment_matrix(deviations),
            moment_weights=self._moment_weights(deviations)
        )

    def _update_metric_params(self, mkps):
        min_markup, max_markup = mkps
        if self._is_efficient():
            self.metric.min_markup = min_markup
            self.metric.max_markup = max_markup

    def _moment_matrix(self, deviations):
        if self._is_rebidding():
            if deviations[0] > -1e-8:
                return rebidding.refined_moment_matrix_up_dev
            return rebidding.refined_moment_matrix()
        return auction_data.moment_matrix(
            analytics.ordered_deviations(deviations), 'slope')

    def _moment_weights(self, deviations):
        if self._is_rebidding():
            if deviations[0] > -1e-8:
                return rebidding.refined_weights_up_dev
            elif deviations[2] < 1e-8:
                return rebidding.refined_weights_down_dev
            return np.identity(5)
        return np.identity(len(analytics.ordered_deviations(deviations)))

    def _is_rebidding(self):
        return self.solver_cls == rebidding.ParallelRefinedMultistageSolver

    def _is_efficient(self):
        return issubclass(self.metric, analytics.EfficientIsNonCompetitive)


compute_efficient_solution_parallel = ComputeMinimizationSolution(
    metric=analytics.EfficientIsNonCompetitive,
    solver_cls=solvers.ParallelSolver)


def dev_repr(devs):
    dev_str = ', '.join([str(d) for d in analytics.ordered_deviations(devs) if
                         np.abs(d) > 10e-8 or d == 0])
    return r'\{' + dev_str + r'\}'


def ensure_decreasing(l):
    #sl = sorted(l, reverse=True)
    #return sl
    return l


def pretty_plot(title, list_solutions, labels, mark=np.array(['k.:', 'k.-']),
                xticks=(0.5, 1, 1.5, 2), max_y=1.05, xlabel='k',
                ylabel='share of competitive histories',
                expect_decreasing=True):
    plt.figure()
    for i, (solutions, label) in enumerate(zip(list_solutions, labels)):
        if expect_decreasing:
            solutions = ensure_decreasing(solutions)
        plt.plot(xticks, solutions, mark[i], label=label)
    if labels[0] is not None:
        plt.legend(loc='best', fontsize=12)
    plt.axis([xticks[0], xticks[-1], 0, max_y])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(
        os.path.join(path_figures, '{}.pdf'.format(title.replace(' ', '_'))),
        bbox_inches='tight')
    plt.clf()


def save2frame(data, columns, title, index=False):
    pd.DataFrame(data=data, columns=columns).to_csv(
        os.path.join(path_figures, '{}.csv'.format(title)), index=index)


def plot_delta(data, rho=.05, filename=None):
    delta = data.df_bids.norm_bid - data.df_bids.most_competitive
    delta = delta[delta.between(-rho, rho)]
    hist_plot(delta, 'distribution of normalized bid differences')
    plt.xlabel(r'normalized bid difference $\Delta$')
    if filename is not None:
        plt.savefig(os.path.join(path_figures, '{}.pdf'.format(filename)))

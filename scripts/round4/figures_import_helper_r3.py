import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
from itertools import product

from mb_api import asymptotics, analytics, environments, auction_data, \
    rebidding, solvers

from matplotlib import rc

# set path/to/data in a script_config.py file, which you must create
# script_config.py is included in .gitignore
from scripts.round3.script_config import path_data

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
NUM_POINTS = 1000
NUM_EVAL = 100

all_deviations = [-.02, .0, .001]
all_deviations_small_sample = [-.02, .0, .002]
up_deviations = [.0, .001]
down_deviations = [-.02, .0]

all_deviations_tsuchiura = [-.02, .0, .0008]
up_deviations_tsuchiura = [.0, .0008]

r3_min_markups = [.0, .025, .05, .1, .2, .4]
r3_markups = list(product(r3_min_markups, [.5]))

r3_constraints = [[environments.MarkupConstraint(
    max_markup=max_markup, min_markup=min_markup)]
    for min_markup, max_markup in r3_markups]

empty_constraints = [[environments.EmptyConstraint()]] * len(r3_markups)

moment_matrix = np.array([[-1, 0, 0],
                          [-1, 1, 0],
                          [0, 1, 0],
                          [0, 1, -1],
                          [0, 0, -1]])
moment_matrix_up = np.array([[1, 0], [1, -1], [0, -1]])
moment_matrix_down = np.array([[-1, 0], [-1, 1], [0, 1]])

# multistage_moment_matrix = np.array([
#     [-.98, 1, .5, 1, 0],
#     [1, -1, -.5, -1, 0],
#     [0, 0, 0, 0, -1],
#     [0, 0, 0, 1, -1]])

multistage_moment_matrix = np.array([
    [-1, 0, 0, 0, 0],
    [-1, 1, .5, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, -1],
    [0, 0, 0, 0, -1]])


# multistage_moment_matrix_up = np.array([
#     [0, 0, 0, 0, -1], [0, 0, 0, 1, -1]])
# multistage_moment_matrix_down = np.array([
#     [-.98, 1, .5, 1, 0],
#     [1, -1, -.5, -1, 0]])

multistage_moment_matrix_up = np.array([
    [0, 0, 0, 1, 0], [0, 0, 0, 0, -1], [0, 0, 0, 1, -1]])
multistage_moment_matrix_down = np.array([
    [-1, 0, 0, 0, 0],
    [-1, 1, .5, 1, 0],
    [0, 0, 0, 1, 0]])


class ComputeMinimizationSolution:
    _NUM_POINTS = NUM_POINTS
    _NUM_EVAL = NUM_EVAL

    def __init__(
            self, metric=analytics.IsNonCompetitive,
            markups=r3_markups, constraints=empty_constraints,
            seed=0, solver_cls=None, confidence_level=.95,
            enhanced_guesses=True):
        self.metric = metric
        self.markups_list = markups
        self.constraints_list = constraints
        self.filtering = True
        self.seed = seed
        self.solver_cls = solver_cls or solvers.ParallelSolver
        self.confidence_level = confidence_level
        self.enhanced_guesses = enhanced_guesses

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
            project=False,
            filter_ties=None,
            num_evaluations=self._NUM_EVAL,
            confidence_level=self.confidence_level,
            moment_matrix=self._moment_matrix(deviations),
            moment_weights=None,
            enhanced_guesses=self.enhanced_guesses
        )

    def _update_metric_params(self, mkps):
        min_markup, max_markup = mkps
        if self._is_efficient():
            self.metric.min_markup = min_markup
            self.metric.max_markup = max_markup

    def _moment_matrix(self, deviations):
        if self._is_multistage_solver():
            return self._multistage_moment_matrix(deviations)
        else:
            return self._singlestage_moment_matrix(deviations)

    def _is_multistage_solver(self):
        return issubclass(self.solver_cls,
                          (asymptotics.AsymptoticMultistageSolver,
                           asymptotics.ParallelAsymptoticMultistageSolver))

    @staticmethod
    def _multistage_moment_matrix(deviations):
        if deviations[0] > -1e-8:
            return multistage_moment_matrix_up
        elif deviations[-1] < 1e-8:
            return multistage_moment_matrix_down
        return multistage_moment_matrix

    @staticmethod
    def _singlestage_moment_matrix(deviations):
        if deviations[0] > -1e-8:
            return moment_matrix_up
        elif deviations[-1] < 1e-8:
            return moment_matrix_down
        return moment_matrix

    def _is_efficient(self):
        return issubclass(self.metric, analytics.EfficientIsNonCompetitive)


compute_asymptotic_solution = ComputeMinimizationSolution(
    metric=analytics.EfficientIsNonCompetitive,
    solver_cls=asymptotics.ParallelAsymptoticSolver)

compute_asymptotic_multistage_solution_95 = ComputeMinimizationSolution(
    metric=rebidding.EfficientMultistageIsNonCompetitive,
    solver_cls=asymptotics.ParallelAsymptoticMultistageSolver)

compute_asymptotic_multistage_solution_10 = ComputeMinimizationSolution(
    metric=rebidding.EfficientMultistageIsNonCompetitive,
    solver_cls=asymptotics.ParallelAsymptoticMultistageSolver,
    confidence_level=.1
)

compute_asymptotic_multistage_solution = \
    compute_asymptotic_multistage_solution_95


def dev_repr(devs):
    dev_str = ', '.join([str(d) for d in analytics.ordered_deviations(devs) if
                         np.abs(d) > 10e-8 or d == 0])
    return r'\{' + dev_str + r'\}'


def pretty_plot(title, list_solutions, labels, mark=np.array(['k.:', 'k.-']),
                xticks=(0.5, 1, 1.5, 2), max_y=1.05, xlabel='k',
                ylabel='share of competitive histories', l1=True):
    plt.figure()
    for i, (solutions, label) in enumerate(zip(list_solutions, labels)):
        plt.plot(xticks, solutions, mark[i], label=label)
    if l1:
        plt.plot(xticks, [1] * len(xticks), linestyle='-', linewidth=.75,
                 color='red')
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

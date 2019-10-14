import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

import lazy_property

from analytics import MinCollusionSolver, MinCollusionResult


class IteratedSolver:
    max_best_sol_index = 2500
    _solution_threshold = 0.01
    _solver_cls = MinCollusionSolver

    def __init__(self, data, deviations, metric, plausibility_constraints,
                 tolerance=None, num_points=1e6, seed=0, project=False,
                 filter_ties=None, number_iterations=1, moment_matrix=None,
                 moment_weights=None, confidence_level=.95):
        self.solver = self._solver_cls(
            data, deviations, metric, plausibility_constraints,
            tolerance=tolerance, num_points=num_points, seed=seed,
            project=project, filter_ties=filter_ties,
            moment_matrix=moment_matrix, moment_weights=moment_weights,
            confidence_level=confidence_level)
        self._number_iterations = number_iterations

    @property
    def _interim_result(self):
        return MinCollusionResult(
            self.solver.problem, self.solver.epigraph_extreme_points,
            self.solver.deviations, self.solver.argmin_columns)

    @property
    def result(self):
        selected_guesses = None
        list_solutions = []
        interim_result = None
        for seed_delta in range(self._number_iterations):
            interim_result = self._interim_result
            list_solutions.append(interim_result.solution)
            selected_guesses = self._get_new_guesses(
                interim_result, selected_guesses)
            self._set_guesses_and_seed(selected_guesses, seed_delta)
        interim_result._solver_data = {'iterated_solutions': list_solutions}
        return interim_result

    def _get_new_guesses(self, interim_result, selected_guesses):
        argmin = interim_result.argmin_array_quantile(
            1 - self._solution_threshold)
        argmin = argmin[:self.max_best_sol_index + 1]
        selected_argmin = argmin[:, 1:-1]

        return np.concatenate((selected_guesses, selected_argmin), axis=0) if \
            selected_guesses is not None else selected_argmin

    def _set_guesses_and_seed(self, selected_guesses, seed_delta):
        self.solver.set_initial_guesses(selected_guesses)
        self.solver.set_seed(self.solver.seed + seed_delta + 1)


MinCollusionIterativeSolver = IteratedSolver


class ParallelSolver:
    _solver_cls = MinCollusionSolver
    _solution_threshold = 0.01

    def __init__(self, data, deviations, metric, plausibility_constraints,
                 num_points=1e6, seed=0, project=False,
                 filter_ties=None, num_evaluations=1, moment_matrix=None,
                 moment_weights=None, confidence_level=.95):
        self._number_evaluations = num_evaluations

        def get_solver(sub_seed, tolerance=None):
            return self._solver_cls(
                data=data, deviations=deviations, metric=metric,
                plausibility_constraints=plausibility_constraints,
                tolerance=tolerance, num_points=num_points,
                project=project, filter_ties=filter_ties, seed=seed + sub_seed,
                moment_matrix=moment_matrix, moment_weights=moment_weights,
                confidence_level=confidence_level)
        self.get_solver = get_solver

    def get_interim_result(self, sub_seed):
        solver = self.get_solver(sub_seed, tolerance=self.tolerance)
        return solver.result.argmin_array_quantile(
            1 - self._solution_threshold)

    @lazy_property.LazyProperty
    def tolerance(self):
        return self.get_solver(0).tolerance

    
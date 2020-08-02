import lazy_property
import numpy as np
import cvxpy
from scipy.stats import norm

from mb_api.auction_data import AuctionData
from mb_api.rebidding import (RefinedMultistageEnvironment,
                              _check_up_down_deviations)
from mb_api.analytics import MinCollusionSolver, ConvexProblem
from mb_api.solvers import ParallelSolver


class PIDMeanAuctionData(AuctionData):

    def _get_counterfactual_demand(self, df_bids, rho):
        df_bids['new_wins'] = self._get_new_wins(df_bids, rho)
        pid_counterfactual_demand = df_bids.groupby('pid')['new_wins'].mean()
        return pid_counterfactual_demand.mean()

    @staticmethod
    def _get_new_wins(df_bids, rho):
        new_bids = df_bids.norm_bid * (1 + rho)
        new_wins = 1. * (new_bids < df_bids.most_competitive) + \
                   .5 * (new_bids == df_bids.most_competitive)
        return new_wins

    def standard_deviation(self, deviations, weights):
        list_moments = self.moment_names(deviations)
        win_vector = self._win_vector(self.df_bids, deviations)
        centered_wins = win_vector[list_moments] - self._demand_vector(
            win_vector, deviations)
        win_vector['square_residual'] = \
            np.square(np.dot(centered_wins, weights))
        variance = win_vector.groupby('pid')['square_residual'].mean().mean()
        return max(np.sqrt(variance), 1e-8)

    def _win_vector(self, df_bids, deviations):
        for rho in deviations:
            df_bids[rho] = self._get_new_wins(df_bids, rho)
        return df_bids

    def _demand_vector(self, win_vector, deviations):
        list_moments = self.moment_names(deviations)
        return win_vector.groupby('pid')[list_moments].mean().mean(axis=0)

    def demand_vector(self, deviations):
        return self._demand_vector(
            self._win_vector(self.df_bids, deviations), deviations)

    @lazy_property.LazyProperty
    def num_auctions(self):
        return len(set(self.df_bids.pid))

    def confidence_threshold(self, weights, deviations, pvalue=.05):
        x = norm.ppf(1 - pvalue)
        demand_vector = self.demand_vector(deviations)
        return np.dot(demand_vector, weights) + x * self.standard_deviation(
            deviations, weights)/np.sqrt(self.num_auctions)

    def moment_names(self, deviations):
        return deviations

    def assemble_target_moments(self, list_rhos, **kwargs):
        return self.demand_vector(list_rhos)


class MultistagePIDMeanAuctionData(PIDMeanAuctionData):

    def _win_vector(self, df_bids, deviations):
        _check_up_down_deviations(deviations)
        down_dev = deviations[0]
        for col, rho in zip(['win_down', 'win0', 'win_up'], deviations):
            df_bids[col] = self._get_new_wins(df_bids, rho)
        df_bids['marg_cont'] = self.is_marginal_cont(df_bids, down_dev)
        df_bids['marg_info'] = self.is_marginal_info(df_bids, down_dev)
        return df_bids[['pid'] + self.moment_names()]

    @staticmethod
    def bids_after_deviation(df_bids, rho):
        return df_bids.norm_bid * (1 + rho)

    @classmethod
    def is_marginal_info(cls, df_bids, rho):
        new_bids = cls.bids_after_deviation(df_bids, rho)
        marginal_info = (1 < new_bids) & (new_bids < df_bids['lowest'])
        return 1. * marginal_info

    @classmethod
    def is_marginal_cont(cls, df_bids, rho):
        new_bids = cls.bids_after_deviation(df_bids, rho)
        marginal_cont = (df_bids['lowest'] > 1) & (new_bids <= 1)
        return 1. * marginal_cont

    def moment_names(self, deviations=None):
        return ['win_down', 'marg_cont', 'marg_info', 'win0', 'win_up']


class AsymptoticProblem(ConvexProblem):

    @property
    def _moment_constraint(self):
        rationalizing_demands = cvxpy.matmul(self._beliefs.T, self.variable)
        moment = 1e2 * cvxpy.matmul(self._moment_matrix, rationalizing_demands)
        return [moment <= 1e2 * self._tolerance]


class AsymptoticMinCollusionSolver(MinCollusionSolver):
    _pbm_cls = AsymptoticProblem

    @property
    def pvalues(self):
        return self._get_pvalues(self._confidence_level)

    def _get_pvalues(self, confidence_level):
        if isinstance(confidence_level, float):
            num_moments = self._moment_matrix.shape[0]
            return np.array([1-confidence_level] * num_moments)/num_moments
        else:
            return 1 - np.array(confidence_level)

    @property
    def tolerance(self):
        return self._get_tolerance(self.pvalues)

    def _get_tolerance(self, pvalues):
        assert isinstance(self._data, PIDMeanAuctionData)
        list_tol = []
        for weights, p in zip(self._moment_matrix, pvalues):
            list_tol.append(
                self.filtered_data.confidence_threshold(
                    weights, self.deviations, p))
        return np.array(list_tol).reshape(-1, 1)


class AsymptoticMultistageSolver(AsymptoticMinCollusionSolver):
    _environment_cls = RefinedMultistageEnvironment

    def _get_pvalues(self, confidence_level):
        if isinstance(confidence_level, float):
            return np.array([1-confidence_level] * 5)/5
        else:
            return 1 - np.array(confidence_level)

    @property
    def default_moment_matrix(self):
        return np.diag([-1, 1, 1, 1, -1])

    @property
    def default_moment_weights(self):
        return None

    @property
    def _winner_env(self):
        return [[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1]]

    @property
    def _loser_env(self):
        return [[1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]]

    @property
    def argmin_columns(self):
        return ['win_down', 'marg_cont', 'marg_info', 'win0', 'win_up',
                'cost', 'metric']


class ParallelAsymptoticSolver(ParallelSolver):
    _solver_cls = AsymptoticMinCollusionSolver


class ParallelAsymptoticMultistageSolver(ParallelSolver):
    _solver_cls = AsymptoticMultistageSolver

from auction_data import AuctionData
from analytics import (DimensionlessCollusionMetrics, ordered_deviations,
                       IsNonCompetitive, MinCollusionSolver)
from solvers import IteratedSolver, ParallelSolver
from environments import EnvironmentBase, descending_sort
import numpy as np


class MultistageAuctionData(AuctionData):

    def _truncated_lowest_bid(self, df_bids):
        return df_bids['lowest']

    @property
    def share_second_round(self):
        return (self.df_auctions['lowest'] > 1).mean()

    @classmethod
    def get_share_marginal(cls, df_bids, rho):
        cls._raise_error_if_neg(rho)
        return cls.share_marginal_info(df_bids, rho) + \
               cls.share_marginal_cont(df_bids, rho)

    @staticmethod
    def _raise_error_if_neg(rho):
        if rho >= 0:
            raise NotImplementedError(
                'marginality not implemented for positive values of rho')

    @classmethod
    def share_marginal_info(cls, df_bids, rho):
        new_bids = cls.bids_after_deviation(df_bids, rho)
        marginal_info = (1 < new_bids) & (new_bids < df_bids['lowest'])
        return marginal_info.mean()

    @classmethod
    def share_marginal_cont(cls, df_bids, rho):
        new_bids = cls.bids_after_deviation(df_bids, rho)
        marginal_cont = (df_bids['lowest'] > 1) & (new_bids <= 1)
        return marginal_cont.mean()

    @staticmethod
    def bids_after_deviation(df_bids, rho):
        return df_bids.norm_bid * (1 + rho)

    def _get_counterfactual_demand(self, df_bids, rho):
        counterfactual_demand = AuctionData._get_counterfactual_demand(
            df_bids, rho)
        if rho < 0:
            counterfactual_demand -= self.get_share_marginal(df_bids, rho)
        return counterfactual_demand


class RefinedMultistageData(MultistageAuctionData):
    def _get_counterfactual_demand(self, df_bids, rho):
        counterfactual_demand = AuctionData._get_counterfactual_demand(
            df_bids, rho)
        if rho < 0:
            counterfactual_demand = [
                counterfactual_demand,
                self.share_marginal_cont(self.df_bids, rho),
                self.share_marginal_info(self.df_bids, rho)
            ]
        return counterfactual_demand


class MultistageIsNonCompetitive(DimensionlessCollusionMetrics):
    max_win_prob = 1

    def __call__(self, env):
        return self._downward_non_ic(env) | self._upward_non_ic(env)

    def _upward_non_ic(self, env):
        payoffs = self._get_payoffs(env)
        upward_payoffs = payoffs[self.equilibrium_index:]
        return np.max(upward_payoffs) > upward_payoffs[0]

    def _downward_non_ic(self, env):
        if self.all_upward_dev:
            return False
        payoffs = self._get_payoffs(env)
        penalty = self._get_penalty(env)
        return (np.max(payoffs[:self.equilibrium_index] + penalty)
                > payoffs[self.equilibrium_index])

    def _get_penalty(self, env):
        downward_beliefs = env[:self.equilibrium_index]
        return np.multiply(
            self.max_win_prob - downward_beliefs,
            self._deviations[: self.equilibrium_index])

    @property
    def all_upward_dev(self):
        return all(self._deviations >= 0)


def _check_up_down_deviations(deviations):
    dev = ordered_deviations(deviations)
    if (sum(dev < 0) != 1) or (sum(dev > 0) != 1):
        raise ValueError('profile of deviations must have one '
                         'downward and one upward deviation')


class RefinedMultistageIsNonCompetitive(IsNonCompetitive):
    coeff_marginal_cont = 1
    coeff_marginal_info = .5

    def __init__(self, deviations):
        _check_up_down_deviations(deviations)
        super().__init__(deviations)
        self.rho_down, _, self.rho_up = self._deviations

    def _get_payoffs(self, env):
        win_down, marg_cont, marg_info, win0, win_up, cost = env
        payoff_down = self._payoff_down(env)
        payoff_up = self._payoff_up(env)
        return [payoff_down, win0 * (1 - cost), payoff_up]

    def _payoff_down(self, env):
        win_down, marg_cont, marg_info, win0, win_up, cost = env
        if self.rho_down > -1e-8:
            return win0 * (1 + self.rho_down - cost)
        reserve = 1 + .5 * self._deviations[0]
        v = reserve - cost
        return win_down * (1 + self.rho_down - cost) - \
            marg_cont * self.coeff_marginal_cont * v - \
            marg_info * self.coeff_marginal_info * self.coeff_marginal_cont * v

    def _payoff_up(self, env):
        win_down, marg_cont, marg_info, win0, win_up, cost = env
        if self.rho_up < 1e-8:
            return win0 * (1 - cost) - 1e-8
        return win_up * (1 + self.rho_up - cost)


class RefinedMultistageEnvironment(EnvironmentBase):
    def _generate_raw_environments(self, num, seed):
        """win_down, marg_cont, marg_info, win0, win_up, cost"""
        win_down, marg_cont, marg_info, win0, win_up, cost = \
            range(6)
        np.random.seed(seed)
        num = int(num)
        env = np.empty((num, 6))
        env[:, [win_down, win0, win_up]] = descending_sort(
            np.random.rand(num, 3))
        env[:, marg_cont] = np.random.rand(num) * (
                env[:, win_down] - env[:, win0])
        env[:, marg_info] = np.random.rand(num) * (1 - env[:, win_down])
        env[:, cost] = np.random.rand(num)
        return env


def refined_moment_matrix(slope=True):
    if slope:
        return np.array([
            [1, 0, 0, 0, 0],
            [1, -1, 0, -1, 0],
            [0, -1, 1, 0, 0],
            [-1, 0, 0, 1, 0],
            [0, 0, 0, -1, 1]
        ])
    else:
        return np.identity(5)


refined_moment_matrix_up_dev = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, -1, 1]
        ])


refined_weights_down_dev = np.diag([1, 1, 1, 1, 0])
refined_weights_up_dev = np.diag([0, 0, 0, 1, 1])


class RefinedMultistageSolver(MinCollusionSolver):
    _environment_cls = RefinedMultistageEnvironment

    @property
    def default_moment_matrix(self):
        return refined_moment_matrix()

    @property
    def default_moment_weights(self):
        return np.array(5 * [1])

    @property
    def argmin_columns(self):
        return ['win_down', 'marg_cont', 'marg_info', 'win0', 'win_up',
                'cost', 'metric']


class IteratedRefinedMultistageSolver(IteratedSolver):
    _solver_cls = RefinedMultistageSolver


class ParallelRefinedMultistageSolver(ParallelSolver):
    _solver_cls = RefinedMultistageSolver

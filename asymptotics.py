import lazy_property
import numpy as np
import cvxpy
from scipy.stats import norm

from .auction_data import AuctionData
from .analytics import MinCollusionSolver, ConvexProblem


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
        win_vector = self._win_vector(self.df_bids, deviations)
        centered_wins = win_vector[deviations] - self._demand_vector(
            win_vector, deviations)
        win_vector['square_residual'] = \
            np.square(np.dot(centered_wins, weights))
        variance = win_vector.groupby('pid')['square_residual'].mean().mean()
        return np.sqrt(variance)

    def _win_vector(self, df_bids, deviations):
        for rho in deviations:
            df_bids[rho] = self._get_new_wins(df_bids, rho)
        return df_bids

    @staticmethod
    def _demand_vector(win_vector, deviations):
        return win_vector.groupby('pid')[deviations].mean().mean(axis=0)

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
            num_dev = len(self._deviations)
            return np.array([1-confidence_level] * num_dev)/num_dev
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
        return np.array(list_tol)





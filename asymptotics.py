import lazy_property
import numpy as np
from scipy.stats import norm

from auction_data import AuctionData


class AuctionDataPIDMean(AuctionData):

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

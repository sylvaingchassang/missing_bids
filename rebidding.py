from auction_data import AuctionData
from analytics import DimensionlessCollusionMetrics
import environments
import numpy as np


class MultistageAuctionData(AuctionData):

    def _truncated_lowest_bid(self, df_bids):
        return df_bids['lowest']

    @property
    def share_second_round(self):
        return (self.df_auctions['lowest'] > 1).mean()

    def get_share_marginal(self, rho):
        if rho >= 0:
            raise NotImplementedError(
                'marginality not implemented for positive values of rho')
        new_bids = self.df_bids.norm_bid * (1 + rho)
        marginal_cont = (self.df_bids['lowest'] > 1) & (new_bids < 1)
        marginal_info = (1 < new_bids) & (new_bids < self.df_bids['lowest'])
        return (marginal_cont | marginal_info).mean()


class MultistageIsNonCompetitive(DimensionlessCollusionMetrics):

    def __init__(self, deviations, bounds_proba_win=1):
        super().__init__(deviations)
        self.max_win_prob = np.array(bounds_proba_win)

    def __call__(self, env):
        return self._downward_non_ic(env) | self._upward_non_ic(env)

    def _upward_non_ic(self, env):
        payoffs = self._get_payoffs(env)
        upward_payoffs = payoffs[self.equilibrium_index:]
        return np.max(upward_payoffs) > upward_payoffs[0]

    def _downward_non_ic(self, env):
        payoffs = self._get_payoffs(env)
        penalty = self._get_penalty(env)
        return (np.max(payoffs[:self.equilibrium_index] + penalty)
                > payoffs[self.equilibrium_index])

    def _get_penalty(self, env):
        downward_beliefs = env[:self.equilibrium_index]
        return np.multiply(
            self.max_win_prob - downward_beliefs,
            self._deviations[: self.equilibrium_index])

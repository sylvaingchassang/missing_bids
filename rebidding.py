from auction_data import AuctionData
import analytics
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

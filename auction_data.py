import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lazy_property


class AuctionData(object):
    def __init__(self, bidding_data_path):
        self._raw_data = None
        self._read_bids(bidding_data_path)

    def _read_bids(self, bidding_data):
        _raw_data = pd.read_csv(bidding_data)
        self._raw_data = _raw_data.loc[~_raw_data.norm_bid.isnull()]

    @lazy_property.LazyProperty
    def df_auctions(self):
        auctions = self._raw_data.groupby('pid').norm_bid
        df_auctions = pd.concat(
            (auctions.min(), auctions.apply(self._second_lowest_bid)), axis=1)
        df_auctions.columns = ['lowest', 'second_lowest']
        return df_auctions[auctions.count() > 1]

    @staticmethod
    def _second_lowest_bid(auction_bids):
            v = auction_bids.values
            if len(v) == 1:
                return 1
            v.sort()
            return v[1]

    @lazy_property.LazyProperty
    def df_bids(self):
        df_bids = self._raw_data[['pid', 'norm_bid', 'bid', 'reserveprice']]
        df_bids = df_bids.loc[df_bids.pid.isin(self.df_auctions.index)]
        competition = self._get_competition(df_bids)
        df_bids = pd.concat((df_bids, competition), axis=1)
        return df_bids

    def _get_competition(self, df_bids):
        competition = df_bids[['pid', 'norm_bid']].apply(
            self._competitive_lowest_and_second_lowest_bids, axis=1)
        return pd.DataFrame(
            data=np.array(tuple(competition)),
            index=competition.index,
            columns=['most_competitive', 'lowest', 'second_lowest']
        )

    def _competitive_lowest_and_second_lowest_bids(self, x):
        auction_id, bid = x
        bids = self.df_auctions.loc[auction_id]
        if bid == bids.lowest:
            return bids.second_lowest, bids.lowest, bids.second_lowest
        else:
            return bids.lowest, bids.lowest, bids.second_lowest

    def get_counterfactual_demand(self, rho):
        new_bids = self.df_bids.norm_bid * (1+rho)
        wins = (new_bids < self.df_bids.most_competitive).mean()
        ties = .5 * (new_bids == self.df_bids.most_competitive).mean()
        return wins + ties

    def demand_function(self, start, stop, num=500):
        range_rho = np.linspace(start=start, stop=stop, num=num)
        demand = pd.DataFrame(
            data=[self.get_counterfactual_demand(rho) for rho in range_rho],
            index=range_rho,
            columns=['counterfactual demand']
        )
        return demand


def hist_plot(df, title=''):
    plt.figure(figsize=(10, 6))
    sns.distplot(
        df, kde=False, hist_kws=dict(alpha=1), bins=200, hist=True,
        norm_hist=1)
    plt.title(title)
    plt.tight_layout(), plt.show()

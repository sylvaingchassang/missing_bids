import pandas as pd
import numpy as np


class AuctionData(object):
    def __init__(self, reference_file):
        self.df_bids = pd.read_csv(reference_file)
        self.df_auctions = None
        self._list_available_auctions = None

    def generate_auction_data(self):
        '''
        generates an auction-level dataframe from bid-level dataframe df
        includes second lowest bidder and cartel metrics
        '''

        list_auctions = list(set(self.df_bids.pid))
        list_auctions.sort()
        data = []
        for a in list_auctions:
            this_df = self.df_bids[self.df_bids.pid == a]
            bids = list(this_df.norm_bid.dropna())
            if len(bids) > 1:
                bids.sort()
                lowest = bids[0]
                second_lowest = bids[1]
                data.append(
                    [a, lowest, second_lowest])

        df_auctions = pd.DataFrame(
            data=data,
            columns=['pid', 'lowest', 'second_lowest']
        )
        df_auctions = df_auctions.set_index('pid')

        self.df_auctions = df_auctions
        self._list_available_auctions = set(self.df_auctions.index)

    def add_most_competitive(self):
        self.df_bids['most_competitive'] = 0
        self.df_bids['most_competitive'] = \
            self.df_bids[['pid', 'norm_bid']].apply(
            self._add_most_competitive, axis=1
        )

    def _add_most_competitive(self, x):
        u, v = x
        if u in self.list_available_auctions:
            low1 = self.df_auctions.ix[u].lowest
            low2 = self.df_auctions.ix[u].second_lowest
            if v == low1:
                return low2
            else:
                return low1
        else:
            return np.NaN

    @property
    def list_available_auctions(self):
        return self._list_available_auctions

    rho_p = .001
    rho_m = .05

    df['hD'] = 1. * (df.norm_bid < df.most_competitive)
    bid_up = (1 + rho_p) * df.norm_bid
    bid_down = (1 - rho_m) * df.norm_bid

    df['hP_p'] = 1. * (df.most_competitive < bid_up) * (
    df.norm_bid < df.most_competitive)

    df['hP_m'] = 1. * (df.most_competitive > bid_down) * (
    df.norm_bid > df.most_competitive)

    df['category'] = 0

    def encode(x, y, z):
        return 1000 + 100. * x + 10. * y + 1. * z

    df['category'] = encode(df['hD'], df['hP_m'], df['hP_p'])

    # enumerating number of histories in each category

    enum_cat = {}
    for v in Y.values():
        c = encode(*v)
        enum_cat[v] = np.sum(df.category == c)

    num_hist = sum(enum_cat.values())

    def get_demand(p_C):
        mean = np.array([0., 0., 0.])
        for i, z in enumerate(p_C):
            v = Y[i]
            mean += z * np.array(v) * enum_cat[v] / float(num_hist)
        return tuple(mean)

    def get_competitive_share(p_C):
        c = 0
        for i, z in enumerate(p_C):
            v = Y[i]
            c += z * enum_cat[v] / float(num_hist)
        return c


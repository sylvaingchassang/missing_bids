import pandas as pd
import numpy as np


class AuctionData(object):
    def __init__(self, reference_file):
        self.df_bids = pd.read_csv(reference_file)
        self.df_auctions = None
        self._list_available_auctions = None
        # by convention demand outcomes are y = (D, Pm, Pp)
        self._demand_outcomes = \
            {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (1, 0, 1)}
        self._enum_categories = None
        self._num_histories = None

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

    @staticmethod
    def _encode(x, y, z):
        return 1000 + 100. * x + 10. * y + 1. * z

    def compute_demand_moments(self, rho_p=.001, rho_m=.05):
        self.df_bids['sample_D'] = \
            1. * (self.df_bids.norm_bid < self.df_bids.most_competitive)
        bid_up = (1 + rho_p) * self.df_bids.norm_bid
        bid_down = (1 - rho_m) * self.df_bids.norm_bid

        self.df_bids['sample_Pp'] = \
            1. * (self.df_bids.most_competitive < bid_up) * (
                self.df_bids.norm_bid < self.df_bids.most_competitive)

        self.df_bids['sample_Pm'] = \
            1. * (self.df_bids.most_competitive > bid_down) * (
                self.df_bids.norm_bid > self.df_bids.most_competitive)

        self.df_bids['category'] = 0

        self.df_bids['category'] = self._encode(
            self.df_bids['sample_D'], self.df_bids['sample_Pm'],
            self.df_bids['sample_Pp']
        )

    def categorize_histories(self):
        self._enum_categories = {}
        for v in self.demand_outcomes.values():
            c = self._encode(*v)
            self.enum_categories[v] = np.sum(self.df_bids.category == c)

        self._num_histories = sum(self.enum_categories.values())

    def get_demand(self, p_c):
        mean = np.array([0., 0., 0.])
        for i, z in enumerate(p_c):
            v = self.demand_outcomes[i]
            mean += z * np.array(v) * self.enum_categories[v] / float(
                self.num_histories)
        return tuple(mean)

    def get_competitive_share(self, p_c):
        c = 0
        for i, z in enumerate(p_c):
            v = self.demand_outcomes[i]
            c += z * self.enum_categories[v] / float(self.enum_categories)
        return c

    @property
    def enum_categories(self):
        return self._enum_categories

    @property
    def num_histories(self):
        return self._num_histories

    @property
    def demand_outcomes(self):
        return self._demand_outcomes

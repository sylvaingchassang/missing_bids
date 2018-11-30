import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lazy_property


class AuctionData(object):
    def __init__(self, bidding_data):
        self.df_bids = pd.read_csv(bidding_data)
        self.df_tied_bids = None
        self.df_auctions = None
        self._bid_gap = None
        self._list_available_auctions = None
        self._list_tied_auctions = None
        # by convention demand outcomes are y = (D, Pm, Pp)
        self._demand_outcomes = \
            {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (1, 0, 1)}
        self._enum_categories = None
        self._num_histories = None
        self.set_bid_data(self.df_bids)

    def _collect_auctions(self):
        self._list_available_auctions = set(self.df_auctions.index)

    def set_bid_data(self, df_bids):
        self.df_bids = df_bids
        self.generate_auction_data()
        self.add_auction_characteristics()

        self._collect_auctions()

        self.df_bids = self.df_bids.loc[
            ~ (df_bids.norm_bid.isnull() |
               self.df_bids.most_competitive.isnull())
        ]
        self.df_tied_bids = self.df_bids.loc[
            self.df_bids.norm_bid == self.df_bids.most_competitive
        ]
        self.df_bids = self.df_bids.loc[
            self.df_bids.norm_bid != self.df_bids.most_competitive
        ]

    @property
    def all_bids(self):
        return pd.concat((self.df_bids, self.df_tied_bids), axis=0)

    def save_data(self, path):
        self.all_bids.to_csv(path)

    def generate_auction_data(self):
        """
        generates an auction-level dataframe from bid-level dataframe df
        includes second lowest bidder
        """

        list_auctions = list(set(self.df_bids.pid))
        list_auctions.sort()
        data = []
        self._bid_gap = []
        for a in list_auctions:
            this_df = self.df_bids[self.df_bids.pid == a]
            bids = this_df.norm_bid.dropna().values
            if len(bids) > 1:
                bids.sort()
                self._bid_gap += list(
                    np.divide(bids[1:] - bids[:-1], bids[1:])
                )
                lowest = bids[0]
                second_lowest = bids[1]
                data.append(
                    [a, lowest, second_lowest])

        df_auctions = pd.DataFrame(
            data=data,
            columns=['pid', 'lowest', 'second_lowest']
        )
        self.df_auctions = df_auctions.set_index('pid')

    def add_auction_characteristics(self):
        self._collect_auctions()
        self.df_bids.loc[:, 'most_competitive'] = 0
        self.df_bids.loc[:, 'lowest'] = 0
        self.df_bids.loc[:, 'second_lowest'] = 0
        competition_frame = \
            self.df_bids[['pid', 'norm_bid']].apply(
                self._add_most_competitive, axis=1
            )
        self.df_bids.loc[:, 'most_competitive'] = competition_frame.apply(
            lambda x: x[0])
        self.df_bids.loc[:, 'lowest'] = competition_frame.apply(lambda x: x[1])
        self.df_bids.loc[:, 'second_lowest'] = competition_frame.apply(
            lambda x: x[2])

    def _add_most_competitive(self, x):
        u, v = x
        if u in self.list_available_auctions:
            low1 = self.df_auctions.ix[u].lowest
            low2 = self.df_auctions.ix[u].second_lowest
            if v == low1:
                return low2, low1, low2
            else:
                return low1, low1, low2
        else:
            return np.NaN, np.NaN, np.NaN

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
            1. * (self.df_bids.most_competitive >= bid_down) * (
                self.df_bids.norm_bid >= self.df_bids.most_competitive)

        self.df_bids.loc[:, 'category'] = 0

        self.df_bids.loc[:, 'category'] = self._encode(
            self.df_bids['sample_D'], self.df_bids['sample_Pm'],
            self.df_bids['sample_Pp']
        )

    def get_counterfactual_demand(self, rho_p, rho_m, num=500):
        assert (rho_p >= 0) & (rho_m >= 0)
        range_rho = zip(np.linspace(0, rho_m,  num=num),
                        np.linspace(0, rho_p, num=num))
        data = {'demand': [], 'revenue': []}
        index = []
        for this_rho_m, this_rho_p in range_rho:
            bid_up = (1+this_rho_p) * self.df_bids.bid
            bid_down = (1-this_rho_m) * self.df_bids.bid
            index += [-this_rho_m, this_rho_p]
            self.compute_demand_moments(this_rho_p, this_rho_m)
            d_m = self.df_bids.sample_D + self.df_bids.sample_Pm
            d_p = self.df_bids.sample_D - self.df_bids.sample_Pp
            data['demand'] += [d_m.mean(), d_p.mean()]
            data['revenue'] += [(d_m * bid_down).mean(),
                                (d_p * bid_up).mean()]
        return pd.DataFrame(data=data,
                            index=index,
                            columns=['demand', 'revenue']).sort_index()

    def get_bid_gaps(self):
        self._bid_gap.sort()
        return pd.DataFrame(data=self._bid_gap)

    def categorize_histories(self):
        ''' category = 1xyz with x, y, z = d, p_m, p_p'''
        self._enum_categories = {}
        for v in self.demand_outcomes.values():
            c = self._encode(*v)
            self._enum_categories[v] = np.sum(self.df_bids.category == c)

        self._num_histories = sum(self.enum_categories.values())

    def get_demand(self, p_c):
        """
        :param p_c: likelihood of conserving auctions in each category
        :return: sample tuple (D, Pm, Pp) corresponding to selection p_C
        """
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
            c += z * self.enum_categories[v] / float(self._num_histories)
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

    @property
    def tied_winners(self):
        num_tied = self.df_tied_bids.shape[0]
        num_untied = self.df_bids.shape[0]
        return num_tied / float(num_tied + num_untied)


def hist_plot(this_delta, title =''):
    plt.figure(figsize=(10,6))
    sns.distplot(
        this_delta, kde=False,
        hist_kws=dict(alpha=1),
        bins=200,
        hist=True,
        #norm_hist=1,
    )
    plt.title(title)
    plt.tight_layout(), plt.show()

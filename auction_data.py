import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lazy_property
import multiprocessing


def _read_bids(bidding_data_or_path):
    df = bidding_data_or_path
    if isinstance(df, str):
        df = pd.read_csv(df)
    return df.loc[~df.norm_bid.isnull()]


class AuctionData:
    def __init__(self, bidding_data_or_path, clean=True):
        self.raw_data = _read_bids(bidding_data_or_path)
        self.data = self._drop_auction_with_too_few_data(clean)

    def _drop_auction_with_too_few_data(self, clean):
        bids_count = self.raw_data.groupby("pid").norm_bid.count()
        df = self.raw_data.set_index("pid").loc[bids_count > 1 * clean]
        return df.reset_index()

    @classmethod
    def from_clean_bids(cls, df_bids):
        auction_data = cls(df_bids)
        auction_data._df_bids = df_bids
        return auction_data

    @lazy_property.LazyProperty
    def df_auctions(self):
        auctions = self.data.groupby('pid').norm_bid
        df_auctions = pd.concat(
            (auctions.min(), auctions.apply(self._second_lowest_bid)), axis=1)
        df_auctions.columns = ['lowest', 'second_lowest']
        return df_auctions

    @staticmethod
    def _second_lowest_bid(auction_bids):
        v = auction_bids.values
        if len(v) == 1:
            return 1
        v.sort()
        return v[1]

    @lazy_property.LazyProperty
    def df_bids(self):
        cols = ['pid', 'norm_bid', 'bid', 'reserveprice']
        df_bids = self.data[cols].set_index("pid")
        df_bids["lowest"] = self.df_auctions["lowest"]
        df_bids["second_lowest"] = self.df_auctions["second_lowest"]

        df_bids["most_competitive"] = df_bids["lowest"]
        is_bid_lowest = np.isclose(df_bids["norm_bid"], df_bids["lowest"])
        df_bids.loc[is_bid_lowest, "most_competitive"] = df_bids.loc[
            is_bid_lowest, "second_lowest"]
        return df_bids.reset_index()

    def get_counterfactual_demand(self, rho):
        return self._get_counterfactual_demand(self.df_bids, rho)

    @staticmethod
    def _get_counterfactual_demand(df_bids, rho):
        new_bids = df_bids.norm_bid * (1 + rho)
        wins = (new_bids < df_bids.most_competitive).mean()
        ties = .5 * (new_bids == df_bids.most_competitive).mean()
        return wins + ties

    def demand_function(self, start, stop, num=500):
        range_rho = np.linspace(start=start, stop=stop, num=num)
        return pd.DataFrame(
            list(map(self.get_counterfactual_demand, range_rho)),
            index=range_rho, columns=['counterfactual demand']
        )

    def bootstrap_demand_sample(self, list_rhos, num_samples=100):
        pool = multiprocessing.Pool()
        sample_size = self.df_bids.shape[0]
        return pd.DataFrame(pool.map(
            self._single_bootstrap,
            [(sample_size, i, list_rhos) for i in range(num_samples)]))

    def _single_bootstrap(self, args):
        sample_size, random_state, list_rhos = args
        resampled_bids = self.df_bids.sample(
            sample_size, replace=True, random_state=random_state)
        return [self._get_counterfactual_demand(resampled_bids, rho)
                for rho in list_rhos]


def moment_matrix(n, option='slope'):
    n = n if isinstance(n, int) else len(n)
    if option.lower() == 'slope':
        return np.diag(n * [1], 0) - np.diag((n-1) * [1], -1)
    elif option.lower() == 'level':
        return np.identity(n)


def moment_distance(candidate_demand, target_demand, weights, mat=None):
    mat = mat if mat is not None else moment_matrix(len(candidate_demand))
    candidate_moment, target_moment = \
        [np.dot(np.array(d), mat.T) for d in (candidate_demand, target_demand)]
    return np.dot(np.array(weights),
                  np.square(candidate_moment - target_moment))


class FilterTies:
    def __init__(self, tolerance=.0001):
        self.tolerance = tolerance

    def __call__(self, auction_data):
        original_data = auction_data.data.copy()
        ties = self.get_ties(auction_data)
        original_data = original_data.loc[~ties]
        return AuctionData(original_data)

    def get_ties(self, auction_data):
        return np.isclose(
            auction_data.df_bids['lowest'],
            auction_data.df_bids['second_lowest'],
            atol=self.tolerance
        )


def hist_plot(df, title=''):
    plt.figure(figsize=(10, 6))
    sns.distplot(
        df, kde=False, hist_kws=dict(alpha=1), bins=200, hist=True,
        norm_hist=1)
    plt.title(title)
    plt.tight_layout(), plt.show()

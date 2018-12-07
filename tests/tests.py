from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal
import numpy as np
import auction_data
import os
import analytics
from parameterized import parameterized


class TestAuctionData(TestCase):
    def setUp(self):
        self.auctions = auction_data.AuctionData(
            bidding_data=os.path.join('reference_data', 'tsuchiura_data.csv'))

    def test_bid_data(self):
        assert self.auctions.df_bids.shape == (5876, 7)
        assert self.auctions.df_auctions.shape == (1469, 2)
        assert_array_equal(
            self.auctions._df_bids.pid.values[:10],
            np.array([15, 15, 15, 15, 15, 16, 16, 16, 16, 16])
        )

    def test_auction_data(self):
        assert_array_almost_equal(
            self.auctions.df_auctions.lowest.values[:10],
            np.array([0.89655173, 0.94766617, 0.94867122, 0.69997638,
                      0.9385258, 0.74189192, 0.7299363, 0.94310075,
                      0.96039605, 0.97354496])
        )

    def test_most_competitive(self):
        assert_array_almost_equal(
            self.auctions.df_bids.most_competitive.values[10:20],
            np.array(
                [0.79662162, 0.74189192, 0.74189192, 0.74189192, 0.74189192,
                 0.74189192, 0.74189192, 0.74189192, 0.74189192, 0.74189192])
        )

    def test_counterfactual_demand(self):
        dmd = self.auctions.get_counterfactual_demand(.05)
        assert_array_almost_equal(dmd, 0.02067733151)

    def test_demand_function(self):
        dmd = self.auctions.demand_function(start=-.01, stop=.01, num=4)
        assert_array_almost_equal(dmd, [[0.495575], [0.293397], [0.21341],
                                        [0.105599]])


class TestEnvironments(TestCase):

    def setUp(self):
        self.constraints = [analytics.MarkupConstraint(.6),
                            analytics.InformationConstraint(.5, [.65, .48])]
        self.env_no_cons = analytics.Environments(num_actions=2)

    def test_generate_raw_environments(self):
        assert_array_almost_equal(
            self.env_no_cons._generate_raw_environments(3, seed=0),
            [[0.715189, 0.548814, 0.602763],
             [0.544883, 0.423655, 0.645894],
             [0.891773, 0.437587, 0.963663]]
        )

    def test_generate_environments_no_cons(self):
        assert_array_almost_equal(
            self.env_no_cons._generate_raw_environments(3, seed=0),
            self.env_no_cons.generate_environments(3, seed=0),
        )

    @parameterized.expand([
        [[0], [[0.544883, 0.423655, 0.645894],
               [0.891773, 0.437587, 0.963663]]],
        [[1], [[0.715189, 0.548814, 0.602763],
               [0.544883, 0.423655, 0.645894]]],
        [[0, 1], [[0.544883, 0.423655, 0.645894]]]
    ])
    def test_generate_environments_cons(self, cons_id, expected):
        env = analytics.Environments(
            num_actions=2,
            constraints=[self.constraints[i] for i in cons_id]
        )
        assert_array_almost_equal(
            env.generate_environments(3, seed=0),
            expected
        )


class TestConstraints(TestCase):

    def setUp(self):
        self.mkp = analytics.MarkupConstraint(2.)
        self.info = analytics.InformationConstraint(.01, [.5, .4, .3])

    def test_markup_constraint(self):
        assert not self.mkp([.5, .6, .33])
        assert self.mkp([.5, .6, .34])

    def test_info_bounds(self):
        assert_array_almost_equal(
            self.info.belief_bounds,
            [[0.4975, 0.5025], [0.397602, 0.402402], [0.297904, 0.302104]]
        )

    def test_info(self):
        assert self.info([.5, .4, .3, .5])
        assert not self.info([.5, .4, .35, .5])
        assert not self.info([.45, .4, .3, .5])





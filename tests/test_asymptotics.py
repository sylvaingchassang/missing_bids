import os
from unittest.case import TestCase

from numpy.testing import assert_array_almost_equal, assert_almost_equal

import asymptotics


class TestAuctionDataPIDMean(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        self.auctions = asymptotics.AuctionDataPIDMean(
            bidding_data_or_path=path
        )
        self.deviations = [-.02, 0, .005]

    def test_counterfactual_demand(self):
        assert_array_almost_equal(
            [self.auctions.get_counterfactual_demand(r) for r in [-.05, .05]],
            [0.896378, 0.023846])

    def test_standard_deviation(self):
        assert_almost_equal(self.auctions.standard_deviation(
            self.deviations, (.4, .2, .4)), 0.34073885)

    def test_win_vector(self):
        df_bids = self.auctions._win_vector(
            self.auctions.df_bids, self.deviations)
        assert_almost_equal(
            df_bids[['pid'] + self.deviations].head(3),
            [[15, 1., 1., 0.],
             [15, 1., 0., 0.],
             [15, 1., 0., 0.]])

    def test_demand_vector(self):
        assert_array_almost_equal(
            self.auctions.demand_vector(self.deviations),
            [0.757466, 0.29353, 0.211717])

    def test_num_auctions(self):
        assert self.auctions.num_auctions == 1469

    def test_confidence_threshold(self):
        assert_almost_equal(
            self.auctions.confidence_threshold([-1, 1, 0], self.deviations),
            -0.4425795)

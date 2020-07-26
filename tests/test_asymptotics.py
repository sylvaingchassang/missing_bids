import os
import numpy as np
from unittest.case import TestCase

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from .. import asymptotics
from .. import environments
from ..auction_data import moment_matrix, FilterTies
from ..analytics import EfficientIsNonCompetitive


class TestAuctionDataPIDMean(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        self.auctions = asymptotics.PIDMeanAuctionData(
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


class TestAsymptoticSolver(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data',
            'tsuchiura_data.csv')
        self.data = asymptotics.PIDMeanAuctionData(bidding_data_or_path=path)
        self.constraints = [
            environments.MarkupConstraint(.6),
            environments.InformationConstraint(.5, [.65, .48])]
        self.solver = asymptotics.AsymptoticMinCollusionSolver(
            deviations=[-.02, 0, .005], data=self.data,
            metric=EfficientIsNonCompetitive, project=False,
            tolerance=None, plausibility_constraints=self.constraints,
            seed=0, num_points=10000, filter_ties=FilterTies(),
            moment_matrix=moment_matrix(3)
        )

    def test_pvalues(self):
        assert_array_almost_equal(self.solver.pvalues, [0.016667] * 3)
        assert_array_almost_equal(self.solver._get_pvalues([.98, .97, .99]),
                                  [.02, .03, .01])

    def test_tolerance(self):
        assert_array_almost_equal(self.solver.tolerance,
                                  [0.781659, -0.436149, -0.065206])
        assert_array_almost_equal(self.solver._get_tolerance([.005, .1, .1]),
                                  [0.786685, -0.447208, -0.071234])


class TestAsymptoticProblem(TestCase):
    def setUp(self):
        self.metrics = [1., 0, 1, 1, 0]
        self.demands = [.5, .4]
        self.beliefs = np.array(
            [[.6, .5], [.45, .4], [.7, .6], [.4, .3], [.4, .2]])
        tolerance = .0005
        self.cvx = asymptotics.AsymptoticProblem(
            self.metrics, self.beliefs, self.demands, tolerance,
            moment_matrix=moment_matrix(len(self.demands), 'level'),
            moment_weights=np.ones_like(self.demands)
        )
        self.res = self.cvx.solution
        self.argmin = self.cvx.variable.value

    def test_problem(self):
        assert 1 == 0
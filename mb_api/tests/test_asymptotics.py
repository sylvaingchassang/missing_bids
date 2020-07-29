import os
import numpy as np
from unittest.case import TestCase

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from .. import asymptotics
from .. import environments
from ..auction_data import _read_bids, FilterTies
from ..analytics import EfficientIsNonCompetitive
from .test_rebidding import _load_multistage_data


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


def _load_data_constraints_metric(
        data_cls=asymptotics.PIDMeanAuctionData, coeff=1):
    path = os.path.join(
        os.path.dirname(__file__), 'reference_data',
        'tsuchiura_data.csv')
    raw_data = _read_bids(path)
    raw_data['reserveprice'] *= coeff
    raw_data['norm_bid'] = raw_data['bid'] / raw_data['reserveprice']
    if coeff != 1:
        data = data_cls(bidding_data_or_path=raw_data)
    else:
        data = data_cls(bidding_data_or_path=path)
    constraints = [environments.EmptyConstraint()]
    metric = EfficientIsNonCompetitive
    metric.max_markup = .5
    metric.min_markup = .05

    return data, constraints, metric


class TestAsymptoticSolver(TestCase):
    def setUp(self):
        data, constraints, metric = \
            _load_data_constraints_metric()
        self.solver = asymptotics.AsymptoticMinCollusionSolver(
            deviations=[-.02, 0, .005], data=data,
            metric=metric, project=False,
            tolerance=None, plausibility_constraints=constraints,
            seed=0, num_points=10000, filter_ties=FilterTies(),
            moment_matrix=np.diag([-1, 1, -1])
        )
        self.solver_enhanced = asymptotics.AsymptoticMinCollusionSolver(
            deviations=[-.02, 0, .005], data=data,
            metric=metric, project=False,
            tolerance=None, plausibility_constraints=constraints,
            seed=0, num_points=10000, filter_ties=FilterTies(),
            moment_matrix=np.diag([-1, 1, -1]), enhanced_guesses=True
        )

    def test_pvalues(self):
        assert_array_almost_equal(self.solver.pvalues, [0.016667] * 3)
        assert_array_almost_equal(self.solver._get_pvalues([.98, .97, .99]),
                                  [.02, .03, .01])

    def test_tolerance(self):
        assert_array_almost_equal(self.solver.tolerance.T,
                                  [[-0.733889,  0.319218, -0.190617]])
        assert_array_almost_equal(self.solver._get_tolerance([.005, .1, .1]).T,
                                  [[-0.728863,  0.309116, -0.199704]])

    def test_solution(self):
        assert_almost_equal(self.solver.result.solution, 0.2042775, decimal=5)

    def test_guesses(self):
        assert_array_almost_equal(
            self.solver_enhanced._initial_guesses[-5:],
            [[0.757774, 0.293822, 0.213461, 1.], [1, 1, 1, 1],
             [1., 1., 0., 0.], [0., 0., 0., 1.], [1., 0., 0., 1.]]
        )
        assert_almost_equal(
            self.solver_enhanced.result.solution, 0.18362843, decimal=5)


class TestAsymptoticProblem(TestCase):
    def setUp(self):
        self.metrics = [1., 0, 1, 1, 0]
        self.demands = [.5, .4]
        self.beliefs = np.array(
            [[.6, .5], [.45, .4], [.7, .6], [.4, .3], [.4, .2]])
        tolerance = np.array([-.55, .45]).reshape(-1, 1)
        self.cvx = asymptotics.AsymptoticProblem(
            self.metrics, self.beliefs, self.demands, tolerance,
            moment_matrix=np.array([[-1, 0], [0, 1]]),
            moment_weights=np.ones_like(self.demands)
        )
        self.res = self.cvx.solution
        self.argmin = self.cvx.variable.value

    def test_minimal_value(self):
        assert_almost_equal(self.res, 0.4375)

    def test_solution(self):
        assert_array_almost_equal(
            self.argmin, [[0], [.375], [.4375], [0], [.1875]],
            decimal=5)


class TestMultistagePIDMeanAuctionData(TestCase):
    def setUp(self):
        self.data = asymptotics.MultistagePIDMeanAuctionData(
            _load_multistage_data())
        self.deviations = [-.01, 0, .005]

    def test_error(self):
        self.assertRaises(ValueError, self.data._win_vector,
                          self.data.df_bids, [0., .005])

    def test_win_vector(self):
        assert_array_almost_equal(
            self.data._win_vector(self.data.df_bids, self.deviations).head(3),
            [[15., 1., 0., 0., 1., 0.],
             [15., 1., 0., 0., 0., 0.],
             [15., 0, 0., 0., 0., 0.]]
        )
        assert_array_almost_equal(
            self.data._win_vector(self.data.df_bids, self.deviations).mean(),
            [9.611816e+02, 4.745575e-01, 5.973451e-02, 2.382573e-02,
             2.501702e-01, 1.806501e-01], decimal=5
        )

    def test_demands(self):
        assert_array_almost_equal(
            self.data.demand_vector(self.deviations),
            [0.532985, 0.08288, 0.034218, 0.29353, 0.212029]
        )

    def test_confidence_thresholds(self):
        assert_almost_equal(
            self.data.confidence_threshold([1, 0, 0, -1, 0], self.deviations),
            0.2577102084
        )


class TestAsymptoticMultistageSolver(TestCase):
    def setUp(self):
        data, constraints, metric = _load_data_constraints_metric(
                asymptotics.MultistagePIDMeanAuctionData, .985)
        self.solver = asymptotics.AsymptoticMultistageSolver(
            deviations=[-.01, 0, .005], data=data,
            metric=metric, project=False,
            tolerance=None, plausibility_constraints=constraints,
            seed=0, num_points=500, filter_ties=FilterTies(),
            moment_matrix=np.diag([-1, 1, 1, 1, -1])
        )
        self.solver_enhanced = asymptotics.AsymptoticMultistageSolver(
            deviations=[-.01, 0, .005], data=data,
            metric=metric, project=False,
            tolerance=None, plausibility_constraints=constraints,
            seed=0, num_points=500, filter_ties=FilterTies(),
            moment_matrix=np.diag([-1, 1, 1, 1, -1]), enhanced_guesses=True
        )

    def test_pvalues(self):
        assert_array_almost_equal(self.solver.pvalues, [0.01] * 5)

    def test_tolerance(self):
        assert_array_almost_equal(
            self.solver.tolerance.T,
            [[-0.502121,  0.099678,  0.045623,  0.321584, -0.188801]])

    def test_demands(self):
        assert_array_almost_equal(
            self.solver.demands,
            [0.532521, 0.082876, 0.0345, 0.293822, 0.213775])

    def test_solution(self):
        assert_almost_equal(
            self.solver.result.solution, 0.8231369, decimal=5)

    def test_guesses(self):
        assert_array_almost_equal(
            self.solver_enhanced._initial_guesses[-6:],
            [[1., 0., 0., 1., 1., 0.],
             [1., 0., 0., 1., 1., 1.],
             [1., 0., 0., 0., 0., 1.],
             [1., 1., 1., 0., 0., 0.],
             [1., 1., 1., 0., 0., 1.],
             [0., 0., 0., 0., 0., 0.]]
        )
        assert_almost_equal(
            self.solver_enhanced.result.solution, 0.814052448, decimal=5)


class TestParallelAsymptoticSolver(TestCase):
    def setUp(self):
        data, constraints, metric = \
            _load_data_constraints_metric()
        self.solver = asymptotics.ParallelAsymptoticSolver(
            deviations=[-.01, 0, .005], data=data,
            metric=metric, project=False, num_evaluations=2,
            plausibility_constraints=constraints,
            seed=0, num_points=1000, filter_ties=FilterTies(),
            moment_matrix=np.diag([-1, 1, -1])
        )

    def test_tolerance(self):
        assert_array_almost_equal(
            self.solver.tolerance.T, [[-0.534184,  0.319218, -0.190617]])

    def test_solution(self):
        assert_almost_equal(
            self.solver.result.solution, 0.1426484, decimal=5)


class TestParallelAsymptoticMultistageSolver(TestCase):
    def setUp(self):
        data, constraints, metric = _load_data_constraints_metric(
                asymptotics.MultistagePIDMeanAuctionData, .985)
        self.solver = asymptotics.ParallelAsymptoticMultistageSolver(
            deviations=[-.01, 0, .005], data=data,
            metric=metric, project=False, num_evaluations=2,
            plausibility_constraints=constraints,
            seed=0, num_points=500, filter_ties=FilterTies(),
            moment_matrix=np.diag([-1, 1, 1, 1, -1])
        )

    def test_tolerance(self):
        assert_array_almost_equal(
            self.solver.tolerance.T,
            [[-0.502121,  0.099678,  0.045623,  0.321584, -0.188801]])

    def test_solution(self):
        assert_almost_equal(
            self.solver.result.solution, 0.82257359, decimal=5)

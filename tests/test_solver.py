import os
from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal, assert_almost_equal
from solvers import ParallelSolver
from auction_data import AuctionData
import environments
from analytics import IsNonCompetitive, MinCollusionSolver


class TestParallelSolver(TestCase):

    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        self.data = AuctionData(bidding_data_or_path=path)
        self.constraints = [environments.MarkupConstraint(.6),
                            environments.InformationConstraint(.5, [.65, .48])]
        self.ps = ParallelSolver(
            self.data, [-.02], IsNonCompetitive, self.constraints,
            tolerance=None, num_points=1000, seed=1, project=False,
            filter_ties=None, num_evaluations=5, moment_matrix=None,
            moment_weights=None, confidence_level=.95
        )

    def test_get_solver(self):
        solver = self.ps.get_solver(sub_seed=2)
        isinstance(solver, MinCollusionSolver)
        assert solver._seed == 3

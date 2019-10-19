import os
import numpy as np
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
        self.constraints = [environments.MarkupConstraint(.6)]
        self.ps = ParallelSolver(
            self.data, [-.02], IsNonCompetitive, self.constraints,
            num_points=200, seed=1, project=True,
            filter_ties=None, num_evaluations=5, moment_matrix=None,
            moment_weights=None, confidence_level=.95
        )

    def test_get_solver(self):
        solver = self.ps.get_solver(sub_seed=2)
        isinstance(solver, MinCollusionSolver)
        assert solver._seed == 3

    def test_tolerance(self):
        assert_almost_equal(self.ps.tolerance, 0.00024576331835932043)

    def test_get_interim_result(self):
        assert_array_almost_equal(
            self.ps.get_interim_result(0)[:2],
            [[4.063856e-01, 9.186018e-01, 4.020250e-04, 9.912847e-01, 0],
             [1.069201e-01, 1.350792e-01, 1.263295e-01, 8.146233e-01, 0]])
        assert_array_almost_equal(
            self.ps.get_interim_result(1)[:2],
            [[0.386962, 0.993852, 0.067144, 0.988968, 0.],
             [0.079128, 0.101488, 0.087256, 0.93857, 0.]])

    def test_get_all_interim_results(self):
        all_res = self.ps.get_all_interim_results()
        assert len(all_res) == 5
        assert_array_almost_equal(
            np.concatenate([res[:2] for res in all_res[:2]], axis=0),
            [[4.063856e-01, 9.186018e-01, 4.020250e-04, 9.912847e-01, 0],
             [1.069201e-01, 1.350792e-01, 1.263295e-01, 8.146233e-01, 0],
             [0.386962, 0.993852, 0.067144, 0.988968, 0.],
             [0.079128, 0.101488, 0.087256, 0.93857, 0.]])

    def test_get_all_interim_solutions(self):
        all_res = self.ps.get_all_interim_solutions()
        assert len(all_res) == 5
        assert_array_almost_equal(
            all_res,
            [1.93867e-12, 4.65251e-12, 2.04209e-11, 5.97352e-11, 6.35486e-11])

    def test_interim_guesses(self):
        guesses = self.ps.get_interim_guesses()
        assert guesses.shape == (98, 3)
        assert_array_almost_equal(
            guesses[[0, 10, 20]],
            [[9.186018e-01, 4.020250e-04, 9.912847e-01],
             [7.508121e-01, 7.259980e-01, 9.562398e-01],
             [1.014882e-01, 8.725561e-02, 9.385696e-01]]
        )

    def test_result(self):
        min_interim_sol = min(self.ps.get_all_interim_solutions())
        overall_sol = self.ps.result.solution
        assert_array_almost_equal(
            [min_interim_sol, overall_sol], [1.938671e-12, 2.182863e-12]
        )

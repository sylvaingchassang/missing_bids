from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal
from parameterized import parameterized

from .. import environments


class TestEnvironments(TestCase):
    def setUp(self):
        self.constraints = [environments.MarkupConstraint(.6),
                            environments.InformationConstraint(.5, [.65, .48])]
        self.env_no_cons = environments.Environment(num_actions=2)
        self.env_with_initial_guesses = \
            environments.Environment(num_actions=2,
                                     initial_guesses=np.array([[0.7, 0.5, 0.6]]))

    def test_generate_raw_environments(self):
        assert_array_almost_equal(
            self.env_no_cons._generate_raw_environments(3, seed=0),
            [[0.715189, 0.548814, 0.602763],
             [0.544883, 0.423655, 0.645894],
             [0.891773, 0.437587, 0.963663]]
        )

    def test_generate_environments_with_initial_guesses(self):
        assert_array_almost_equal(
            self.env_with_initial_guesses.generate_environments(
                num_points=3, seed=0),
            [[0.715189, 0.548814, 0.602763],
             [0.544883, 0.423655, 0.645894],
             [0.891773, 0.437587, 0.963663],
             [0.7, 0.5, 0.6]]
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
        env = environments.Environment(
            num_actions=2, constraints=[self.constraints[i] for i in cons_id])
        assert_array_almost_equal(
            env.generate_environments(3, seed=0),
            expected)

    def test_generate_environment_with_projection(self):
        env = environments.Environment(
            num_actions=2, constraints=self.constraints,
            project_constraint=True)
        assert_array_almost_equal(
            env.generate_environments(3, seed=1),
            [[0.691139, 0.460906, 0.625043],
             [0.597473, 0.394812, 0.659627],
             [0.60716, 0.404473, 0.773788]])


class TestConstraints(TestCase):
    def setUp(self):
        self.mkp = environments.MarkupConstraint(2.)
        self.mkp_lower = environments.MarkupConstraint(2., 1.)
        self.info = environments.InformationConstraint(.01, [.5, .4, .3])
        self.ref_environments = np.array([[.8, .4, .3, .1],
                                          [.9, .3, .1, .8]])

    def test_markup_constraint(self):
        assert not self.mkp([.5, .6, .33])
        assert self.mkp([.5, .6, .34])
        assert_array_almost_equal(
            self.mkp.project(self.ref_environments),
            [[0.8, 0.4, 0.3, 0.4],
             [0.9, 0.3, 0.1, 0.866667]])

    def test_markup_lower_bound(self):
        assert not self.mkp_lower([.5, .6, .51])
        assert self.mkp_lower([.5, .6, .49])
        assert_array_almost_equal(
            self.mkp_lower.project(self.ref_environments),
            [[0.8, 0.4, 0.3, 0.35],
             [0.9, 0.3, 0.1, 0.466667]])

    def test_info_bounds(self):
        assert_array_almost_equal(
            self.info.belief_bounds,
            [[0.4975, 0.5025], [0.397602, 0.402402], [0.297904, 0.302104]])
        assert_array_almost_equal(
            self.info.project(self.ref_environments),
            [[0.5015, 0.399522, 0.299164, 0.1],
             [0.502, 0.399042, 0.298324, 0.8]])

    def test_info(self):
        assert self.info([.5, .4, .3, .5])
        assert not self.info([.5, .4, .35, .5])
        assert not self.info([.45, .4, .3, .5])
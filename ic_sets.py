import numpy as np
from auction_data import AuctionData
import itertools


class ICSets(object):

    def __init__(self, rho_p, rho_m, auction_data, m, t, k):
        self._rho_p = None
        self._rho_m = None
        self._auction_data = None
        self._m = None
        self._t = None
        self._k = None
        self.set_deviations(rho_p=rho_p, rho_m=rho_m)
        self.set_data(auction_data)
        self.set_tolerance_parameters(m, t, k)

    def set_deviations(self, rho_p, rho_m):
        self._rho_p = rho_p
        self._rho_m = rho_m

    def set_data(self, auction_data):
        assert isinstance(auction_data, AuctionData)
        auction_data.compute_demand_moments(rho_m=self.rho_m, rho_p=self.rho_p)
        auction_data.categorize_histories()
        self._auction_data = auction_data

    def set_tolerance_parameters(self, m, t, k):
        self._m = m
        self._t = t
        self._k = k

    @property
    def rho_p(self):
        return self._rho_p

    @property
    def rho_m(self):
        return self._rho_m

    @property
    def auction_data(self):
        return self._auction_data

    @property
    def m(self):
        return self._m

    @property
    def t(self):
        return self._t

    @property
    def k(self):
        return self._k

    @property
    def lower_slope_pp(self):
        return self.rho_p/(1. + self.rho_p - 1. / (1.+self.m))

    @property
    def tangent_binding_pm(self):
        return self.rho_p/(self.rho_p + self.rho_m)

    def value_pm(self, d, pp=None):
        if pp is None:
            pp = self.lower_slope_pp * d
        if pp > (self.rho_p/(self.rho_p + self.rho_m)) * d:
            return 1 - d
        else:
            return min(
                1 - d,
                (self.rho_m * d) / (self.rho_p * (-1 + d / pp) - self.rho_m)
            )

    def extreme_points_set_z(self, u_d, o_d):
        list_points_unconstrained, list_points_constrained = (
            [(u_d, 0, a * u_d), (u_d, 0, u_d), (o_d, 0, a * o_d),
             (o_d, 0, o_d), (u_d, 1 - u_d, a * u_d), (u_d, 1 - u_d, u_d),
             (o_d, 1 - o_d, a * o_d), (o_d, 1 - o_d, o_d)]
            for a in [self.lower_slope_pp, self.tangent_binding_pm]
        )

        if self.tangent_binding_pm <= self.lower_slope_pp:
            list_points = list_points_unconstrained
        else:
            list_points = list_points_constrained
            list_points += [
                (u_d, 0, self.lower_slope_pp * u_d),
                (u_d, self.value_pm(u_d), self.lower_slope_pp * u_d),
                (o_d, 0, self.lower_slope_pp * o_d),
                (o_d, self.value_pm(o_d), self.lower_slope_pp * o_d)
            ]

            denominator = -self.rho_p + (self.rho_p/self.lower_slope_pp)
            intersect_d = (denominator - self.rho_m)/denominator
            if intersect_d <= o_d:
                list_points.append(
                    (intersect_d, self.value_pm(intersect_d),
                     self.lower_slope_pp * intersect_d)
                )
        return np.array(list_points)

    def compute_set_a(self, p_c):
        d, pm, pp = self.auction_data.get_demand(p_c)
        t = self.t
        range_d = [d-t, d+t]
        range_pm = [pm - t, pm + t]
        range_pp = [pp - t, pp + t]
        return range_d, range_pm, range_pp

    @staticmethod
    def intersect_range(range_a, range_b):
        c, d = (max(range_a[0], range_b[0]),
                min(range_a[1], range_b[1]))
        if c <= d:
            return [c, d]
        else:
            return None

    def intersect_box(self, set_a, set_b):
        """Note: a box in three dimensions is described by by three ranges"""
        return [self.intersect_range(r_a, r_b) for
                (r_a, r_b) in zip(set_a, set_b)]

    def underline_b(self, x):
        return self.overline_b(x, -self.k)

    def overline_b(self, x, k=None):
        k = self.k if k is None else k
        l = float(x) / (1. - x)
        return l * np.exp(k)/(1. + l * np.exp(k))

    def compute_set_b(self, p_c):
        d, pm, pp = self.auction_data.get_demand(p_c)
        return tuple(
            [self.underline_b(X), self.overline_b(X)]
            for X in [d, pm, pp]
        )

    @staticmethod
    def get_box_extreme_points(box):
        return np.array(list(itertools.product(*box)))

    @staticmethod
    def get_perpendicular(x, y=None, z=None):
        if y is None:
            y = x[3:6]
            z = x[6:9]
            x = x[0:3]
        u = x - y
        v = x - z
        return np.cross(u, v)

    @staticmethod
    def choose_3_in(n):
        list_sel = []
        for a in xrange(n-2):
            for b in xrange(a+1, n-1):
                for c in xrange(b+1, n):
                    list_sel.append([a,b,c])
        return np.array(list_sel)

    def get_triplets(self, array_points):
        n = len(array_points)
        array_sel = self.choose_3_in(n)
        return np.concatenate(
            tuple(array_points[array_sel[:, i]] for i in range(3)),
            axis=1
        )

#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         mmnn.py
@Author:       AMASE team
@Website:      https://github.com/AMASE-Project
@Description:  Wavelength calibration functions for the AMASE project.
'''

import numpy as np
from typing import Optional
from itertools import combinations
from itertools import product
from scipy.optimize import minimize
from amasedrp.utils.parallel_processing import run as prun
from .wavcal_utils import find_nearest
from .wavcal_utils import inverse_wavelength_solution
from .wavcal_utils import check_solution_monotonicity
from .wavcal_utils import fit_solution
from .wavcal_utils import fit_monotonic_solution


class MmNn():
    """
    Class to handle wavelength calibration using the M-m-N-n algorithm.
    """

    def __init__(
        self,
        M_known_wls: list[float],
        m_known_wls: list[float],
        N_peak_ys: list[float],
        n_peak_ys: list[float],
        y_domain: Optional[list[int]] = None,
        deg: int = 3,
        poly_form: object = np.polynomial.Polynomial,
        wl_increases_with_y: bool = True,
        reject_residual_outliers: bool = True,
    ):
        """Initialize the MmNn class for wavelength calibration.

        # NOTE: deg + 1 <= len(m_known_wls) <= len(n_peak_ys)
        # NOTE:
        # (1) M_known_wls should be included in N_peak_ys, as much as possible
        # (2) m_known_wls should be included in n_peak_ys, as much as possible
        """
        self.M_known_wls = M_known_wls
        self.m_known_wls = m_known_wls
        self.N_peak_ys = N_peak_ys
        self.n_peak_ys = n_peak_ys
        self.y_domain = y_domain
        self.deg = deg
        self.poly_form = poly_form
        self.wl_increases_with_y = wl_increases_with_y
        self.reject_residual_outliers = reject_residual_outliers

    # -------------------------------------------------------------------------
    # pre-processing
    # -------------------------------------------------------------------------
    def pre_process(self):
        self.solution = None
        self.inv_solution = None
        self.solution_rmse = np.nan
        self.solution_monotonicity = None
        if self.y_domain is None:
            self.y_domain = [np.min(self.N_peak_ys), np.max(self.N_peak_ys)]
        self.y_min, self.y_max = np.min(self.y_domain), np.max(self.y_domain)

    # -------------------------------------------------------------------------
    # fitting all possible wavelength solutions and finding the best one
    # -------------------------------------------------------------------------

    def generate_possible_combinations(self):
        """
        For given
        (1) m known lines that are most reliably detectable;
        (2) n strong detected peaks that some of them are expected to match
            the known lines;
        this function generates all possible combinations of selecting
        (deg+1) known lines from m known lines, and (deg+1) detected peaks
        from n detected peaks.
        These combinations can be used to fit possible wavelength solutions.
        """
        # possible combination
        ys = np.array(list(
            combinations(self.n_peak_ys, self.deg+1)
        ))
        wls = np.array(list(
            combinations(self.m_known_wls, self.deg+1)
        ))
        # all possible pairs of combinations
        poss_comb = list(product(
            ys, wls
        ))
        del ys, wls
        return poss_comb

    # # without monotonicity constraint
    # def _fit(ys, wls, deg, poly_form, **kwargs):
    #     """
    #     Fit the wavelength solution for given y coordinates and wavelengths.
    #     """
    #     try:
    #         poss_poly = fit_solution(ys, wls, deg=deg, poly_form=poly_form)
    #     except:  # noqa: E722  # NOTE: check which cases may raise exceptions
    #         print(f"Warning: fitting failed for ys: {ys}, wls: {wls}.")
    #         coeffs = np.full(deg+1, 0., dtype=float)
    #         coeffs[0] = -1.
    #         poss_poly = poly_form(coeffs)
    #     return poss_poly

    # with monotonicity constraint
    def _fit(ys, wls, deg, poly_form, monotonicity, y_min, y_max):
        """
        Fit the wavelength solution for given y coordinates and wavelengths.
        """
        try:
            poss_poly = fit_monotonic_solution(
                ys, wls, deg=deg, poly_form=poly_form,
                monotonicity=monotonicity, y_min=y_min, y_max=y_max,
            )
        except:  # noqa: E722  # NOTE: check which cases may raise exceptions
            print(f"Warning: fitting failed for ys: {ys}, wls: {wls}.")
            coeffs = np.full(deg+1, 0., dtype=float)
            coeffs[0] = -1. if monotonicity == 1. else 1.
            poss_poly = poly_form(coeffs)
        return poss_poly

    def fit_possible_solution(self, ys, wls):
        """
        Fit the wavelength solution for given possible combinations of
        y coordinates and wavelengths.
        """
        # monotonicity handling
        if self.wl_increases_with_y:
            ys = np.sort(ys)
            wls = np.sort(wls)
        else:
            ys = np.sort(ys)
            wls = np.sort(wls)[::-1]
        # fit the wavelength solution
        poss_poly = MmNn._fit(
            ys, wls,
            deg=self.deg,
            poly_form=self.poly_form,
            monotonicity=1. if self.wl_increases_with_y else -1.,
            y_min=None,
            y_max=None,
            # y_min=self.y_min,
            # y_max=self.y_max,
        )
        rmse = self.calculate_fitting_rmse(poss_poly, verbose=False)
        # store the coefficients and RMSE
        coeffs = poss_poly.coef
        output = np.append(coeffs, rmse)
        return output

    def find_best_possible_solution(
            self,
            parallel: bool = True,
            n_jobs: int = -1,
            backend: str = 'loky',
    ):
        """
        Find the best possible wavelength solution by:
        (1) generating all possible combinations of selecting
            (deg+1) known lines from m known lines, and
            (deg+1) detected peaks from n detected peaks;
        (2) fitting all possible solutions for these combinations,
            and calculating the corresponding RMSE;
        (3) selecting the solution with the smallest RMSE as the best one.
        """
        # generate all possible combinations
        inputs = self.generate_possible_combinations()
        if len(inputs) == 0:
            raise ValueError(
                "Not enough known lines or detected peaks to fit the "
                "wavelength solution.")
        # fit for all possible pairs of combinations
        outputs = prun(
            function=self.fit_possible_solution,
            inputs=inputs,
            parallel=parallel, n_jobs=n_jobs, backend=backend,
        )
        # find the best coefficients
        rmses = [outputs[i][-1] for i in range(len(outputs))]
        rmse = np.nanmin(rmses)
        poss_coeffs = outputs[np.nanargmin(rmses)][:-1]
        poss_poly = self.poly_form(poss_coeffs)
        return poss_poly, rmse

    # -------------------------------------------------------------------------
    # evaluating the fitting quality of possible wavelength solutions
    # -------------------------------------------------------------------------

    def calculate_fitting_residuals(self, poss_poly):
        """
        Suppose that:
        (1) M known lines with wavelengths are given;
        (2) N peaks with y coordinates are detected;
        (3) each known line should be matched to one of the detected peaks as
            much as possible.
        Given the wavelength solution, this function calculates the fitting
        residuals (i.e, absolute differences) by matching M known lines to the
        closest detected peak among N peaks.
        The smaller the residuals, the better the fitting.
        """
        N_peak_wls = poss_poly(self.N_peak_ys)
        M_known_wls, _, indices = find_nearest(
            self.M_known_wls, N_peak_wls)
        M_closest_ys = self.N_peak_ys[indices]
        residuals = M_known_wls - N_peak_wls[indices]
        residuals = np.abs(residuals)
        return residuals, M_known_wls, M_closest_ys

    def calculate_fitting_rmse(self, poss_poly, verbose=False):
        """
        Suppose that:
        (1) M known lines with wavelengths are given;
        (2) N peaks with y coordinates are detected;
        (3) each known line should be matched to one of the detected peaks as
            much as possible.
        Given the wavelength solution, this function calculates the RMSE
        by matching M known lines to the closest detected peak among N peaks.
        The smaller the RMSE, the better the fitting.
        """
        # calculate the fitting residuals (absolute differences)
        residuals, M_known_wls, M_closest_ys \
            = self.calculate_fitting_residuals(poss_poly)
        # whether to reject the extreme residuals as outliers
        # (e.g., extreme residuals may be caused by bad detected peaks)
        if self.reject_residual_outliers:
            bounds = [5, 95]
            bounds = np.percentile(residuals, bounds)
            bounds = np.array([np.nanmin(bounds), np.nanmax(bounds)])
            cond = (bounds[0] <= residuals) & (residuals <= bounds[1])
            residuals = residuals[cond]
            M_known_wls = M_known_wls[cond]
            M_closest_ys = M_closest_ys[cond]
            del bounds, cond
        # calculate the fitting RMSE
        rmse = np.sqrt(np.sum(residuals ** 2)) / len(residuals)
        if verbose:
            return rmse, M_known_wls, M_closest_ys
        else:
            return rmse

    # -------------------------------------------------------------------------
    # refining the possible wavelength solution by e.g.,
    # (1) use least squares fitting to minimize the RMSE and slightly adjust
    #     the coefficients of the polynomial;
    # (2) use M_known_wls and M_closest_ys to re-fit the polynomial.
    # -------------------------------------------------------------------------

    def refine_possible_solution(self, ini_poss_poly):
        """
        Refine the wavelength solution. Use least squares fitting to minimize
        the RMSE by slightly adjusting the initial possible polynomial.
        """
        # i.e., poss_poly = a * ini_poss_poly + b
        # where a and b are the coefficients to be determined.
        def objective(params):
            a, b = params

            # define the new polynomial as a * ini_poss_poly + b
            def poss_poly(x):
                return a * ini_poss_poly(x) + b

            # calculate the fitting RMSE
            return self.calculate_fitting_rmse(poss_poly, verbose=False)

        # initial guess for a and b
        initial_guess = [1.0, 0.0]
        initial_rmse = objective(initial_guess)
        # minimize the objective function
        result = minimize(objective, initial_guess)
        # extract the optimized parameters
        a_opt, b_opt = result.x
        del result
        # optimized result
        if initial_rmse < objective([a_opt, b_opt]):
            a_opt, b_opt = initial_guess
        poss_poly = a_opt * ini_poss_poly + b_opt
        rmse = self.calculate_fitting_rmse(poss_poly, verbose=False)
        return poss_poly, rmse

    def refit_possible_solution(self, ini_poss_poly, ensure_monotonicity=True):
        """
        Re-fit the wavelength solution using M_known_wls and M_closest_ys.
        """
        # based on the given possible polynomial,
        # get the M most closest y coordinates for M known wavelengths
        _, M_known_wls, M_closest_ys = \
            self.calculate_fitting_rmse(ini_poss_poly, verbose=True)
        if ensure_monotonicity:
            poss_poly = fit_monotonic_solution(
                M_closest_ys, M_known_wls,
                deg=self.deg,
                poly_form=self.poly_form,
                monotonicity=1. if self.wl_increases_with_y else -1.,
                y_min=self.y_min,
                y_max=self.y_max,
            )
        else:
            poss_poly = fit_solution(
                M_closest_ys, M_known_wls,
                deg=self.deg,
                poly_form=self.poly_form,
            )
        rmse = self.calculate_fitting_rmse(poss_poly, verbose=False)
        return poss_poly, rmse

    # -------------------------------------------------------------------------
    # post-processing
    # -------------------------------------------------------------------------
    def post_process(
            self,
            refine: bool = True,  # NOTE: for debugging, can be removed later
            refit: bool = True,  # NOTE: for debugging, can be removed later
    ):
        # the current solution
        poss_poly, rmse = self.solution, self.solution_rmse

        # refine the solution (until convergence)
        if refine:
            while True:
                old_rmse = float(rmse)
                poss_poly, rmse = self.refine_possible_solution(poss_poly)
                if np.isclose(old_rmse, rmse, atol=1e-8):
                    break
            rmse = self.calculate_fitting_rmse(poss_poly)

        # re-fit the solution (to ensure the mono. and further reduce the RMSE)
        # and further refine it (until convergence)
        if refit:
            poss_poly, rmse = self.refit_possible_solution(poss_poly)
            while True:
                old_rmse = float(rmse)
                poss_poly, rmse = self.refine_possible_solution(poss_poly)
                if np.isclose(old_rmse, rmse, atol=1e-8):
                    break
            rmse = self.calculate_fitting_rmse(poss_poly)

        # store the solution after refine/re-fit/...
        self.solution = poss_poly
        self.solution_rmse = rmse

        # get and store the inverse solution (i.e., from wavelength to y)
        if ~np.isnan(self.y_min) and ~np.isnan(self.y_max):
            self.inv_solution = lambda wl: inverse_wavelength_solution(
                poss_poly, wl, y_min=self.y_min, y_max=self.y_max
            )
        else:
            self.inv_solution = None
        # check the monotonicity of the solution
        self.solution_monotonicity = check_solution_monotonicity(
            self.solution,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    # -------------------------------------------------------------------------
    # core function for wavelength calibration
    # -------------------------------------------------------------------------

    def wavelength_calibration(
            self,
            parallel: bool = True,
            n_jobs: int = -1,
            backend: str = 'loky',
    ):
        """
        Perform wavelength calibration using the M-m-N-n algorithm, i.e.,
        (1) based on m most reliably detectable known lines and
            n strongest detected peaks, find all possible wavelength solutions;
        (2) based on M known lines and N detected peaks, evaluate the fitting
            quality of each possible solution, and select the best one.
        """
        # pre-processing
        self.pre_process()
        # find the best possible wavelength solution
        poss_poly, rmse = self.find_best_possible_solution(
            parallel=parallel,
            n_jobs=n_jobs,
            backend=backend,
        )
        # store
        self.solution = poss_poly
        self.solution_rmse = rmse
        # post-processing
        self.post_process()

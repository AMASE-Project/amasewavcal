#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         mmnn.py
@Author:       AMASE team
@Website:      https://github.com/AMASE-Project
@Description:  Wavelength calibration functions for the AMASE project.
'''

import numpy as np
from itertools import combinations
from itertools import product
from scipy.optimize import minimize
from amasedrp.utils.parallel_processing import run as prun
from .wavcal_utils import find_nearest
from .wavcal_utils import inverse_wavelength_solution
from .wavcal_utils import check_solution_monotonicity


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
        peak_y_coor_lim: list[int] = [0, 10000],
        deg: int = 3,
        poly_form: object = np.polynomial.Legendre,
        wl_increases_with_y: bool = True,
        reject_residual_outliers: bool = True,
    ):
        """Initialize the MmNn class for wavelength calibration.

        Parameters
        ----------
        M_known_wls : list[float]
            M prominent emission lines from calibration lamp spectra.
        m_known_wls : list[float]
            m of M most reliably detectable lines.
        N_peak_ys : list[float]
            N significant peaks detected in the uncalibrated spectrum.
        n_peak_ys : list[float]
            n of N most strongest peaks.
        peak_y_coor_lim : list[int], optional
            The y-coordinate limits for detected peaks, by default [0, 10000].
        deg : int, optional
            Degree of the polynomial for wavelength solution, by default 3.
        poly_form : object, optional
            Polynomial form to use, by default np.polynomial.Legendre.
        wl_increases_with_y : bool, optional
            Whether wavelength increases with y-coordinate, by default True.
        reject_residual_outliers : bool, optional
            Whether to reject outliers in residuals when calculating RMSE,
            by default True.
        solution : object
            The fitted wavelength solution function (from y to wavelength).
        inv_solution : object
            The inverse wavelength solution function (from wavelength to y).
        solution_rmse : float
            The RMSE of the fitted wavelength solution.

        # NOTE: deg + 1 <= len(m_known_wls) <= len(n_peak_ys)
        # NOTE:
        # (1) M_known_wls should be included in N_peak_ys, as much as possible
        # (2) m_known_wls should be included in n_peak_ys, as much as possible
        """
        self.M_known_wls = M_known_wls
        self.m_known_wls = m_known_wls
        self.N_peak_ys = N_peak_ys
        self.n_peak_ys = n_peak_ys
        self.peak_y_coor_lim = peak_y_coor_lim
        self.deg = deg
        self.poly_form = poly_form
        self.wl_increases_with_y = wl_increases_with_y
        self.reject_residual_outliers = reject_residual_outliers
        self.solution = None
        self.inv_solution = None
        self.solution_rmse = np.nan
        self.solution_monotonicity = None

    # -------------------------------------------------------------------------
    # finding possible wavelength solutions
    # -------------------------------------------------------------------------

    def _fit_wavelength_solution(
            self,
            ys: list[float],
            wls: list[float],
    ):
        """
        Fit the wavelength solution for given y coordinates and wavelengths.
        And use the known wavelengths and all detected peaks to calculate
        the RMSE for this fitting result.
        """
        if self.wl_increases_with_y:
            ys = np.sort(ys)
            wls = np.sort(wls)
        else:
            ys = np.sort(ys)
            wls = np.sort(wls)[::-1]
        try:
            coeffs = self.poly_form.fit(ys, wls, deg=self.deg).convert().coef
            poss_poly = self.poly_form(coeffs)
            # calculate the RMSE for this fitting
            rmse = self.calculate_fitting_rmse(poss_poly)
        except:  # noqa: E722  # NOTE: check which cases may raise exceptions
            coeffs = np.full(self.deg+1, 0., dtype=float)
            coeffs[0] = -1.
            rmse = -1.
        output = np.append(coeffs, rmse)
        return output

    def find_possible_wavelength_solution(
            self,
            full_search: bool = True,
            parallel: bool = True,
            n_jobs: int = -1,
            backend: str = 'loky',
    ):
        """
        Find the possible wavelength solution using the M-m-N-n algorithm.
        """
        # NOTE:
        # If full_search is True, then the return degree of the polynomial
        # could be larger than "deg".
        # full search: try to find the best solution. Could take a while.
        if full_search:
            inputs = []
            for n in range(
                    self.deg+1,
                    min(len(self.n_peak_ys), len(self.m_known_wls))+1
            ):
                # possible combination
                ys_poss_comb = np.array(list(
                    combinations(self.n_peak_ys, n)
                ))
                wls_poss_comb = np.array(list(
                    combinations(self.m_known_wls, n)
                ))
                # all possible pairs of combinations
                inputs += list(product(
                    ys_poss_comb, wls_poss_comb
                ))
            del n, ys_poss_comb, wls_poss_comb
        # NOTE:
        # if the initial guess is good enough:
        # (1) e.g., "m_known_wls" and "n_peak_ys" exactly match each other,
        #     corresponding to [deg+1] known lines.
        # (2) e.g., at least [deg+1] lines of "m_known_wls" are
        #     included in those of "n_peak_ys"
        else:
            # possible combination
            ys_poss_comb = np.array(list(
                combinations(self.n_peak_ys, self.deg+1)
            ))
            wls_poss_comb = np.array(list(
                combinations(self.m_known_wls, self.deg+1)
            ))
            # all possible pairs of combinations
            inputs = list(product(
                ys_poss_comb, wls_poss_comb
            ))
            del ys_poss_comb, wls_poss_comb

        # fit for all possible pairs of combinations
        if len(inputs):
            outputs = prun(
                function=self._fit_wavelength_solution,
                inputs=inputs,
                parallel=parallel, n_jobs=n_jobs, backend=backend,
            )
            # find the best coefficients
            rmses = [outputs[i][-1] for i in range(len(outputs))]
            rmse = np.nanmin(rmses)
            poss_coeffs = outputs[np.nanargmin(rmses)][:-1]
            poss_poly = self.poly_form(poss_coeffs)
        else:
            coeffs = np.full(self.deg+1, 0., dtype=float)
            coeffs[0] = -1.
            rmse = -1.
            poss_poly = self.poly_form(coeffs)
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
            bounds = np.percentile(residuals, [16, 84])
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
    # (2) use N_peak_wls and N_closest_wls to re-fit the polynomial.
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

    # -------------------------------------------------------------------------
    # core function for wavelength calibration
    # -------------------------------------------------------------------------

    def wavelength_calibration(
            self,
            full_search: bool = True,
            parallel: bool = True,
            n_jobs: int = -1,
            backend: str = 'loky',
            refine_until_convergence: bool = True,
            refine_tolerance: float = 1e-8,
    ):
        """
        Perform wavelength calibration using the M-m-N-n algorithm.
        """
        # NOTE: this is the core function for wavelength calibration

        # find the possible wavelength solution
        poss_poly, rmse = self.find_possible_wavelength_solution(
            full_search=full_search,
            parallel=parallel,
            n_jobs=n_jobs,
            backend=backend,
        )

        # keep refining the solution until convergence
        if refine_until_convergence:
            while True:
                # use N_peak_ys and M_known_wls to refine the solution
                old_rmse = float(rmse)
                poss_poly, rmse = self.refine_possible_solution(
                    poss_poly
                )
                if np.isclose(old_rmse, rmse, atol=refine_tolerance):
                    break
            rmse = self.calculate_fitting_rmse(poss_poly)

        # store the final result
        self.solution = poss_poly
        self.solution_rmse = rmse

        # get and store the inverse solution (i.e., from wavelength to y)
        y_coor_min = np.min(self.peak_y_coor_lim).astype(float)
        y_coor_max = np.max(self.peak_y_coor_lim).astype(float)
        if not np.isnan(y_coor_min) and not np.isnan(y_coor_max):
            self.inv_solution = lambda wl: inverse_wavelength_solution(
                poss_poly, wl, y_min=y_coor_min, y_max=y_coor_max
            )
        else:
            self.inv_solution = None

        # check the monotonicity of the solution
        self.solution_monotonicity = check_solution_monotonicity(
            self.solution,
            y_min=y_coor_min,
            y_max=y_coor_max,
        )

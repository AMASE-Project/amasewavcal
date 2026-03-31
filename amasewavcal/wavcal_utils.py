#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         mmnn.py
@Author:       AMASE team
@Website:      https://github.com/AMASE-Project
@Description:  Wavelength calibration functions for the AMASE project.
'''

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


def detect_lines(spectrum, N=100, n=20, smooth=True, smooth_sigma=1.):
    """
    For given uncalibrated spectrum, this function detects N significant peaks,
    and further selects the n strongest peaks.
    """
    # default output
    N_peak_ys = np.array([], dtype=float)
    N_peak_fluxes = np.array([], dtype=float)
    n_peak_ys = np.array([], dtype=float)
    n_peak_fluxes = np.array([], dtype=float)
    # whether to smooth the spectrum before line detection
    if smooth:
        spectrum = gaussian_filter1d(spectrum, sigma=smooth_sigma)
    # detect peaks
    peaks, _ = find_peaks(spectrum)
    if len(peaks) == 0:
        return N_peak_ys, N_peak_fluxes, n_peak_ys, n_peak_fluxes
    ys = np.arange(len(spectrum))[peaks]
    fluxes = spectrum[peaks]
    # sort by flux
    sorted_indices = np.argsort(fluxes)[::-1]
    ys = ys[sorted_indices]
    fluxes = fluxes[sorted_indices]
    del sorted_indices
    # only keep the N strongest peaks
    N = min(N, len(ys))
    N_peak_ys = ys[:N]
    N_peak_fluxes = fluxes[:N]
    # only keep the n strongest peaks
    n = min(n, len(ys))
    n_peak_ys = ys[:n]
    n_peak_fluxes = fluxes[:n]
    return N_peak_ys, N_peak_fluxes, n_peak_ys, n_peak_fluxes


def inverse_wavelength_solution(
        solution, wl, y_min=0., y_max=9600., atol=1e-5,
):
    """ Inverse the wavelength solution to get y from wavelength. """
    y_min, y_max = float(y_min), float(y_max)
    y_mid = (y_min + y_max) / 2.
    while np.abs(y_max - y_min) > atol:
        wl_mid = solution(y_mid)
        if wl_mid < wl:
            y_min = y_mid
        else:
            y_max = y_mid
        y_mid = (y_min + y_max) / 2.
    return y_mid


def find_nearest(array_1, array_2):
    """
    For each element in array_1, find the nearest element in array_2.
    Return the elements in array_1, the nearest elements in array_2,
    and the indices of the nearest elements in array_2.
    """
    indices = []
    for val in array_1:
        idx = (np.abs(array_2 - val)).argmin()
        indices.append(idx)
    indices = np.array(indices, dtype=int)
    return array_1, array_2[indices], indices


def check_solution_monotonicity(solution, y_min, y_max):
    """
    Check whether the wavelength solution is monotonic in the given range.
    Return 1 if wl increases with y, -1 if wl decreases with y,
    and 0 if not monotonic.
    """
    ys = np.arange(y_min, y_max + 1)
    wls = solution(ys)
    diffs = np.diff(wls)
    if np.all(diffs > 0):
        return 1
    elif np.all(diffs < 0):
        return -1
    else:
        return 0


def fit_solution(ys, wls, deg=2, poly_form=np.polynomial.Polynomial):
    """
    Fit a polynomial wavelength solution to the given (ys, wls).
    Return the fitted function.
    """
    if len(ys) != len(wls):
        raise ValueError("ys and wls must have the same length.")
    if len(ys) < deg + 1:
        raise ValueError("Not enough data points to fit the polynomial.")
    coeffs = poly_form.fit(ys, wls, deg=deg).convert().coef
    solution = poly_form(coeffs)
    return solution


def fit_monotonic_solution(
        ys, wls, deg=2, poly_form=np.polynomial.Polynomial,
        monotonicity=1., y_min=None, y_max=None):
    """
    Fit a monotonic polynomial wavelength solution to the given (ys, wls).
    Return the fitted function.
    """
    if len(ys) != len(wls):
        raise ValueError("ys and wls must have the same length.")
    if len(ys) < deg + 1:
        raise ValueError("Not enough data points to fit the polynomial.")
    # preprocess
    if y_min is None:
        y_min = np.min(ys)
    if y_max is None:
        y_max = np.max(ys)

    # the objective function to minimize
    def objective(coeffs):
        return np.sum((poly_form(coeffs)(ys) - wls) ** 2)  # MSE loss

    # initial guess for the coefficients
    ini_coeffs = poly_form.fit(ys, wls, deg=deg).convert().coef

    # the constraint function to ensure monotonicity
    def constraint(coeffs):
        y_test = np.arange(y_min, y_max + 1)
        wls_test = poly_form(coeffs)(y_test)
        diffs = np.diff(wls_test)
        # for increasing, all diffs should be > 0
        if monotonicity == 1.:
            return np.min(diffs)
        # for decreasing, all diffs should be < 0
        elif monotonicity == -1.:
            return -np.max(diffs)
        else:
            raise ValueError("Monotonicity should be either 1 or -1.")

    # optimize the coefficients with the monotonicity constraint
    constraints = {'type': 'ineq', 'fun': constraint}
    result = minimize(objective, ini_coeffs, constraints=constraints)
    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)
    coeffs = result.x
    solution = poly_form(coeffs)
    return solution

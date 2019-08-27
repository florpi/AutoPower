"""
Methods for generating power spectrum data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import camb
from camb import model, initialpower
import numpy as np

from typing import Tuple


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_power_spectrum(h,
                       omc,
                       omb: float = 0.0486,
                       redshift: float = 0,
                       mode: str = 'linear',
                       nkpoints: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use CAMB to compute the power spectrum for the given parameters.

    Args:
        h: The reduced Hubble constant: h = H0 / (100 km / s / Mpc).
        omc: Dark matter density parameter.
        omb: Baryon density parameter.
        redshift: Redshift at which to compute the power spectrum.
        mode: Use linear Perturbation Theory or non-linear one.
        nkpoints: Number of points (in k-space) of the spectrum.

    Returns:
        A tuple (kh, pk), where kh is an array of k/h, and pk[j] is the
        value of the power spectrum at k/h[j] (for the given redshift).
    """

    # Convert input parameters to quantities used for simulation
    H0 = 100. * h
    ombh2 = omb * h ** 2
    omch2 = omc * h ** 2

    # Set up a CAMB object and define the cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params()
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)

    if mode == 'linear':

        pars.NonLinear = model.NonLinear_none
    
    elif mode == 'non-linear':
        pars.NonLinear = model.NonLinear_both

    # Compute the power spectrum (Halofit)
    pars.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    kh, z, pk= \
        results.get_matter_power_spectrum(minkh=1e-4,
                                          maxkh=1,
                                          npoints=nkpoints)

    return kh, pk[0].ravel() # 0 for only one redshift

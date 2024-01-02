"""
=================================
Spherical Harmonic Wave Transform
=================================

"""

import logging
from pathlib import Path

import numpy as np
import healpy as hp
from tqdm import tqdm
from scipy import special
from astropy import units as u
from astropy import constants as consts
from astropy import coordinates as coord
from pyuvdata import UVData

import cmasher as cmr
from matplotlib import pyplot as plt

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s %(levelname).4s ] %(message)s", "%Y-%m-%d %H:%M:%S"
)

# file_handler = logging.FileHandler("swht.log")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class SWHT:
    """
    The Beautiful Spherical Wave Harmonic Trasform

    A simple python implementation by Aman Chokshi
    Base on T. Carozzi (2015)
    Imaging on a Sphere with Interferometers: the Spherical Wave Harmonic Transform
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.451L...6C/abstract

    Attributes
    ----------
    uvdata: pyuvdata.uvdata.UVData object
        UVData object containing interferometric data


    Methods
    -------

    """

    def __init__(self, uvdata):

        print("\n")
        logger.info("Beginning SWHT Black Magic")

        self.uvdata = uvdata

        # UVW in meters
        self.uvw = self.uvdata.uvw_array

        # Data array: [Nblts, Nfreqs, Npols]
        self.data = self.uvdata.data_array

        # Frequencies in Hz
        self.freqs = self.uvdata.freq_array

        # Array of times in Julian date
        self.times = self.uvdata.time_array

        logger.info("Converting UVW coordinates from cartesian to spherical")
        try:
            uvw_cart = coord.CartesianRepresentation(
                x=self.uvw[:, 0], y=self.uvw[:, 1], z=self.uvw[:, 2]
            )

            uvw_sph = uvw_cart.represent_as(coord.SphericalRepresentation)
            # Radial distance (0 <= r <= ∞] [m]
            rad = uvw_sph.distance.value

            # Elevation angle (0 <= θ <= π] [rad]
            theta = uvw_sph.lat.value + np.pi / 2

            # Azimuthal angle (0 <= ɸ <= 2π] [rad]
            phi = uvw_sph.lon.value

            # rad, theta, phi (rtp)
            self.rtp = np.column_stack((rad, theta, phi))

        except Exception:
            logger.exception("Error during coversion to spherical uvw")

        max_baseline = self.rtp[:, 0].max()
        logger.info(f"Max Baseline: [{max_baseline:.2f} m]")
        logger.info(f"Max Frequency: [{self.freqs.max()} Hz]")

        max_lambda = consts.c.value / self.freqs.max()
        logger.info(f"Max Wavelength: [{max_lambda:.2f} m]")

        max_res = (max_lambda / max_baseline) * u.rad
        logger.info(f"Max Resolution: [{max_res:.4f}, {max_res.to(u.deg):.2f}]")

        # index l corresponds roughly to the inverse scale size in radians
        # https://physics.stackexchange.com/questions/54124/relation-between-multipole-moment-and-angular-scale-of-cmb
        # https://justinwillmert.com/articles/2020/notes-on-calculating-the-spherical-harmonics/
        # https://web.physics.utah.edu/~sommers/faq/b3.html
        self.max_ll = int(np.ceil(np.pi / max_res.value))
        logger.info(rf"Max 𝑙: [{self.max_ll:.0f}]")

    # Scipy has an inverse definition of theta and phi
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html#scipy.special.sph_harm
    def _ylm(self, l, m, theta, phi):
        return special.sph_harm(m, l, phi, theta)

    def compute_blms(self, max_ll=None, nchans=None):
        """
        The heart of SWHT - Eq: 16, 11 from T. Carozzi (2015)

        Spherical harmonic brightness coefficients
        linear combinations of cartesian visibilities!

        Parameters
        ----------
        max_ll: maximum l to use, overrided default determined by telescope res
        nchans: use the first n frequency channels instead of all
        """
        if max_ll:
            self.max_ll = max_ll
            logger.info(rf"Max 𝑙 used: [{self.max_ll:.0f}]")

        logger.info("Computing SWHT brightness coefficients")

        if nchans:
            if nchans >= self.freqs.size:
                nchans = self.freqs.size
        else:
            nchans = self.freqs.size

        logger.info(rf"Number of channels [{nchans:.0f}]")

        # Equation 16
        # Make l, m matrix of emplty values
        vlms = np.zeros(
            (self.max_ll + 1, 2 * self.max_ll + 1, nchans),
            dtype=complex,
        )

        # Wavenumber
        lambdas = consts.c.value / self.freqs.max()
        k = 2.0 * np.pi / lambdas
        kr = k * self.rtp[:, 0]

        for i in tqdm(
            np.arange(nchans),
            desc=r"ν ",
            ascii=" >=",
            colour="#5E4FA1",
            disable=False,
        ):

            for ll in tqdm(
                np.arange(self.max_ll + 1),
                leave=False,
                desc=r"𝑙 ",
                ascii=" >=",
                colour="#3287BC",
                disable=False,
            ):

                jl = special.spherical_jn(ll, kr)

                for mm in tqdm(
                    np.arange(-1 * ll, ll + 1),
                    leave=False,
                    desc=r"𝑚 ",
                    ascii=" >=",
                    colour="#66C1A4",
                    disable=False,
                ):

                    ylm_conj = np.conj(
                        self._ylm(ll, mm, self.rtp[:, 1], self.rtp[:, 2])
                    )

                    # I = XX + YY
                    II = self.data[:, :, 0] + self.data[:, :, 1]

                    vlms[ll, ll + mm, i] = ((2.0 * (k**2)) / np.pi) * np.sum(
                        II[:, i] * jl * ylm_conj, axis=0
                    )

        vlms = np.mean(vlms, axis=2)
        logger.info("SWHT brightness coefficients computed")

        # Inverting Equation 11
        self.blms = (vlms.T / (4.0 * np.pi * ((-1j) ** np.arange(self.max_ll + 1)))).T

    def plot_swht(self, NSIDE=32):
        logger.info("Making healpix image")
        B_map = np.zeros((hp.nside2npix(NSIDE)), dtype=complex)
        theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))

        for ll in tqdm(
            np.arange(self.max_ll + 1), desc=r"𝑙 ", ascii=" >=", colour="#5E4FA1"
        ):
            for mm in tqdm(
                np.arange(-1 * ll, ll + 1),
                leave=False,
                desc=r"𝑚 ",
                ascii=" >=",
                colour="#3287BC",
            ):
                B_map += self.blms[ll, ll + mm] * self._ylm(ll, mm, theta, phi)
        logger.info("Healpix image generated")

        hp.visufunc.orthview(
            B_map.real,
            rot=(0, 90, 90),
            cmap=cmr.pride,
            half_sky=True,
        )
        # plt.savefig("swht_real.png")
        plt.show()
        # plt.close()


if __name__ == "__main__":

    uvd = UVData.from_file(
        "data/EDA2_haslam_band01.uvfits",
        use_future_array_shapes=True,
        run_check=False
        # "data/old/EDA2_haslam_band01.uvfits", use_future_array_shapes=True, run_check=False
    )
    logger.info("Manually setting integration time")
    uvd.integration_time = 10 * np.ones(uvd.Nblts)

    # uvd = UVData.from_file(
    #     "data/nbarry/EDA2_prior_mono_si_gp15_float_2s_80kHz_hbeam__band01.uvfits",
    #     use_future_array_shapes=True,
    #     run_check=False,
    # )

    s = SWHT(uvdata=uvd)
    s.compute_blms(max_ll=64, nchans=None)
    s.plot_swht(NSIDE=64)

    # alm = []
    # # Healpy alm indexing order
    # # https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.Alm.getidx.html
    # for m in np.arange(max_l + 1):
    #     for l in np.arange(m, max_l + 1):
    #         # print(f"m: {m}, l:{l}, [{b_lm[l, l+m]}]")
    #         alm.append(b_lm[l, l+m])
    #
    # B_map = hp.alm2map(np.asarray(alm), NSIDE)
    # # hp.mollview(B_map, cmap="Spectral")
    # hp.visufunc.orthview(B_map, rot=(0, 90, 90), cmap=cmr.pride, half_sky=True, bgcolor='black')
    # plt.show()

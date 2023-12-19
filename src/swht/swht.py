"""
=================================
Spherical Harmonic Wave Transform
=================================

Base on T. Carozzi (2015)
Imaging on a Sphere with Interferometers: the Spherical Wave Harmonic Transform
https://ui.adsabs.harvard.edu/abs/2015MNRAS.451L...6C/abstract
"""

import logging
from pathlib import Path

import numpy as np
import healpy as hp
from tqdm import tqdm
from scipy import special
from astropy.io import fits
from astropy import units as u
from astropy import constants as consts
from astropy import coordinates as coord

import cmasher as cmr
from matplotlib import pyplot as plt

plt.style.use("dark_background")

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s %(levelname).4s ] %(message)s", "%Y-%m-%d %H:%M:%S"
)

# file_handler = logging.FileHandler("swht.log")
# file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class UVFITS:
    """
    A class to read uvfits files

    Attributes
    ----------
    path: str
        path to uvfits file

    hdr: astropy.io.fits.header.Header
        uvfits header

    data: astropy.io.fits.hdu.groups.GroupData
        raw uvfits data

    freqs: np.ndarray
        frequencies in uvfits file [Hz]

    uu: np.ndarray
        baseline components UU [m]

    vv: np.ndarray
        baseline components VV [m]

    ww: np.ndarray
        baseline components WW [m]

    XX: np.ndarray
        Complex instumental visibility per baseline, per time

    XY: np.ndarray
        Complex instumental visibility per baseline, per time

    YX: np.ndarray
        Complex instumental visibility per baseline, per time

    YY: np.ndarray
        Complex instumental visibility per baseline, per time

    II: np.ndarray
        Complex stokes visibility per baseline, per time

    QQ: np.ndarray
        Complex stokes visibility per baseline, per time

    UU: np.ndarray
        Complex stokes visibility per baseline, per time

    VV: np.ndarray
        Complex stokes visibility per baseline, per time

    Methods
    -------
    get_freqs()
        Extract frequencies from uvfits file

    get_uvw()
        Extract UU, VV, WW from uvfits file

    get_visi()
        Extract instrumental and stokes visibilities from uvfits file

    """

    def __init__(self, path):
        self.path = Path(path)

        logger.info(f"Reading in '{str(self.path)}'")
        try:
            with fits.open(str(self.path)) as hdul:
                self.hdr = hdul[0].header
                self.data = hdul[0].data

        except FileNotFoundError:
            logger.exception("FileNotFoundError")

        except Exception:
            logger.exception("Probably an invalid fits file")

    def get_freqs(self):
        """Get frequencies from uvfits"""

        logger.info("Getting uvfits frequencies")
        try:
            cent_freq = self.hdr["CRVAL4"]
            logger.info(f"Central frequency: {cent_freq} Hz")

            # Subtract one because this is one indexed not zero
            cent_pix = self.hdr["CRPIX4"] - 1
            freq_res = self.hdr["CDELT4"]
            logger.info(f"Frequency resolution: {freq_res} Hz")

            num_freqs = self.data.data.shape[3]
            logger.info(f"Number of channels: {num_freqs}")

            freqs = cent_freq + (np.arange(num_freqs) - cent_pix) * freq_res

            self.freqs = freqs * u.Hz

        except AttributeError:
            logger.exception("AttributeError")

        except Exception:
            logger.exception("Maybe a different data structure")

    def get_uvw(self):
        """Get uvw values from uvfits file in [m]"""

        logger.info("Getting uvfits UU, VV, WW")
        try:
            self.uu = self.data["UU"] * consts.c.value
            self.vv = self.data["VV"] * consts.c.value
            self.ww = self.data["WW"] * consts.c.value

        except AttributeError:
            logger.exception("AttributeError")

        except Exception:
            logger.exception("Maybe a different data structure")

    def uvw_spherical(self):
        """Convert cartesian uvw coords to spherical"""

        logger.info("Converting UVW coordinates from cartesian to spherical")
        try:
            uvw = coord.CartesianRepresentation(
                x=self.uu, y=self.vv, z=self.ww, unit=u.meter
            )

            sph_uvw = uvw.represent_as(coord.SphericalRepresentation)
            # Radial distance (0 <= r <= ∞] [m]
            self.r = sph_uvw.distance

            # Elevation angle (0 <= θ <= π] [rad]
            self.theta = sph_uvw.lat + np.pi/2*u.rad

            # Azimuthal angle (0 <= ɸ <= 2π] [rad]
            self.phi = sph_uvw.lon

        except AttributeError:
            logger.exception("AttributeError")

        except Exception:
            logger.exception("Maybe a different data structure")

    def get_visi(self):
        """Get instumental and stokes visibilities from uvfits file

        First dimension is all baselines for all time steps. All baselines for
        first timestep first (aka first 8128 baselines is XX[:8128])
        """

        logger.info("Getting uvfits instumental and stokes visibilities")
        logger.warning("Double check indexing and stokes definitions")
        try:
            # Gets rid of extra data axes?
            data = np.squeeze(self.data.data)

            # Deal with edge case where there is only one fine freq channel
            if len(data.shape) == 3:
                data = data[:, np.newaxis, :, :]

            # TODO: double check crosspols
            self.XX = data[:, :, 0, 0] + 1j * data[:, :, 0, 1]
            self.YY = data[:, :, 1, 0] + 1j * data[:, :, 1, 1]
            self.XY = data[:, :, 2, 0] + 1j * data[:, :, 2, 1]
            self.YX = data[:, :, 3, 0] + 1j * data[:, :, 3, 1]

            self.II = self.XX + self.YY
            self.QQ = self.XX - self.YY
            self.UU = self.XY + self.YX
            self.VV = -1j * (self.XY - self.YX)

        except AttributeError:
            logger.exception("AttributeError")

        except Exception:
            logger.exception("Maybe a different data structure")

    def plot_uvw(self):
        """Plot 3D UVW coverage"""

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.set_title("UVW Coverage")
        ax.scatter(uvf.uu, uvf.vv, uvf.ww, c=uvf.r, marker="o", s=49, cmap=cmr.pride)

        ax.set_xlabel("$u\,\,[m]$")
        ax.set_ylabel("$v\,\,[m]$")
        ax.set_zlabel("$w\,\,[m]$")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    uvf = UVFITS(path="data/EDA2_haslam_band01.uvfits")
    uvf.get_freqs()
    uvf.get_uvw()
    uvf.uvw_spherical()
    uvf.get_visi()
    # uvf.plot_uvw()

    logger.info(f"Max Baseline: [{uvf.r.max():.2f}]")
    logger.info(f"Max Frequency: [{uvf.freqs.max()}]")

    max_lambda = (consts.c / uvf.freqs.max()).to(
        u.m, equivalencies=u.dimensionless_angles()
    )
    logger.info(f"Max Wavelength: [{max_lambda}]")

    max_res = (max_lambda / uvf.r.max()) * u.rad
    logger.info(f"Max Resolution: [{max_res:.4f}, {max_res.to(u.deg):.2f}]")

    # index l corresponds roughly to the inverse scale size in radians
    # https://physics.stackexchange.com/questions/54124/relation-between-multipole-moment-and-angular-scale-of-cmb
    # https://justinwillmert.com/articles/2020/notes-on-calculating-the-spherical-harmonics/
    # https://web.physics.utah.edu/~sommers/faq/b3.html
    max_l = int(np.ceil(np.pi / max_res.value)) * 1
    logger.info(rf"Max 𝑙: [{max_l:.0f}]")

    #####################################
    # Equation 16
    #####################################
    logger.info("Computing v_lm s")

    # Make l, m matrix of emplty values
    v_lm = np.zeros((max_l + 1, 2 * max_l + 1), dtype=complex)

    # Scipy has an inverse definition of theta and phi
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html#scipy.special.sph_harm
    def ylm(l, m, theta, phi):
        return special.sph_harm(m, l, phi, theta)

    # Wavenumber
    lambdas = (consts.c / uvf.freqs.max()).to(
        u.m, equivalencies=u.dimensionless_angles()
    )
    k = 2.0 * np.pi * u.rad / lambdas
    kr = k * uvf.r

    for l in tqdm(np.arange(max_l + 1), desc=r"𝑙 ", ascii=' >=', colour="#66C1A4", disable=False):
        jl = special.spherical_jn(l, kr.value)
        for m in tqdm(np.arange(-1 * l, l + 1), leave=False, desc=r"𝑚 ", ascii=' >=', colour="#3287BC", disable=False):
            y_lm_conj = np.conj(ylm(l, m, uvf.theta.value, uvf.phi.value))
            v_lm[l, l + m] = ((2.0 * (k.value**2)) / np.pi) * np.sum(
                    uvf.II.reshape(jl.shape) * jl * y_lm_conj, axis=0
            )
            # print(f"l: [{l}], m: [{m}], v_lm: [{np.abs(v_lm[l, l + m]):.0f}, {np.rad2deg(np.angle(v_lm[l, l + m])):.0f}]")

    #####################################
    # Inverting Equation 11
    #####################################
    logger.info("Computing b_lm s")
    b_lm = (v_lm.T / (4.0 * np.pi * ((-1j) ** np.arange(max_l + 1)))).T


    #####################################
    # Healpix Imaging
    #####################################
    logger.info("Making healpix image")
    NSIDE = 32 * 1
    B_map = np.zeros((hp.nside2npix(NSIDE)), dtype=complex)
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))

    for l in tqdm(np.arange(max_l + 1), desc=r"𝑙 ", ascii=' >=', colour="#66C1A4"):
        for m in tqdm(np.arange(-1 * l, l + 1), leave=False, desc=r"𝑚 ", ascii=' >=', colour="#3287BC"):
            B_map += b_lm[l, l+m] * ylm(l, m, theta, phi)


    # hp.visufunc.mollview(B_map.real, cmap=cmr.pride) 
    hp.visufunc.orthview(B_map.real, rot=(0, 90, 90), cmap=cmr.pride, half_sky=True, bgcolor='black')
    plt.savefig("swht_real.png")
    # plt.show()
    plt.close()

    hp.visufunc.orthview(B_map.imag, rot=(0, 90, 90), cmap=cmr.pride, half_sky=True, bgcolor='black')
    plt.savefig("swht_imag.png")
    # plt.show()
    plt.close()

    hp.visufunc.orthview(np.abs(B_map), rot=(0, 90, 90), cmap=cmr.pride, half_sky=True, bgcolor='black')
    plt.savefig("swht_amp.png")
    # plt.show()
    plt.close()

    hp.visufunc.orthview(np.angle(B_map), rot=(0, 90, 90), cmap=cmr.pride, half_sky=True, bgcolor='black')
    plt.savefig("swht_phase.png")
    # plt.show()
    plt.close()


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

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
from scipy import special
from astropy.io import fits
from astropy import constants as consts
from astropy.coordinates import CartesianRepresentation


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

            self.freqs = freqs

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


if __name__ == "__main__":

    uvf = UVFITS(path="data/EDA2_haslam_band01.uvfits")
    uvf.get_freqs()
    uvf.get_uvw()
    uvf.get_visi()

    # import cmasher as cmr
    # from matplotlib import pyplot as plt
    #
    # plt.style.use("dark_background")
    #
    # fig, ax = plt.subplots()
    # ax.hexbin(uvf.uu, uvf.vv, cmap=cmr.pride)
    # ax.set_xlabel('UU [m]')
    # ax.set_ylabel('VV [m]')
    # ax.set_aspect('equal')
    #
    # plt.tight_layout()
    # plt.show()

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("adsnet")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0.dev0"

from adsnet.models import AdsNet
from adsnet.data import AseData, collate_atomsdata
from adsnet.calculators import MLCalculator, EnsembleCalculator
from adsnet.trainer import Trainer
from adsnet.tools import relax_structure

__all__ = [
    "__version__",
    "AdsNet",
    "AseData",
    "collate_atomsdata",
    "MLCalculator",
    "EnsembleCalculator",
    "Trainer",
    "relax_structure",
]

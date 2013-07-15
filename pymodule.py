import psi4
import re
import os
import inputparser
import math
import warnings
from driver import *
from wrappers import *
from molutil import *
import p4util
from psiexceptions import *


def run_cdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    cks can be called via :py:func:`~driver.energy`.

    >>> energy('cdft')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    returnvalue = psi4.plugin('cdft.so')

    return returnvalue


# Integration with driver routines
procedures['energy']['cdft'] = run_cdft

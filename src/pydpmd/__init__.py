"""pydpmd package

Top-level exports primarily re-export the data subpackage for convenience.
Use `pydpmd.data.load` to load systems and `pydpmd.data.CLASS_MAP` to
register particle classes.
"""

from . import data

__all__ = ["data"]



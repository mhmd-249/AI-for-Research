"""The static lens roster: the nine methodological lenses as configuration data.
Lens behavior/prompts are built in later slices; this module is identity + frame
+ tool-access flags only.
"""

from .roster import ROSTER, LensConfig, get_lens_config

__all__ = ["ROSTER", "LensConfig", "get_lens_config"]

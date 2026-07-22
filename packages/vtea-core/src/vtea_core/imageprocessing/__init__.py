"""Image preprocessing: Gaussian blur, median filter, contrast, background subtraction.

Ports vtea.imageprocessing.builtin from the Java codebase. LinearUnmixing and
IJMacro (arbitrary embedded ImageJ macros) are not ported - see
PORT_PLAN.md's open question on ImageJ macro compatibility.
"""

from vtea_core.imageprocessing.filters import enhance_contrast, gaussian_blur, median_filter, subtract_background

__all__ = ["gaussian_blur", "median_filter", "enhance_contrast", "subtract_background"]

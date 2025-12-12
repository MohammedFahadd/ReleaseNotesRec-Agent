# src/distutils_shim.py
"""
Shim to satisfy `from distutils.version import LooseVersion` on Python 3.11+.

FAISS still imports distutils.version.LooseVersion, but distutils was
removed from the stdlib. We register a minimal replacement in sys.modules
so any `import distutils.version` works.
"""

import re
import sys
import types


class LooseVersion:
    """
    Minimal backport of distutils.version.LooseVersion.
    Just enough for faiss to compare version strings.
    """

    component_re = re.compile(r"(\d+|[a-z]+)", re.I)

    def __init__(self, vstring=None):
        if vstring is not None:
            self.parse(vstring)

    def __repr__(self):
        return f"LooseVersion ('{self.vstring}')"

    def parse(self, vstring):
        self.vstring = vstring
        components = []
        for part in self.component_re.findall(vstring):
            if part.isdigit():
                components.append(int(part))
            else:
                components.append(part.lower())
        self.version = components

    def _cmp(self, other):
        if isinstance(other, str):
            other = LooseVersion(other)

        v1 = self.version
        v2 = other.version

        maxlen = max(len(v1), len(v2))
        v1 = list(v1) + [0] * (maxlen - len(v1))
        v2 = list(v2) + [0] * (maxlen - len(v2))

        for a, b in zip(v1, v2):
            if a == b:
                continue
            if isinstance(a, int) and isinstance(b, int):
                return -1 if a < b else 1
            return -1 if str(a) < str(b) else 1
        return 0

    def __lt__(self, other): return self._cmp(other) < 0
    def __le__(self, other): return self._cmp(other) <= 0
    def __eq__(self, other): return self._cmp(other) == 0
    def __ne__(self, other): return self._cmp(other) != 0
    def __gt__(self, other): return self._cmp(other) > 0
    def __ge__(self, other): return self._cmp(other) >= 0


def register():
    """
    Put fake `distutils` and `distutils.version` into sys.modules
    so `from distutils.version import LooseVersion` succeeds.
    """
    if "distutils.version" in sys.modules:
        return  # already registered

    distutils_mod = types.ModuleType("distutils")
    version_mod = types.ModuleType("distutils.version")

    version_mod.LooseVersion = LooseVersion
    distutils_mod.version = version_mod

    sys.modules["distutils"] = distutils_mod
    sys.modules["distutils.version"] = version_mod


# Register immediately on import
register()
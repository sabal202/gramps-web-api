"""Helper to load a single immich_import module by file path.

Not a test module itself (no ``test_`` prefix, pytest won't collect it).

``gramps_webapi.api.resources.immich_import.regions``/``mapping`` are
designed to have zero module-level gramps/gi imports, so they can be loaded
directly via ``importlib.util.spec_from_file_location`` - this deliberately
bypasses the normal dotted import (``import gramps_webapi.api.resources...``)
which would first execute ``gramps_webapi/api/__init__.py`` (the whole REST
API blueprint file), which imports ``gramps.gen.lib`` at module scope and
therefore requires ``gi``/PyGObject to be installed. There is no ``gi`` on
this Windows dev machine (see workspace CLAUDE.md / memory
``gramps-forks-build-deploy``: gi-dependent backend tests only run in the
gi-container on the NAS) - loading by file path lets the *pure* halves of
these modules be unit-tested here anyway.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

_HERE = pathlib.Path(__file__).resolve().parent
_IMMICH_IMPORT_DIR = (
    _HERE.parent
    / "gramps_webapi"
    / "api"
    / "resources"
    / "immich_import"
)


def load_module(filename: str, module_name: str) -> types.ModuleType:
    """Load ``<repo>/gramps_webapi/api/resources/immich_import/<filename>``
    as a standalone module named *module_name*, without importing any of its
    parent packages.
    """
    path = _IMMICH_IMPORT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

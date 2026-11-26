"""Installs runtime type and shape checking for every module under `src`."""

from jaxtyping import install_import_hook

install_import_hook("src", "beartype.beartype")

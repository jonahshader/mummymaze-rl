"""The hand-written stub for `mummymaze_rust` must name what the module exports.

Checks names only: a function or `Level` member added, removed or renamed in
`crates/mummymaze/src/python.rs` without the stub fails here. A changed
argument or return type does not, so update the stub with the signature.
"""

import ast
from pathlib import Path

import mummymaze_rust

STUB = (
  Path(__file__).resolve().parent.parent / "crates" / "mummymaze" / "mummymaze_rust.pyi"
)


def _public(names: list[str]) -> set[str]:
  return {n for n in names if not n.startswith("_")}


def _is_typed_dict(node: ast.ClassDef) -> bool:
  return any(isinstance(b, ast.Name) and b.id == "TypedDict" for b in node.bases)


def _stub_tree() -> ast.Module:
  return ast.parse(STUB.read_text())


def test_stub_matches_module_exports() -> None:
  stub_names = [
    node.name
    for node in _stub_tree().body
    if isinstance(node, ast.FunctionDef)
    or (isinstance(node, ast.ClassDef) and not _is_typed_dict(node))
  ]
  # maturin re-exports the inner extension module under its own name.
  runtime = _public(dir(mummymaze_rust)) - {"mummymaze_rust"}
  assert _public(stub_names) == runtime


def test_stub_matches_level_members() -> None:
  level = next(
    node
    for node in _stub_tree().body
    if isinstance(node, ast.ClassDef) and node.name == "Level"
  )
  stub_members = [n.name for n in level.body if isinstance(n, ast.FunctionDef)]
  assert _public(stub_members) == _public(dir(mummymaze_rust.Level))


def test_installed_stub_is_current() -> None:
  """ty reads the installed copy, which is stale until the crate is reinstalled."""
  module_file = mummymaze_rust.__file__
  assert module_file is not None
  installed = Path(module_file).parent / "__init__.pyi"
  assert installed.read_text() == STUB.read_text(), (
    "run `uv pip install -e crates/mummymaze/`"
  )

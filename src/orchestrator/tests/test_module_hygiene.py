"""Static checks for the modules no unit test can import.

The training entrypoints (``src/pipelines.py`` and friends) import ``torch``,
``datasets`` and ``trl``. None of those are installed where the orchestrator
suite runs, so every test in this directory that touches them has to stub or
skip. The practical consequence is that a plain ``NameError`` in a pipeline
method is invisible here and only surfaces on the GPU VM, minutes into a
launched campaign, after the model weights have already been pulled.

That is exactly how a reference to an undefined ``logger`` reached a live
run: ``src/pipelines.py`` reports through ``print`` and has never had a
module-level logger, but nothing in the local loop said so.

These tests read the source with ``ast`` instead of importing it, so they
cost nothing and run everywhere. They are deliberately conservative - a name
is reported only when it is bound *nowhere* in its file - because a
scope-accurate checker would duplicate pyflakes, and a checker that cries
wolf gets deleted. The narrow version still catches the whole class of
"this identifier does not exist" typos.
"""

import ast
import builtins
import os
import unittest


#: Repository root, three levels up from ``src/orchestrator/tests``.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

#: The tree to check. Everything the campaign can execute lives under here.
SOURCE_ROOT = os.path.join(REPO_ROOT, "src")

#: Names that are always available without being bound in the file.
ALWAYS_DEFINED = frozenset(dir(builtins)) | frozenset({
    "__file__",
    "__name__",
    "__doc__",
    "__spec__",
    "__package__",
    "__builtins__",
    "__debug__",
})


def python_sources(root: str) -> list[str]:
  """Lists every Python file under ``root``, skipping caches.

  Args:
    root: Directory to walk.

  Returns:
    Absolute paths, sorted so failures are reported deterministically.
  """
  found = []
  for current, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    found.extend(
        os.path.join(current, f) for f in files if f.endswith(".py")
    )
  return sorted(found)


def _bind_arguments(args: ast.arguments, into: set[str]) -> None:
  """Adds every parameter name of a signature to ``into``.

  Args:
    args: The ``arguments`` node of a function or lambda.
    into: Set to add the names to, mutated in place.
  """
  for arg in args.posonlyargs + args.args + args.kwonlyargs:
    into.add(arg.arg)
  if args.vararg:
    into.add(args.vararg.arg)
  if args.kwarg:
    into.add(args.kwarg.arg)


def bound_names(tree: ast.AST) -> set[str]:
  """Collects every name the module binds, at any scope.

  Scope is deliberately ignored: a name bound inside one function counts as
  bound for the whole file. That makes the check blind to genuine
  cross-function leaks, and in exchange it never reports a name that is
  merely used somewhere unusual. Only the "bound nowhere at all" case is
  meant to be caught.

  Args:
    tree: Parsed module.

  Returns:
    The set of bound identifiers.
  """
  names: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
      names.add(node.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      for alias in node.names:
        names.add((alias.asname or alias.name).split(".")[0])
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      names.add(node.name)
      _bind_arguments(node.args, names)
    elif isinstance(node, ast.Lambda):
      _bind_arguments(node.args, names)
    elif isinstance(node, ast.ClassDef):
      names.add(node.name)
    elif isinstance(node, ast.ExceptHandler) and node.name:
      names.add(node.name)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
      names.update(node.names)
    elif isinstance(node, ast.MatchAs) and node.name:
      names.add(node.name)
    elif isinstance(node, ast.MatchStar) and node.name:
      names.add(node.name)
    elif isinstance(node, ast.MatchMapping) and node.rest:
      names.add(node.rest)
  return names


def postponed_annotations(tree: ast.Module) -> bool:
  """Reports whether the module opts into PEP 563 string annotations.

  Args:
    tree: Parsed module.

  Returns:
    True if ``from __future__ import annotations`` is present.
  """
  for node in tree.body:
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
      if any(alias.name == "annotations" for alias in node.names):
        return True
  return False


def annotation_nodes(tree: ast.AST) -> set[int]:
  """Collects the ``id()`` of every node that sits inside an annotation.

  Under PEP 563 an annotation is never evaluated, so an undefined name there
  cannot raise at runtime. ``src/pipelines.py`` relies on this: it annotates
  with ``Dict`` and ``Tuple`` while importing neither.

  Args:
    tree: Parsed module.

  Returns:
    Identities of all nodes reachable from an annotation.
  """
  inside: set[int] = set()
  for node in ast.walk(tree):
    annotations = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      annotations.append(node.returns)
      args = node.args
      for arg in args.posonlyargs + args.args + args.kwonlyargs:
        annotations.append(arg.annotation)
      if args.vararg:
        annotations.append(args.vararg.annotation)
      if args.kwarg:
        annotations.append(args.kwarg.annotation)
    elif isinstance(node, ast.AnnAssign):
      annotations.append(node.annotation)
    for annotation in annotations:
      if annotation is None:
        continue
      for child in ast.walk(annotation):
        inside.add(id(child))
  return inside


def undefined_names(path: str) -> dict[str, list[int]]:
  """Finds identifiers a file reads but never binds.

  Args:
    path: Python file to inspect.

  Returns:
    A mapping of name to the line numbers that read it. Empty when clean.
  """
  with open(path, encoding="utf-8", errors="replace") as handle:
    tree = ast.parse(handle.read(), filename=path)

  known = bound_names(tree) | ALWAYS_DEFINED
  skip = annotation_nodes(tree) if postponed_annotations(tree) else set()

  offenders: dict[str, list[int]] = {}
  for node in ast.walk(tree):
    if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
      continue
    if node.id in known or id(node) in skip:
      continue
    offenders.setdefault(node.id, []).append(node.lineno)
  return offenders


class UndefinedNameTest(unittest.TestCase):
  """Guards the pipelines against identifiers that exist nowhere."""

  def test_no_source_file_reads_an_unbound_name(self):
    """Every name read in ``src`` is bound somewhere in its own file."""
    failures = []
    for path in python_sources(SOURCE_ROOT):
      for name, lines in sorted(undefined_names(path).items()):
        relative = os.path.relpath(path, REPO_ROOT)
        failures.append(f"{relative}: {name!r} read at lines {sorted(lines)}")
    self.assertEqual(
        failures,
        [],
        "These identifiers are read but never bound in their own file. A "
        "campaign will die on the VM when the line is reached:\n  "
        + "\n  ".join(failures),
    )

  def test_checker_catches_a_planted_undefined_name(self):
    """The check fails on a file that uses an unbound name.

    Without this, a checker that silently stopped inspecting anything would
    keep passing forever.
    """
    planted = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_hygiene_fixture.py"
    )
    with open(planted, "w", encoding="utf-8") as handle:
      handle.write("def f():\n  logger.info('nope')\n")
    try:
      self.assertEqual(undefined_names(planted), {"logger": [2]})
    finally:
      os.remove(planted)

  def test_checker_accepts_bound_names(self):
    """Imports, parameters, lambdas and comprehensions all count as bound."""
    clean = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_hygiene_clean.py"
    )
    with open(clean, "w", encoding="utf-8") as handle:
      handle.write(
          "import os\n"
          "def f(a, *rest, key=None, **kw):\n"
          "  g = lambda x: x + a\n"
          "  return [g(i) for i in rest], os.sep, key, kw\n"
      )
    try:
      self.assertEqual(undefined_names(clean), {})
    finally:
      os.remove(clean)


class PipelinesLoggingConventionTest(unittest.TestCase):
  """Pins the reporting convention of the training entrypoints."""

  def test_pipelines_does_not_reference_a_logger(self):
    """``src/pipelines.py`` reports through ``print``, not a logger.

    The module has no module-level logger and its ~185 existing report sites
    all use ``print`` with a bracketed tag, which the orchestrator streams
    into the dashboard verbatim. A ``logger.info`` added by analogy with
    other files raises ``NameError`` the first time that branch is taken.
    """
    path = os.path.join(SOURCE_ROOT, "pipelines.py")
    with open(path, encoding="utf-8") as handle:
      tree = ast.parse(handle.read(), filename=path)

    uses = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in ("logger", "logging")
    ]
    self.assertEqual(
        uses,
        [],
        "src/pipelines.py has no logger; use print('[Tag] ...') instead. "
        f"Offending lines: {uses}",
    )


if __name__ == "__main__":
  unittest.main()

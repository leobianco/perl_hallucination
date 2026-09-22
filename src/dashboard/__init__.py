"""Static dashboard for Auto-PERL campaigns.

This package is a *read-only projection* of what the orchestrator already
writes to disk: ``checkpoints/<task>/<name>_state.json`` and
``reports/<name>_summary.md``. It never writes into either, never imports the
campaign engine, and never talks to a running campaign. Deleting
``src/dashboard/`` and ``scripts/dashboard.py`` restores the repository
exactly, which is the same zero-lockin guarantee the orchestrator makes.

The pipeline is deliberately one-directional::

    state files -> index -> build (static site) -> publish (HF Space)

so that the artifact you read in a browser with the VM powered off is the same
artifact you preview locally with ``scripts/dashboard.py serve``.
"""

__all__ = ["index", "links"]

"""Unit tests for the Auto-PERL static dashboard.

Run from the repository root with::

    python3 -m unittest discover -s src/dashboard/tests -t .

The suite is hermetic and stdlib-only: it writes synthetic state files into
temporary directories and never touches the network, so it runs on the same
bare ``/usr/bin/python3`` as the orchestrator's suite.
"""

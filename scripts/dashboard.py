#!/usr/bin/env python3
"""Command line entry point for the Auto-PERL static dashboard.

    python3 scripts/dashboard.py build            # emit ./site
    python3 scripts/dashboard.py serve            # build, then preview locally

The dashboard is a read-only projection of ``checkpoints/`` and ``reports/``.
It never writes into either and never touches a running campaign.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import os
import socketserver
import sys

# Allow running as `python3 scripts/dashboard.py` from the repository root.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# pylint: disable-next=g-import-not-at-top
from src.dashboard import build as build_mod
# pylint: disable-next=g-import-not-at-top
from src.dashboard import publish as publish_mod
# pylint: disable-next=g-import-not-at-top
from src.dashboard import watcher as watcher_mod


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
  """Adds the arguments every subcommand accepts.

  Args:
    parser: The subparser to extend.
  """
  parser.add_argument(
      "--root",
      default=".",
      help="Repository root holding checkpoints/ and reports/ (default: .).",
  )
  parser.add_argument(
      "--out",
      default="site",
      help="Directory the static site is written to (default: ./site).",
  )
  parser.add_argument(
      "--no-archived",
      action="store_true",
      help="Exclude campaigns archived by `run --fresh`.",
  )


def _add_publish_arguments(parser: argparse.ArgumentParser) -> None:
  """Adds the arguments the publishing subcommands share.

  Args:
    parser: The subparser to extend.
  """
  parser.add_argument(
      "--repo-id",
      default=os.environ.get("AUTO_PERL_SPACE"),
      help="Target Hugging Face Space, e.g. leobianco/auto-perl. Defaults to "
      "$AUTO_PERL_SPACE.",
  )
  parser.add_argument(
      "--backend",
      choices=("hf_space", "dir"),
      default="hf_space",
      help="Where to publish (default: hf_space).",
  )
  parser.add_argument(
      "--destination",
      help="Target directory for the 'dir' backend.",
  )
  parser.add_argument(
      "--public",
      action="store_true",
      help="Create the Space public. Only applies when creating it: an "
      "existing Space never has its visibility changed.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Publish even when the content is unchanged.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Build and report what would be published, without uploading.",
  )


def _report(result: build_mod.BuildResult) -> None:
  """Prints a one-paragraph summary of a build.

  Args:
    result: The build to describe.
  """
  print(
      f"Built {result.campaign_count} campaign(s) into {result.out_dir}/ "
      f"({result.page_count} pages, snapshot {result.generated_at})."
  )
  if result.in_progress:
    print(f"  {result.in_progress} campaign(s) in progress at snapshot time.")
  if result.stale:
    print(
        f"  {result.stale} in-progress campaign(s) look stale "
        "(not written recently)."
    )
  if result.degraded_markdown:
    print(
        "  markdown-it-py is missing, so reports are shown verbatim. "
        "Install it for rendered tables."
    )


def cmd_build(args: argparse.Namespace) -> int:
  """Builds the static site.

  Args:
    args: Parsed arguments.

  Returns:
    A process exit code.
  """
  result = build_mod.build_site(
      root=args.root,
      out_dir=args.out,
      include_archived=not args.no_archived,
  )
  _report(result)
  return 0


def cmd_serve(args: argparse.Namespace) -> int:
  """Builds the site and serves it locally for preview.

  Serves the very bytes that would be published, so a local preview can never
  disagree with the deployed Space.

  Args:
    args: Parsed arguments.

  Returns:
    A process exit code.
  """
  result = build_mod.build_site(
      root=args.root,
      out_dir=args.out,
      include_archived=not args.no_archived,
  )
  _report(result)

  handler = functools.partial(
      http.server.SimpleHTTPRequestHandler, directory=os.path.abspath(args.out)
  )
  socketserver.TCPServer.allow_reuse_address = True
  with socketserver.TCPServer((args.host, args.port), handler) as httpd:
    print(f"\nServing {args.out}/ at http://{args.host}:{args.port}/")
    print("Press Ctrl-C to stop. Re-run to pick up new campaigns.")
    try:
      httpd.serve_forever()
    except KeyboardInterrupt:
      print("\nStopped.")
  return 0


def cmd_publish(args: argparse.Namespace) -> int:
  """Builds the site and publishes it, unless nothing changed.

  Args:
    args: Parsed arguments.

  Returns:
    A process exit code: 0 on success or a deliberate skip, 1 on failure.
  """
  result = build_mod.build_site(
      root=args.root,
      out_dir=args.out,
      include_archived=not args.no_archived,
  )
  _report(result)

  if args.dry_run:
    print(
        f"\n[DRY-RUN] Would publish {result.content_hash[:12]} to "
        f"{args.backend}:{args.repo_id or args.destination}."
    )
    return 0

  outcome = publish_mod.publish(
      site_dir=args.out,
      content_hash=result.content_hash,
      backend=args.backend,
      repo_id=args.repo_id,
      destination=args.destination,
      private=not args.public,
      force=args.force,
      state_file=os.path.join(args.root, publish_mod.DEFAULT_STATE_FILE),
  )
  if outcome.error:
    print(f"\nPublish failed: {outcome.error}", file=sys.stderr)
    return 1
  if not outcome.published:
    print(f"\nNothing to publish: {outcome.reason}.")
    if outcome.url:
      print(f"Current snapshot: {outcome.url}")
    return 0
  print(f"\nPublished to {outcome.url or outcome.target}")
  return 0


def cmd_watch(args: argparse.Namespace) -> int:
  """Watches the campaign tree and publishes snapshots as it changes.

  Args:
    args: Parsed arguments.

  Returns:
    A process exit code.
  """
  config = watcher_mod.WatchConfig(
      root=args.root,
      out_dir=args.out,
      include_archived=not args.no_archived,
      backend=args.backend,
      repo_id=args.repo_id,
      destination=args.destination,
      private=not args.public,
      debounce_seconds=args.debounce,
      poll_seconds=args.poll,
      state_file=os.path.join(args.root, publish_mod.DEFAULT_STATE_FILE),
      dry_run=args.dry_run,
  )
  if config.backend == "hf_space" and not config.repo_id:
    print(
        "watch needs --repo-id (or $AUTO_PERL_SPACE) for the hf_space backend.",
        file=sys.stderr,
    )
    return 2

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s  %(message)s",
      datefmt="%H:%M:%S",
  )
  watcher = watcher_mod.Watcher(config)
  watcher_mod.install_signal_handlers(watcher)
  print(watcher_mod.describe_start(config))
  try:
    watcher.run(max_ticks=1 if args.once else None)
  except KeyboardInterrupt:
    print("\nStopped.")
  return 0


def build_parser() -> argparse.ArgumentParser:
  """Builds the argument parser.

  Returns:
    The configured parser.
  """
  parser = argparse.ArgumentParser(
      prog="dashboard",
      description="Build and publish the Auto-PERL campaign dashboard.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  builder = subparsers.add_parser("build", help="Emit the static site.")
  _add_common_arguments(builder)
  builder.set_defaults(handler=cmd_build)

  server = subparsers.add_parser(
      "serve", help="Build, then preview the site over HTTP."
  )
  _add_common_arguments(server)
  server.add_argument("--port", type=int, default=8000, help="Port to bind.")
  server.add_argument(
      "--host",
      default="127.0.0.1",
      help="Address to bind. Defaults to localhost; use an SSH tunnel to "
      "reach it from elsewhere rather than binding a public interface.",
  )
  server.set_defaults(handler=cmd_serve)

  publisher = subparsers.add_parser(
      "publish", help="Build, then upload the snapshot to a static host."
  )
  _add_common_arguments(publisher)
  _add_publish_arguments(publisher)
  publisher.set_defaults(handler=cmd_publish)

  watch = subparsers.add_parser(
      "watch",
      help="Publish a new snapshot whenever a campaign changes.",
  )
  _add_common_arguments(watch)
  _add_publish_arguments(watch)
  watch.add_argument(
      "--debounce",
      type=float,
      default=watcher_mod.DEFAULT_DEBOUNCE_SECONDS,
      help="Minimum seconds between publishes of ordinary progress. A "
      "campaign reaching a terminal status always publishes immediately, "
      "so this never delays a final result (default: %(default)s).",
  )
  watch.add_argument(
      "--poll",
      type=float,
      default=watcher_mod.DEFAULT_POLL_SECONDS,
      help="Seconds between reads of the campaign tree (default: %(default)s).",
  )
  watch.add_argument(
      "--once",
      action="store_true",
      help="Run a single poll and exit, for cron-style use.",
  )
  watch.set_defaults(handler=cmd_watch)

  return parser


def main(argv=None) -> int:
  """Parses arguments and dispatches to a subcommand.

  Args:
    argv: Argument vector, defaulting to ``sys.argv[1:]``.

  Returns:
    A process exit code.
  """
  args = build_parser().parse_args(argv)
  return args.handler(args)


if __name__ == "__main__":
  raise SystemExit(main())

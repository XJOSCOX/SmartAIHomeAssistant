"""Deliberate local visitor administration; never opens a camera or enrolls residents."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from jake.config import load_app_config
from jake.visitors import visitor_match


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage encrypted anonymous visitor memory")
    parser.add_argument("--config", type=Path, default=Path("config/local.toml"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--migrate-store", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--delete", metavar="VISITOR_UUID")
    action.add_argument("--delete-all", action="store_true")
    action.add_argument("--label", metavar="VISITOR_UUID")
    action.add_argument("--remove-label", metavar="VISITOR_UUID")
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    if bool(args.label) != (args.name is not None):
        parser.error("--label requires --name; --name is only allowed with --label")
    try:
        from jake.adapters.visitor_store import EncryptedVisitorStore

        config = load_app_config(args.config)
        store = EncryptedVisitorStore(
            Path(config.identity.store_path), timezone=config.home.timezone
        )
        if args.migrate_store:
            print(store.migrate())
            return 0
        store.expire(datetime.now(UTC), config.visitors.retention_days)
        if args.list:
            print(
                f"Timezone: {config.home.timezone} "
                f"| recurring >= {config.visitors.recurring_distinct_days} "
                f"days | frequent >= {config.visitors.frequent_distinct_days} days"
            )
            for p in store.profiles():
                print(
                    f"{p.visitor_id} | {p.display_name or 'ANONYMOUS'} "
                    f"| {visitor_match(p, config.visitors).state} "
                    f"| Sessions={p.session_count} | Visit Days={p.distinct_visit_days} "
                    f"| completed sessions={p.statistics.completed} "
                    f"| session mean={p.statistics.mean_seconds:.1f}s "
                    f"| variance={p.statistics.variance_seconds:.1f}s^2"
                )
        elif args.delete:
            store.delete(args.delete)
        elif args.delete_all:
            store.delete_all()
        else:
            store.label(
                args.label or args.remove_label,
                args.name.strip() if args.name is not None else None,
            )
        return 0
    except Exception as exc:
        print(f"Visitor management error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

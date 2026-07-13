"""Compatibility entry point for the consolidated core CLI."""

from .core import build_parser, main, parse_time_value

__all__ = ["build_parser", "main", "parse_time_value"]


if __name__ == "__main__":
    raise SystemExit(main())

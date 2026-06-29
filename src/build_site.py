"""
Nav bar sync utility for standalone HTML pages.

Propagates nav bar HTML from template source to all pages.
Usage:
    uv run python -m src.build_site          # sync nav
    uv run python -m src.build_site --check  # dry run
"""

import re
import sys
from pathlib import Path

DOCS = Path("docs")
NAV_PATTERN = r'<nav class="site-nav">.*?</nav>'
ACTIVE_PATTERN = r'class="active"'

PAGES = [
    ("index.html", "Home"),
    ("theory.html", "Theory"),
    ("downforce.html", "Downforce"),
    ("ride_height.html", "Ride Height"),
    ("drs_active_aero.html", "DRS & Active Aero"),
    ("track_setups.html", "Track Setups"),
    ("cornering.html", "Cornering"),
    ("strategy.html", "Strategy"),
    ("cfd_venturi.html", "CFD Venturi"),
    ("implementation.html", "Code"),
]


def get_nav_block(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(NAV_PATTERN, text, re.DOTALL)
    return match.group(0) if match else None


def set_active_class(nav_html: str, page_name: str) -> str:
    """Set the 'active' class on the correct nav link."""
    nav_html = re.sub(ACTIVE_PATTERN, "", nav_html)
    nav_html = re.sub(
        rf'href="{page_name}"',
        f'href="{page_name}" class="active"',
        nav_html,
    )
    return nav_html.strip()


def sync_nav(dry_run: bool = False):
    source_path = DOCS / "downforce.html"
    source_nav = get_nav_block(source_path)
    if not source_nav:
        print("ERROR: Could not find nav block in template.")
        sys.exit(1)

    for filename, label in PAGES:
        path = DOCS / filename
        if not path.exists():
            print(f"  SKIP  {filename} (not found)")
            continue

        old_nav = get_nav_block(path)
        if not old_nav:
            print(f"  WARN  {filename} (no nav block found)")
            continue

        new_nav = set_active_class(source_nav, filename)
        if old_nav == new_nav:
            print(f"  OK    {filename}")
            continue

        if dry_run:
            print(f"  WOULD UPDATE  {filename}")
        else:
            content = path.read_text(encoding="utf-8")
            content = content.replace(old_nav, new_nav)
            path.write_text(content, encoding="utf-8")
            print(f"  UPDATED  {filename}")


if __name__ == "__main__":
    dry_run = "--check" in sys.argv
    sync_nav(dry_run=dry_run)

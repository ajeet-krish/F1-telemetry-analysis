"""
Sidebar sync utility for standalone HTML pages.

Propagates sidebar HTML from template source to all pages.

Usage:
    uv run python -m runners.build_site          # sync sidebar
    uv run python -m runners.build_site --check  # dry run
"""

import re
import sys
from pathlib import Path

DOCS = Path("docs")
SIDEBAR_PATTERN = r'<aside class="sidebar">.*?</aside>'
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
    ("powertrain.html", "Powertrain & Aero"),
    ("cfd_venturi.html", "CFD Venturi"),
    ("implementation.html", "Code"),
]


def get_sidebar_block(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(SIDEBAR_PATTERN, text, re.DOTALL)
    return match.group(0) if match else None


def set_active_sidebar(sidebar_html: str, page_name: str) -> str:
    """Set active class on the correct nav link (skip logo link in sidebar-header)."""
    nav_match = re.search(r'(<nav class="sidebar-nav">.*?</nav>)', sidebar_html, re.DOTALL)
    if nav_match:
        nav_section = nav_match.group(1)
        cleaned_nav = re.sub(ACTIVE_PATTERN, "", nav_section)
        cleaned_nav = re.sub(r'\s+>', '>', cleaned_nav)
        new_nav = re.sub(
            rf'href="{page_name}"',
            f'href="{page_name}" class="active"',
            cleaned_nav,
        )
        sidebar_html = sidebar_html.replace(nav_section, new_nav)
    return sidebar_html


def sync_sidebar(dry_run: bool = False):
    source_path = DOCS / "downforce.html"
    source_sidebar = get_sidebar_block(source_path)
    if not source_sidebar:
        print("ERROR: Could not find sidebar block in template.")
        sys.exit(1)

    for filename, label in PAGES:
        path = DOCS / filename
        if not path.exists():
            print(f"  SKIP  {filename} (not found)")
            continue

        old_sidebar = get_sidebar_block(path)
        if not old_sidebar:
            print(f"  WARN  {filename} (no sidebar block found)")
            continue

        new_sidebar = set_active_sidebar(source_sidebar, filename)
        if old_sidebar == new_sidebar:
            print(f"  OK    {filename}")
            continue

        if dry_run:
            print(f"  WOULD UPDATE  {filename}")
        else:
            content = path.read_text(encoding="utf-8")
            content = content.replace(old_sidebar, new_sidebar)
            path.write_text(content, encoding="utf-8")
            print(f"  UPDATED  {filename}")


if __name__ == "__main__":
    dry_run = "--check" in sys.argv
    sync_sidebar(dry_run=dry_run)

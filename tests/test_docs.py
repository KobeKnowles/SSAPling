"""Every link in the docs into this repository leads somewhere: a file, and a heading if it
names one. The README's links are full GitHub addresses, so they also work on PyPI; they are
checked against the files here."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted([ROOT / "README.md", *ROOT.glob("docs/*.md"), *ROOT.glob("examples/*/README.md")])
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
REPOSITORY = re.compile(r"https://github\.com/KobeKnowles/SSAPling/(?:blob|tree)/main/")


def prose(page: Path) -> str:
    """The page without its code blocks, where brackets are code, not links."""
    return re.sub(r"^```.*?^```", "", page.read_text(encoding="utf-8"), flags=re.M | re.S)


def anchors(page: Path) -> set[str]:
    """The anchors of a page's headings, as GitHub makes them."""
    found: set[str] = set()
    for heading in re.findall(r"^#+ (.+)$", prose(page), flags=re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        found.add(slug if slug not in found else f"{slug}-1")
    return found


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(ROOT)) for p in PAGES])
def test_every_relative_link_leads_somewhere(page: Path):
    broken = []
    for target in LINK.findall(prose(page)):
        base = page.parent
        if REPOSITORY.match(target):
            target, base = REPOSITORY.sub("", target), ROOT  # a file here, by its GitHub address
        elif re.match(r"[a-z]+:", target):
            continue  # a web address elsewhere
        path, _, anchor = target.partition("#")
        linked = (base / path).resolve() if path else page
        if not linked.exists() or (anchor and anchor not in anchors(linked)):
            broken.append(target)
    assert not broken

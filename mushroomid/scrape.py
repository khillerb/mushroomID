"""Scrape species reference data from the Wild Food UK mushroom guide.

Collects an image URL, scientific name, and edibility flag per species page
listed in the site's mushroom sitemap.

Example:
    python -m mushroomid.scrape --output data/species_reference.csv --limit 20

Please keep the default delay in place -- this hits someone else's server.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://www.wildfooduk.com/custom_mushroom-sitemap.xml"
USER_AGENT = "mushroomID/0.1 (dataset research; contact via repository)"
REQUEST_TIMEOUT = 20


@dataclass
class SpeciesRecord:
    url: str
    common_name: str | None
    scientific_name: str | None
    edibility: str
    image_url: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape the Wild Food UK mushroom guide.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/species_reference.csv"),
        help="destination CSV",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N species pages")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds to wait between requests (be polite)",
    )
    return parser.parse_args(argv)


def make_soup(markup: bytes, prefer_xml: bool = False) -> BeautifulSoup:
    """Build a soup, degrading to the stdlib parser when lxml is absent."""
    if prefer_xml:
        try:
            return BeautifulSoup(markup, "xml")
        except Exception:  # noqa: BLE001 - bs4 raises FeatureNotFound for missing lxml
            pass
    return BeautifulSoup(markup, "html.parser")


def fetch(session: requests.Session, url: str) -> bytes | None:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"warning: {url} failed: {exc}", file=sys.stderr)
        return None
    return response.content


def sitemap_urls(session: requests.Session, sitemap_url: str = SITEMAP_URL) -> list[str]:
    markup = fetch(session, sitemap_url)
    if markup is None:
        return []
    soup = make_soup(markup, prefer_xml=True)
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]


def table_value(soup: BeautifulSoup, label: str) -> str | None:
    """Read the cell to the right of a labelled table cell."""
    header = soup.find("td", string=lambda text: text and text.strip() == label)
    if header is None:
        return None
    sibling = header.find_next_sibling("td")
    return sibling.get_text(strip=True) if sibling else None


def extract_edibility(soup: BeautifulSoup) -> str:
    container = soup.find("div", class_="mush-key-icon")
    icon = container.find("img") if container else None
    source = icon.get("src", "") if icon else ""

    if "edible" in source:
        return "Edible"
    if "poisonous" in source:
        return "Poisonous"
    return "Unknown"


def extract_image_url(soup: BeautifulSoup) -> str | None:
    container = soup.find("div", class_="custom_gallery_main_container")
    image = container.find("img") if container else None
    return image.get("src") if image else None


def extract_species(markup: bytes, url: str) -> SpeciesRecord:
    soup = make_soup(markup)
    heading = soup.find("h1")
    return SpeciesRecord(
        url=url,
        common_name=heading.get_text(strip=True) if heading else None,
        scientific_name=table_value(soup, "Scientific Name"),
        edibility=extract_edibility(soup),
        image_url=extract_image_url(soup),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    urls = sitemap_urls(session)
    if not urls:
        raise SystemExit("Could not read the sitemap; nothing to scrape.")

    if args.limit:
        urls = urls[: args.limit]
    print(f"{len(urls)} species pages to fetch")

    records: list[SpeciesRecord] = []
    for position, url in enumerate(urls, start=1):
        markup = fetch(session, url)
        if markup is not None:
            records.append(extract_species(markup, url))
        print(f"  [{position}/{len(urls)}] {url}")
        time.sleep(args.delay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(SpeciesRecord)])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    print(f"\nwrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

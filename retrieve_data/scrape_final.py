#!/usr/bin/env python3
"""Catalog and export selected William Morris texts from Marxists Internet Archive.

From the project root:
    python dataset_WM/scrape_final.py catalog --output morris_corpus
    # Edit catalog.json: set decision to "approved" for the works you want.
    python dataset_WM/scrape_final.py export --output morris_corpus

Use --accept-proposed on export to include clear matches without editing the
catalog. Ambiguous records still require explicit approval. Books use one
complete edition; chapters from multi-file editions are exported separately.
Separately selected chapters already in an exported book are recorded in the
manifest rather than exported twice.
Book records include book_genre, book_authorship, and views_evidence in both
the catalog and export. Rebuild older catalogs to add these labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag

DEFAULT_START = "https://www.marxists.org/archive/morris/works/"
USER_AGENT = "WilliamMorrisCorpusResearch/1.0"
PERSONAL = "personal_letters_and_autobiography"
TALKS = "speeches_talks_and_lectures"
PROSE = "essays_articles_pamphlets_and_manifestos"
INTERVIEWS = "interviews_and_reported_conversations"
NEWS = "standalone_newspaper_contributions"
BOOKS = "books"
CATEGORIES = {PERSONAL, TALKS, PROSE, INTERVIEWS, NEWS, BOOKS}
EXCLUDED_SECTIONS = {"Poems and Songs", "Translations", "Funeral and Obituaries"}
SERIAL = re.compile(
    r"\b(?:news from nowhere|a dream of john ball|the pilgrims of hope|"
    r"chants for socialists|socialism from the root up|socialism: its growth and outcome)\b", re.I
)
FICTION_OR_VERSE = {"1884/justice/01Fable.htm"}
AUTOBIO = re.compile(r"\b(?:autobiograph\w*|how i became a socialist|socialist diary)\b", re.I)
INTERVIEW = re.compile(r"\b(?:interview|talk with mr\.? william morris)\b", re.I)
REPORT = re.compile(r"\b(?:as reported|reported (?:by|in)|report on this talk)\b", re.I)
PUBLIC_LETTER = re.compile(r"\bletter to (?:the |an? )?(?:editor|clarion|daily news|times)\b", re.I)
META = re.compile(
    r"^(?:Sources?|Original publication|First published|Published|Transcrib\w*|"
    r"Proof\w*(?: and HTML)?|HTML(?: mark.?up)?|Date created|Last updated|"
    r"Editor[s]?|Publication|Translated by|Authors?|Copyright)\s*:", re.I
)
END_NOTES = re.compile(
    r"^(?:bibliographi(?:cal|c) (?:note[s]?|information)|editorial notes|"
    r"notes|footnotes|endnotes|references|list of newspaper clippings inserted.*)[.:]?$", re.I
)
LOST = re.compile(r"[\[(]\s*lost\b", re.I)
# The archive offers multiple layouts of the same edition. Choose one
# verified route per book, in reading order, rather than crawling its links.
BOOK_EDITIONS = {
    f"{directory}/index.htm": (f"{directory}/{filename}.htm",)
    for directory, filename in (
        ("1870/volsungs", "volsungs"), ("1884/principles", "principles_xml"),
        ("1885/chants", "chants"), ("1885/pilgrim", "pilgrim"),
        ("1886/johnball", "johnball"), ("1888/wolfings", "wolfings"),
        ("1888/signs", "signs"), ("1889/roots", "roots"),
        ("1890/nowhere", "nowhere"), ("1890/glittpln", "glittpln"),
        ("1891/bytheway", "bytheway"),
    )
}
BOOK_EDITIONS.update({
    "1882/hopes/index.htm": tuple(f"1882/hopes/chapters/chapter{i}.htm" for i in range(1, 6)),
    "1892/wwend/index.htm": tuple(f"1892/wwend/wwend{i}.htm" for i in range(1, 5)),
    "1893/sgo/index.htm": tuple(
        f"1893/sgo/{part}.htm" for part in
        ("preface", "intro", *(f"ch{i:02d}" for i in range(1, 22)), "city")
    ),
})
# Keyed by the Works index entry (after choosing a complete edition), not title keywords.
BOOK_METADATA = {
    # genre, authorship, evidence of Morris's own views
    "1870/volsungs/index.htm": ("saga", "co_translation", "not_original"),
    "1882/hopes/index.htm": ("nonfiction", "sole_author", "direct"),
    "1884/principles/index.htm": ("nonfiction", "coauthored", "joint"),
    "1885/chants/index.htm": ("poetry", "sole_author", "expressive"),
    "1885/pilgrim/index.htm": ("poetry", "sole_author", "fictional_exploration"),
    "1886/johnball/index.htm": ("fiction", "sole_author", "fictional_exploration"),
    "1886/kingless.htm": ("fiction", "sole_author", "fictional_exploration"),
    "1888/wolfings/index.htm": ("fiction", "sole_author", "thematic"),
    "1888/signs/index.htm": ("nonfiction", "sole_author", "direct"),
    "1889/roots/index.htm": ("fiction", "sole_author", "thematic"),
    "1890/nowhere/index.htm": ("fiction", "sole_author", "fictional_exploration"),
    "1890/glittpln/index.htm": ("fiction", "sole_author", "thematic"),
    "1891/bytheway/index.htm": ("poetry", "sole_author", "expressive"),
    "1892/wwend/index.htm": ("fiction", "sole_author", "thematic"),
    "1893/sgo/index.htm": ("nonfiction", "coauthored", "joint"),
    "1894/wdbydwld.htm": ("fiction", "sole_author", "thematic"),
    "1894/chants.htm": ("poetry", "sole_author", "expressive"),
    "1895/chilcris.htm": ("fiction", "sole_author", "thematic"),
}
LINKED_TEXTS = {
    "1883/riches/": ("1883/riches/riches.htm",),
    "1884/as/index.htm": ("1884/as/as.htm",),
    "1884/art-lab-intro.htm": ("1884/art-lab.htm",),
    "1884/hwl/index.htm": ("1884/hwl/hwl.htm",),
    "1887/diary/": ("1887/diary/diary.htm#diary",),
    "1887/societyfuture.htm#6apr1889": ("1887/societyfuture.htm",),
    "1887/societyfuture.htm#13apr1889": ("1887/societyfuture.htm",),
    "1888/ghent/index.htm": ("1888/ghent/ghent.htm",),
    "1889/sa/index.htm": ("1889/sa/sa.htm",),
    "1894/hibs/index.htm": ("1894/hibs/hibs.htm",),
}
MULTIPART = (
    ("A Factory as it Might Be", ("1884/justice/10fact1.htm", "1884/justice/12fact2.htm",
                                 "1884/justice/13fact3.htm")),
    ("The Reward of Labour: A Dialogue", ("1887/commonweal/05-dialogue.htm",
                                         "1887/commonweal/05-reward-labour.htm")),
    ("Whigs Astray", ("1889/commonweal/01-whigs-astray-1.htm",
                     "1889/commonweal/01-whigs-astray-2.htm")),
    ("The Development of Modern Society", tuple(f"1890/commonweal/devel{i}.htm" for i in range(1, 6))),
)
PARTS = {part: (title, urls) for title, urls in MULTIPART for part in urls}
PUBLIC_PRESS = {"1876/turks.htm", "1889/sa/index.htm", "1894/business.htm"}
REPORTED_TALKS = {"1893/freedom.htm", "1894/artlabour.htm"}
COAUTHORED_PATH = "/archive/bax/1886/commune/commune.htm"


def tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allowed_url(url: str, root: str) -> bool:
    target, base = urlsplit(url), urlsplit(root)
    return (target.scheme in {"https", "http"} and target.netloc == base.netloc
            and (target.path.startswith(base.path) or target.path == COAUTHORED_PATH)
            and ".." not in target.path.split("/") and "%" not in target.path
            and target.path.endswith(("/", ".htm", ".html")) and not target.query)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class Archive:
    def __init__(self, root: str, delay: float) -> None:
        self.root = root
        self.delay = max(1.25, delay)
        self.last_request = 0.0
        self.opener = build_opener(NoRedirect())
        self.robots = RobotFileParser()
        robots, _ = self.request(urljoin(root, "/robots.txt"))
        self.robots.parse(robots.decode("utf-8-sig").splitlines())
        self.delay = max(self.delay, self.robots.crawl_delay(USER_AGENT) or 0)

    def request(self, url: str) -> tuple[bytes, str]:
        for attempt in range(3):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            request = Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with self.opener.open(request, timeout=40) as response:
                    return response.read(), response.headers.get_content_type()
            except HTTPError as exc:
                if attempt == 2 or exc.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = exc.headers.get("Retry-After", "")
                if retry_after and not retry_after.isdecimal():
                    raise
                time.sleep(max(2 ** attempt, int(retry_after or "0")))
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("Retry loop exhausted")

    def fetch(self, url: str) -> bytes:
        if not allowed_url(url, self.root) or not self.robots.can_fetch(USER_AGENT, url):
            raise PermissionError(f"Out of scope or disallowed by robots.txt: {url}")
        html, content_type = self.request(url)
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError(f"Expected HTML at {url}, got {content_type}")
        return html


def heading(item: Tag) -> str:
    label = item.find("strong", recursive=False)
    return tidy(label.get_text(" ", strip=True)) if label else ""


def works_items(html: bytes) -> Iterator[tuple[str, str, Tag]]:
    soup = BeautifulSoup(html, "html.parser")
    work_heading = soup.find("h2", string=re.compile(r"^\s*Works\s*$"))
    years = work_heading.find_next_sibling("ul") if work_heading else None
    if not years:
        raise ValueError("Works index layout not recognized")
    for year_item in years.find_all("li", recursive=False):
        year = heading(year_item)
        if not re.fullmatch(r"18\d\d", year):
            continue
        sections = year_item.find("ul", recursive=False)
        if not sections:
            continue
        for section_item in sections.find_all("li", recursive=False):
            section = heading(section_item)
            entries = section_item.find("ul", recursive=False)
            if section and entries:
                for item in entries.find_all("li", recursive=False):
                    yield year, section, item


def journalism_items(html: bytes) -> Iterator[tuple[str, str, Tag]]:
    soup = BeautifulSoup(html, "html.parser")
    sections = [h for h in soup.find_all("h2") if h.get_text(" ", strip=True).startswith("Articles for")]
    if not sections:
        raise ValueError("Journalism index layout not recognized")
    for section in sections:
        years = section.find_next_sibling("ul")
        if not years:
            raise ValueError(f"No entries for {section.get_text(' ', strip=True)}")
        for year_item in years.find_all("li", recursive=False):
            year = heading(year_item)
            entries = year_item.find("ul", recursive=False)
            if re.fullmatch(r"18\d\d", year) and entries:
                for item in entries.find_all("li", recursive=False):
                    yield year, section.get_text(" ", strip=True), item


def classify(section: str, context: str, path: str, newspaper: bool) -> tuple[str, str, str]:
    if section in EXCLUDED_SECTIONS:
        return "", "excluded", f"Outside scope: {section}"
    if SERIAL.search(context) and (newspaper or "serialization" in context.lower()):
        return "", "excluded", "Book-length serial or verse"
    if path in FICTION_OR_VERSE:
        return "", "excluded", "Fiction or verse"
    if LOST.search(context) or not path:
        return "", "unavailable", "No linked text for this entry"
    if section == "Books":
        if path.startswith("1870/volsungs/"):
            return BOOKS, "review", "Co-translated saga; not original Morris prose"
        if path == "1884/sketchley.htm":
            return PROSE, "proposed", "Morris's introduction, not Sketchley's book"
        if path not in BOOK_METADATA:
            return BOOKS, "review", "Verify book genre, authorship, and relation to Morris's views"
        return BOOKS, "proposed", "Complete book text"
    if INTERVIEW.search(context):
        return INTERVIEWS, "proposed", "Interview; may include reporter narration"
    if AUTOBIO.search(context):
        if path == "1887/diary/":
            return PERSONAL, "review", "Edited diary; verify the text and editorial material"
        return PERSONAL, "proposed", "Autobiographical writing"
    if section == "Talks and Speeches":
        return TALKS, "proposed", "Listed as a talk or speech"
    if path in PUBLIC_PRESS or PUBLIC_LETTER.search(context):
        return NEWS, "proposed", "Identified press contribution"
    if newspaper:
        if re.search(r"\b(?:anon(?:ymous)?|attributed)\b", context, re.I):
            return NEWS, "review", "Attribution needs checking"
        if re.search(r"\bseriali[sz]ation\b", context, re.I) and path not in PARTS:
            return NEWS, "review", "Check whether this is standalone nonfiction"
        return NEWS, "proposed", "Individual entry in the journalism index or press section"
    if section == "Letters":
        return PERSONAL, "review", "Verify private recipient or reclassify as a public letter"
    if section == "Articles, Pamphlets and Manifestos":
        return PROSE, "proposed", "Listed as prose, pamphlet or manifesto"
    return "", "review", f"Check genre and attribution: {section}"


def catalog_rows(html: bytes, source: str, root: str, journalism: bool) -> Iterator[dict]:
    items = journalism_items(html) if journalism else works_items(html)
    for year, section, item in items:
        context = tidy(item.get_text(" ", strip=True)).removesuffix(" *")
        newspaper = journalism or "Articles for " in section
        anchors = [(a.get_text(" ", strip=True), urljoin(source, str(a["href"])))
                   for a in item.find_all("a", href=True)
                   if not str(a["href"]).startswith("#") and a.get_text(strip=True) not in {"", "*"}]
        if any(re.match(r"full list of articles\b", label, re.I) for label, _ in anchors):
            continue
        if section == "Books":
            complete = next(((label, url) for label, url in anchors if label.casefold() == "one big file"), None)
            if complete:
                anchors = [(context.split("(", 1)[0].strip(), complete[1])]
        lost = bool(LOST.search(context))
        # A lost entry can link to a related surviving work, not to its own text.
        for label, url in (anchors if anchors and not lost else [(context, "")]):
            path = url.removeprefix(root) if url else ""
            if path == "1882/hopes/index.htm" and section == "Talks and Speeches":
                continue
            category, decision, reason = classify(section, context, path, newspaper)
            title, paths = PARTS.get(path, (label, BOOK_EDITIONS.get(path, LINKED_TEXTS.get(path, ()))))
            if re.search(r"\b(?:published as|published in)\b", context, re.I) and item.em:
                title = item.em.get_text(" ", strip=True)
            urls = [urljoin(root, part) for part in paths] if paths else ([url] if url else [])
            if decision not in {"excluded", "unavailable"} and (
                any(not allowed_url(target, root) for target in urls)
                or (not paths and (url.endswith(("/", "/index.htm", "/index.html"))
                                   or path.endswith("-intro.htm")))
            ):
                decision, reason = "review", "Resolve the text URL or verify the landing page"
            voice = ("translation_not_original_voice" if path.startswith("1870/volsungs/") else
                     "mixed_speakers" if category == INTERVIEWS else
                     "reported_by_third_party" if REPORT.search(context) or path in REPORTED_TALKS else
                     "coauthored_or_uncertain" if re.search(r"\(with\b|, with\b", context, re.I) else
                     "morris_attribution_not_independently_verified")
            row = {
                "title": tidy(title), "year": year, "category": category, "decision": decision,
                "reason": reason, "text_urls": urls, "voice_type": voice,
                "partial": bool(re.search(r"\bportions\b|\bexcerpt\b", context, re.I)),
                "sources": [{"index_url": source, "archive_heading": section,
                             "entry": context, "entry_url": url}],
            }
            if category == BOOKS:
                genre, authorship, evidence = BOOK_METADATA.get(path, ("unknown", "unknown", "unknown"))
                row.update(book_genre=genre, book_authorship=authorship, views_evidence=evidence)
            yield row


def merge_catalog(rows: Iterator[dict]) -> list[dict]:
    records: dict[tuple, dict] = {}
    priority = {BOOKS: 6, INTERVIEWS: 5, PERSONAL: 4, TALKS: 3, NEWS: 2, PROSE: 1, "": 0}
    for row in rows:
        key = (("excluded",) if row["decision"] == "excluded" else ()) + (
            tuple(row["text_urls"]) or (row["year"], row["title"])
        )
        previous = records.get(key)
        if previous is None:
            records[key] = row
            continue
        prefer_row = (
            previous["decision"] == "review" and row["decision"] == "proposed"
            or previous["decision"] == row["decision"]
            and priority[row["category"]] > priority[previous["category"]]
        )
        primary, secondary = (row, previous) if prefer_row else (previous, row)
        sources = list(previous["sources"])
        for source in row["sources"]:
            if source not in sources:
                sources.append(source)
        records[key] = {**primary, "sources": sources,
                        "partial": primary["partial"] or secondary["partial"]}
    return list(records.values())


def content_text(html: bytes, url: str, *, book: bool = False) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    fragment = urlsplit(url).fragment
    if fragment and not (soup.find(id=fragment) or soup.find("a", attrs={"name": fragment})):
        raise ValueError(f"Text anchor #{fragment} not found")
    main = (soup.select_one("article") or soup.select_one("main") or soup.select_one("#content")
            or soup.select_one(".article-content") or soup.body or soup)
    for el in main.select("script, style, noscript, nav, footer, aside, form, "
                          ".menu, .chronology, .navigation, .navbar, .footer, .breadcrumbs, "
                          "#navigation, #footer, #sidebar, #breadcrumbs, "
                          "sup:has(a[href*='footnote'])"):
        el.decompose()
    paragraphs: list[str] = []
    notes: list[str] = []
    in_text = not fragment
    notes_level = 0
    headings = {"h1", "h2", "h3", "h4", "h5", "h6"}
    blocks = {"p", "blockquote", "pre", "li", *headings}
    for el in main.find_all(True):
        if fragment and (el.get("id") == fragment or el.get("name") == fragment):
            in_text = True
        if not in_text or el.name not in blocks:
            continue
        if any(parent.name in {"p", "li", "pre"} for parent in el.parents):
            continue
        if el.name == "blockquote" and el.find(["p", "li", "pre"]):
            continue
        if el.name == "pre":
            text = el.get_text().strip()
        else:
            for br in el.find_all("br"):
                br.replace_with("\x00")
            text = "\n".join(filter(None, (tidy(line) for line in el.get_text().split("\x00"))))
        if not text:
            continue
        if el.name in headings:
            level = int(el.name[1])
            if END_NOTES.fullmatch(text):
                notes_level = level
            elif book and notes_level and level <= notes_level:
                notes_level = 0
        is_info = any({"information", "footnote"}.intersection(node.get("class", []))
                      for node in (el, *el.parents))
        if notes_level or is_info:
            notes.append(text)
        else:
            lines = text.splitlines()
            notes.extend(line for line in lines if META.match(line))
            body = "\n".join(line for line in lines if not META.match(line))
            if body:
                if not book and el.name in headings and not paragraphs:
                    notes.append(body)
                else:
                    paragraphs.append(body)
    return "\n\n".join(paragraphs), notes


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_catalog(archive: Archive, start: str, output: Path) -> None:
    journal = urljoin(archive.root, "articles.htm")
    works = list(catalog_rows(archive.fetch(start), start, archive.root, False))
    news = list(catalog_rows(archive.fetch(journal), journal, archive.root, True))
    if not works or not news:
        raise ValueError("Expected entries in both Works and Journalism indexes")
    rows = merge_catalog(iter(works + news))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "catalog.json", {"start_url": start, "records": rows})
    print(f"{output / 'catalog.json'}: {dict(Counter(row['decision'] for row in rows))}")


def export_corpus(archive: Archive, rows: list[dict], output: Path, limit: int | None,
                  save_html: bool, accept_proposed: bool) -> int:
    decisions = {"approved", "proposed"} if accept_proposed else {"approved"}
    selected = [row for row in rows if row["decision"] in decisions][:limit]
    if not selected:
        raise ValueError("No works selected. Approve entries in catalog.json first.")
    for row in selected:
        if row["category"] not in CATEGORIES or not row["text_urls"]:
            raise ValueError(f"Select an in-scope category and text URLs for {row['title']!r}")
        for url in row["text_urls"]:
            if not allowed_url(url, archive.root):
                raise ValueError(f"Out-of-scope text URL: {url}")
            if urlsplit(url).path.endswith(("/", "/index.htm", "/index.html", "-intro.htm")):
                raise ValueError(f"Replace the landing page with a text URL: {url}")
    text_dir = output / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output / "raw_html"
    if save_html:
        html_dir.mkdir(exist_ok=True)
    seen: dict[str, str] = {}
    included_in_book: dict[str, str] = {}
    manifest: list[dict] = []
    errors: list[dict] = []
    with (output / "corpus.jsonl").open("w", encoding="utf-8") as corpus:
        # Process books first so a failed book never suppresses an otherwise usable chapter.
        for row in sorted(selected, key=lambda item: item["category"] != BOOKS):
            if row["category"] != BOOKS:
                parent_ids = {included_in_book.get(url) for url in row["text_urls"]}
                if len(parent_ids) == 1 and None not in parent_ids:
                    manifest.append({**row, "status": "included_in_book", "duplicate_of": parent_ids.pop()})
                    continue
            try:
                parts, notes = [], {}
                for url in row["text_urls"]:
                    html = archive.fetch(url)
                    text, page_notes = content_text(html, url, book=row["category"] == BOOKS)
                    if not text:
                        raise ValueError(f"No document body extracted from {url}")
                    parts.append(text)
                    notes[url] = page_notes
                    if save_html:
                        (html_dir / f"{stable_hash(url)[:16]}.html").write_bytes(html)
            except (OSError, ValueError) as exc:
                errors.append({"title": row["title"], "text_urls": row["text_urls"], "error": str(exc)})
                manifest.append({**row, "status": "error"})
                print(f"{row['title']}: {exc}", file=sys.stderr)
                continue
            if row["category"] == BOOKS and len(parts) > 1:
                book_parts = []
                for index, (url, part_text) in enumerate(zip(row["text_urls"], parts), start=1):
                    digest = stable_hash(tidy(part_text))
                    identifier = stable_hash(url)[:16]
                    part_status = "collected"
                    part_reference = {
                        "part_number": index,
                        "source_url": url,
                        "id": identifier,
                        "text_file": f"texts/{identifier}.txt",
                    }
                    if digest in seen:
                        duplicate_id = seen[digest]
                        part_status = "duplicate"
                        part_reference.update(text_file=f"texts/{duplicate_id}.txt",
                                              duplicate_of=duplicate_id)
                        identifier = duplicate_id
                    else:
                        seen[digest] = identifier
                        filename = f"texts/{identifier}.txt"
                        (output / filename).write_text(part_text + "\n", encoding="utf-8")
                        record = {
                            **row, "text_urls": [url], "part_number": index,
                            "part_count": len(parts), "id": identifier, "text_file": filename,
                            "text": part_text, "sha256": digest,
                            "source_metadata": {url: notes[url]},
                        }
                        corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
                    part_reference["status"] = part_status
                    book_parts.append(part_reference)
                    included_in_book[url] = identifier
                manifest.append({**row, "status": "collected", "parts": book_parts})
                continue
            text = "\n\n".join(parts)
            digest = stable_hash(tidy(text))
            identifier = stable_hash("\n".join(row["text_urls"]))[:16]
            if digest in seen:
                manifest.append({**row, "status": "duplicate", "duplicate_of": seen[digest]})
                continue
            seen[digest] = identifier
            filename = f"texts/{identifier}.txt"
            (output / filename).write_text(text + "\n", encoding="utf-8")
            record = {**row, "id": identifier, "text_file": filename,
                      "text": text, "sha256": digest, "source_metadata": notes}
            corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.append({**row, "status": "collected", "id": identifier, "text_file": filename})
            if row["category"] == BOOKS:
                included_in_book.update(dict.fromkeys(row["text_urls"], identifier))
    write_json(output / "manifest.json", manifest)
    write_json(output / "errors.json", errors)
    print(f"Exported {len(seen)} unique texts from {len(selected)} selected works to {output}; "
          f"{len(errors)} errors.")
    return int(bool(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="Build a reviewable catalog without downloading work bodies")
    export = commands.add_parser("export", help="Extract the bodies of selected works")
    for command in (catalog, export):
        command.add_argument("--output", type=Path, default=Path("morris_corpus"))
        command.add_argument("--delay", type=float, default=1.5)
    catalog.add_argument("--start", default=DEFAULT_START, help="Works index URL")
    catalog.add_argument("--overwrite", action="store_true", help="Replace an existing catalog and its decisions")
    export.add_argument("--limit", type=int, help="Maximum selected works to export")
    export.add_argument("--accept-proposed", action="store_true", help="Also export clear proposed matches")
    export.add_argument("--save-html", action="store_true", help="Save source HTML for provenance review")
    args = parser.parse_args()
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be a finite nonnegative number")
    path = args.output / "catalog.json"
    try:
        if args.command == "catalog":
            if path.exists() and not args.overwrite:
                parser.error(f"{path} already exists; use --overwrite to discard its decisions")
            start = args.start
        else:
            if args.limit is not None and args.limit < 1:
                parser.error("--limit must be positive")
            data = json.loads(path.read_text(encoding="utf-8"))
            start = data["start_url"]
        parsed = urlsplit(start)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            parser.error("The Works index must be an HTTP(S) URL without query or fragment")
        root = start if start.endswith("/") else urljoin(start, "./")
        archive = Archive(root, args.delay)
        if args.command == "catalog":
            build_catalog(archive, start, args.output)
            return 0
        return export_corpus(archive, data["records"], args.output, args.limit, args.save_html,
                             args.accept_proposed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Scraper: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from retrieve_data import scrape_final as scraper


class MorrisScraperTests(unittest.TestCase):
    def test_filters_entries_and_merges_provenance(self):
        root = scraper.DEFAULT_START
        works = b"""
        <h2>Works</h2><ul><li><strong>1877</strong><ul>
          <li><strong>Talks and Speeches</strong><ul>
            <li><em>The Lesser Arts</em>, published as
              <a href="1882/hopes/chapters/chapter1.htm">Chapter 1</a> of
              <a href="1882/hopes/index.htm">Hopes and Fears for Art</a></li>
            <li>Lost speech [Lost, but see <a href="1877/other.htm">other work</a>]</li>
          </ul></li>
          <li><strong>Books</strong><ul>
            <li><a href="1882/hopes/index.htm">Hopes and Fears for Art</a></li>
          </ul></li>
          <li><strong>Letters</strong><ul>
            <li><a href="1877/private.htm">Unidentified letter</a></li>
          </ul></li>
          <li><strong>Articles, Pamphlets and Manifestos</strong><ul>
            <li><a href="1877/interview.htm">Interview with Morris</a>
              <a href="index.htm#dedication">*</a></li>
            <li><a href="1877/newspaper.htm">A contribution</a></li>
          </ul></li>
        </ul></li></ul>"""
        journal = b"""
        <h2>Articles for <em>Commonweal</em></h2>
        <ul><li><strong>1877</strong><ul>
          <li><a href="1877/newspaper.htm">A contribution</a></li>
          <li>Serialization of <a href="1877/fiction.htm">News from Nowhere</a></li>
        </ul></li></ul>"""
        rows = scraper.merge_catalog(iter([
            *scraper.catalog_rows(works, root, root, False),
            *scraper.catalog_rows(journal, root + "articles.htm", root, True),
        ]))
        by_title = {row["title"]: row for row in rows}
        self.assertEqual(len(rows), 7)
        self.assertEqual(by_title["The Lesser Arts"]["text_urls"],
                         [root + "1882/hopes/chapters/chapter1.htm"])
        book = by_title["Hopes and Fears for Art"]
        self.assertEqual((book["decision"], book["category"]), ("proposed", scraper.BOOKS))
        self.assertEqual(book["text_urls"], [root + f"1882/hopes/chapters/chapter{i}.htm"
                                             for i in range(1, 6)])
        self.assertEqual((book["book_genre"], book["book_authorship"], book["views_evidence"]),
                         ("nonfiction", "sole_author", "direct"))
        self.assertEqual(by_title["Unidentified letter"]["decision"], "review")
        self.assertEqual(by_title["Interview with Morris"]["category"], scraper.INTERVIEWS)
        self.assertEqual(by_title["A contribution"]["category"], scraper.NEWS)
        self.assertEqual(len(by_title["A contribution"]["sources"]), 2)
        self.assertEqual(by_title["News from Nowhere"]["decision"], "excluded")
        self.assertEqual(next(r for r in rows if r["decision"] == "unavailable")["text_urls"], [])

    def test_export_uses_approved_records_and_preserves_multipart_order(self):
        root = scraper.DEFAULT_START
        urls = [root + "1884/part1.htm", root + "1884/part2.htm"]
        archive = Mock(root=root)
        archive.fetch.side_effect = [
            b'<p class="menu">Works | Biography</p><p>Published: Newspaper<br>Source: Archive</p>'
            b'<p>Here is the <em>first</em>, line.<br>Another line.</p>'
            b'<h2>Bibliographical Note</h2><p>Not part of the work.</p>',
            b"<p>Second part.</p>",
        ]
        approved = {"title": "Work", "category": scraper.NEWS, "decision": "approved", "text_urls": urls}
        pending = {**approved, "decision": "review", "text_urls": [root + "1884/pending.htm"]}
        with TemporaryDirectory() as temp:
            output = Path(temp)
            self.assertEqual(scraper.export_corpus(archive, [pending, approved], output, None,
                                                  False, False), 0)
            record = json.loads((output / "corpus.jsonl").read_text())
            self.assertEqual(record["text"], "Here is the first, line.\nAnother line.\n\nSecond part.")
            self.assertIn("Source: Archive", record["source_metadata"][urls[0]])
            self.assertEqual([call.args[0] for call in archive.fetch.call_args_list], urls)

    def test_book_uses_single_edition_and_keeps_text_after_footnotes(self):
        root = scraper.DEFAULT_START
        works = b"""
        <h2>Works</h2><ul><li><strong>1894</strong><ul><li><strong>Books</strong><ul>
          <li>The Wood Beyond the World (<a href="1894/wdbydwld/index.htm">by chapter</a>
             | <a href="1894/wdbydwld.htm">one big file</a>)</li>
        </ul></li></ul></li></ul>"""
        book, = scraper.catalog_rows(works, root, root, False)
        self.assertEqual(book["title"], "The Wood Beyond the World")
        self.assertEqual(book["text_urls"], [root + "1894/wdbydwld.htm"])
        chapter = {**book, "title": "First chapter", "category": scraper.TALKS}
        archive = Mock(root=root)
        archive.fetch.return_value = (
            b"<h2>Chapter One</h2><p>First text.</p><h3>Footnotes:</h3><p>Note.</p>"
            b"<h2>Chapter Two</h2><p>Last text.</p>"
        )
        with TemporaryDirectory() as temp:
            output = Path(temp)
            self.assertEqual(scraper.export_corpus(archive, [chapter, book], output, None,
                                                  False, True), 0)
            record = json.loads((output / "corpus.jsonl").read_text())
            self.assertEqual(record["text"], "Chapter One\n\nFirst text.\n\nChapter Two\n\nLast text.")
            self.assertEqual((record["book_genre"], record["views_evidence"]), ("fiction", "thematic"))
            self.assertEqual(record["source_metadata"][book["text_urls"][0]], ["Footnotes:", "Note."])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest[1]["status"], "included_in_book")
            self.assertEqual(manifest[1]["duplicate_of"], record["id"])
            archive.fetch.assert_called_once_with(book["text_urls"][0])

    def test_multipart_book_exports_each_chapter_as_a_separate_text(self):
        root = scraper.DEFAULT_START
        urls = [root + "1882/hopes/chapters/chapter1.htm",
                root + "1882/hopes/chapters/chapter2.htm"]
        book = {
            "title": "Hopes and Fears for Art", "year": "1882", "category": scraper.BOOKS,
            "decision": "approved", "text_urls": urls,
        }
        chapter = {
            "title": "The Lesser Arts", "year": "1877", "category": scraper.TALKS,
            "decision": "approved", "text_urls": [urls[0]],
        }
        archive = Mock(root=root)
        archive.fetch.side_effect = [
            b"<h2>Chapter One</h2><p>First chapter.</p>",
            b"<h2>Chapter Two</h2><p>Second chapter.</p>",
        ]
        with TemporaryDirectory() as temp:
            output = Path(temp)
            self.assertEqual(scraper.export_corpus(archive, [chapter, book], output, None,
                                                  False, False), 0)
            records = [json.loads(line) for line in (output / "corpus.jsonl").read_text().splitlines()]
            self.assertEqual([record["text"] for record in records],
                             ["Chapter One\n\nFirst chapter.", "Chapter Two\n\nSecond chapter."])
            self.assertEqual([record["text_urls"] for record in records], [[urls[0]], [urls[1]]])
            self.assertNotEqual(records[0]["text_file"], records[1]["text_file"])

            manifest = json.loads((output / "manifest.json").read_text())
            book_manifest, chapter_manifest = manifest
            self.assertEqual(book_manifest["status"], "collected")
            self.assertEqual([part["source_url"] for part in book_manifest["parts"]], urls)
            self.assertTrue(all((output / part["text_file"]).is_file() for part in book_manifest["parts"]))
            self.assertEqual(chapter_manifest["status"], "included_in_book")
            self.assertEqual(chapter_manifest["duplicate_of"], book_manifest["parts"][0]["id"])
            self.assertEqual([call.args[0] for call in archive.fetch.call_args_list], urls)

    def test_distinguishes_political_fiction_from_joint_and_unknown_views(self):
        root = scraper.DEFAULT_START
        works = b"""
        <h2>Works</h2><ul><li><strong>1890</strong><ul><li><strong>Books</strong><ul>
          <li><a href="1890/nowhere/index.htm">News from Nowhere</a></li>
          <li><a href="1893/sgo/index.htm">Socialism: Its Growth and Outcome</a> (with Belfort Bax)</li>
          <li><a href="1870/volsungs/index.htm">The Saga of the Volsungs</a></li>
          <li><a href="1890/new-title.htm">A new title</a></li>
        </ul></li></ul></li></ul>"""
        fiction, joint, translation, unknown = scraper.catalog_rows(works, root, root, False)
        self.assertEqual((fiction["book_genre"], fiction["views_evidence"], fiction["decision"]),
                         ("fiction", "fictional_exploration", "proposed"))
        self.assertEqual((joint["book_genre"], joint["book_authorship"], joint["views_evidence"]),
                         ("nonfiction", "coauthored", "joint"))
        self.assertEqual((translation["book_authorship"], translation["views_evidence"]),
                         ("co_translation", "not_original"))
        self.assertEqual((unknown["book_genre"], unknown["views_evidence"], unknown["decision"]),
                         ("unknown", "unknown", "review"))

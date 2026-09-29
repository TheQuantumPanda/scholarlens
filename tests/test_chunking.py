import unittest

from scholarlens.chunking import chunk_pages
from scholarlens.models import PageText


class ChunkPagesTests(unittest.TestCase):
    def test_default_chunking_uses_provisional_phase_one_settings(self) -> None:
        words = [f"word{i}" for i in range(1, 281)]
        pages = [
            PageText(
                paper_id="paper-a",
                source_filename="paper-a.pdf",
                page_number=1,
                text=" ".join(words),
            )
        ]

        chunks = chunk_pages(pages)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].text.split()), 250)
        self.assertEqual(chunks[1].text.split()[:30], words[220:250])

    def test_preserves_page_provenance(self) -> None:
        pages = [
            PageText(
                paper_id="paper-a",
                source_filename="paper-a.pdf",
                page_number=1,
                text="one two three four",
            ),
            PageText(
                paper_id="paper-a",
                source_filename="paper-a.pdf",
                page_number=2,
                text="five six",
            ),
        ]

        chunks = chunk_pages(pages, max_words=3, overlap_words=0)

        self.assertEqual([chunk.page_number for chunk in chunks], [1, 1, 2])
        self.assertEqual([chunk.text for chunk in chunks], ["one two three", "four", "five six"])
        self.assertTrue(all(chunk.paper_id == "paper-a" for chunk in chunks))
        self.assertTrue(all(chunk.source_filename == "paper-a.pdf" for chunk in chunks))
        self.assertEqual(chunks[0].chunk_id, "paper-a:chunk-0001:p1-c1")
        self.assertEqual(chunks[1].chunk_id, "paper-a:chunk-0002:p1-c2")
        self.assertEqual(chunks[2].chunk_id, "paper-a:chunk-0003:p2-c1")

    def test_skips_pages_without_extractable_text(self) -> None:
        pages = [
            PageText(
                paper_id="paper-a",
                source_filename="paper-a.pdf",
                page_number=1,
                text="   \n\t  ",
            )
        ]

        self.assertEqual(chunk_pages(pages), [])


if __name__ == "__main__":
    unittest.main()

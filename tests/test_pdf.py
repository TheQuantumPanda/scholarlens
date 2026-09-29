import unittest

import pymupdf

from scholarlens.pdf import extract_pdf_pages


class ExtractPdfPagesTests(unittest.TestCase):
    def test_preserves_page_boundaries_and_empty_pages(self) -> None:
        document = pymupdf.open()
        page_with_text = document.new_page()
        page_with_text.insert_text((72, 72), "First page text")
        document.new_page()
        pdf_bytes = document.tobytes()
        document.close()

        pages = extract_pdf_pages(pdf_bytes, source_filename="sample.pdf", paper_id="sample")

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].paper_id, "sample")
        self.assertEqual(pages[0].source_filename, "sample.pdf")
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("First page text", pages[0].text)
        self.assertEqual(pages[1].paper_id, "sample")
        self.assertEqual(pages[1].source_filename, "sample.pdf")
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[1].text, "")


if __name__ == "__main__":
    unittest.main()

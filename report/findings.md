# Findings

## Initial Inventory
- Files present: `ML_household_power_report.docx`, `ML_household_power_report.pdf`, `report.pdf`, `report.html`, `report_draft.md`, `html_preview_cover.png`, and `report_assets/NotoSansCJKsc-Regular.otf`.
- No `tasks/lessons.md` was present at session start.

## Working Notes
- The directory contains editable source candidates (`report.html`, `report_draft.md`), which are likely better inputs than PDF OCR/extraction.
- `report.pdf` metadata shows Chromium/Skia PDF, 15 A4 pages, created July 6, 2026.
- `report.html` contains the final cover, TOC, sections, two result tables, and eight figure references used in the PDF.
- Existing `ML_household_power_report.docx` contains 11,511 characters, 2 tables, and 8 embedded images, but `styles.xml` does not contain Noto/CJK font settings.
- Available tools: Python 3, Pillow, `pdftotext`, `pdfinfo`, `unzip`; unavailable: `pandoc`, `libreoffice`, `python-docx`, BeautifulSoup/lxml.
- `fc-match 'Noto Sans CJK SC'` resolves to `NotoSansCJK-Regular.ttc`, so the regenerated DOCX can explicitly request a good CJK font.
- Generated DOCX verification: `unzip -t` reports no compressed-data errors; XML parts parse successfully; document has 11370 text characters, 150 paragraphs, 2 tables, 8 drawings, 8 embedded image files, title text, final reference text, and Noto CJK style references.

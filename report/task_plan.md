# Task Plan

## Goal
Regenerate a higher-quality Word document for the report, using editable source text and controlled fonts/layout instead of relying on poor PDF-to-DOCX conversion.

## Phases
- [x] Phase 1: Inspect files, source relationships, previous lessons, and current DOCX defects.
- [x] Phase 2: Select conversion route based on available tools, fonts, and source quality.
- [x] Phase 3: Generate the replacement DOCX without overwriting the existing file.
- [x] Phase 4: Verify text, fonts, images, structure, and basic openability.
- [x] Phase 5: Update planning/result notes.

## Decisions
- Preserve original `ML_household_power_report.docx`; write a new output file first.
- Use `report.html` as the primary source, not PDF OCR, because it matches the Chromium-generated PDF and preserves semantic sections, tables, figure captions, and image paths.
- Generate DOCX via controlled OOXML from Python stdlib/Pillow because `pandoc`, `libreoffice`, and `python-docx` are not available.
- Output path is `ML_household_power_report_rebuilt.docx`; original Word is preserved.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| None yet | - | - |

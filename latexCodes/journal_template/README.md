# Official IEEE journal class for the TWC revision

Retrieved on 2026-10-04 through the IEEE Template Selector:
Transactions, Journals and Letters → IEEE Transactions on Wireless
Communications → Original research and Brief → LaTeX.

- Catalog: https://template-selector.ieee.org/api/ieee-template-selector/template/publication-type/1/publication-title/36
- Download: https://template-selector.ieee.org/api/ieee-template-selector/template/296/download
- Archive: `IEEE-Transactions-LaTeX2e-templates-and-instructions.zip`
- Template example: `bare_jrnl_new_sample4.tex` (journal mode).
- `IEEEtran.cls` archive timestamp: 2024-11-25. The class still identifies as
  V1.8b and includes the official 2023 title-font update.

The class is copied without code changes; line endings are normalized to LF.
Keeping it in this directory leaves the original `../IEEEtran.cls` and earlier
manuscript layouts unchanged. The explicit class path can produce a harmless
LaTeX notice that the class provides `IEEEtran` rather than
`journal_template/IEEEtran`.

Build from `latexCodes`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main_revision1_journal.tex
```

The journal manuscript uses 10-point, two-column journal mode with the default
text area, no geometry override, journal-style author affiliation footnotes,
and the standard `IEEEtran` bibliography style. The latter is supplied by TeX
Live. Deletion markup must remain hidden for the submission-length check.

MEET-COBRA: first-revision main-manuscript LaTeX package

Main file: main.tex
Engine: pdfLaTeX with BibTeX

Build from this directory:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Alternatively:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

CONTENTS
  main.tex       Unmarked journal manuscript (black revision text).
  reference.bib  Bibliography database.
  main.bbl       Pre-generated bibliography, supplied as a fallback.
  IEEEtran.cls   Official IEEE journal class, including the 2023 title update.
  IEEEtran.bst   IEEE bibliography style.
  figures/       The 13 figure files used by the manuscript.

All tables are embedded in main.tex. Standard LaTeX packages are supplied
by a normal TeX Live installation; no project-specific .sty files are used.
Use a TeX Live installation with the packages named in the preamble.

SUBMISSION
Upload Manuscript_LaTeX.zip to Main Manuscript (LaTeX). If the portal asks
for the main source file, select main.tex.

The marked-up manuscript and response to reviewers are separate uploads
and are intentionally not included in this archive.

PROVENANCE
This package is derived from latexCodes/main_revision1_journal.tex.
Deleted passages, disabled draft blocks, and drafting comments are omitted.
The retained formatting helpers render in black. Bibliographic entries,
equations, figure data, and manuscript wording are otherwise unchanged.
Image filenames containing spaces or parentheses have been normalized
inside this package only.

The IEEE class was retrieved through the official IEEE Template Selector:
https://template-selector.ieee.org/api/ieee-template-selector/template/296/download
The standard IEEEtran bibliography style is distributed with TeX Live.

VALIDATION
The archive is compiled after extraction into a separate temporary folder.
The resulting main.pdf must be 16 pages and have the same text and page
layout as ../Manuscript_clean.pdf. No compilation artifacts other than the
pre-generated main.bbl are included in the upload archive.

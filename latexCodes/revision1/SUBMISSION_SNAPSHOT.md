# TWC first-revision submission snapshot

- Submission confirmed by the authors on 2026-10-08.
- Manuscript: **MEET-COBRA: Machine-learning-based Energy-Efficient proacTive Coordination of handOver, Beamforming and Resource Allocation**.
- Manuscript ID: `Paper-TW-Feb-26-0360`.
- Git tag: `twc-r1-submitted-2026-10-08`.

## Source files

- Active manuscript: `latexCodes/main_revision1_journal.tex`.
- Reviewer responses: `response_letter/response_letter.tex`.
- Bibliography: `latexCodes/reference.bib`.
- Submitted LaTeX sources: `latexCodes/revision1/Manuscript_LaTeX.zip`; the editable source directory is also archived, including its bibliography output and figures.
- Earlier manuscript versions remain in Git for reference; `latexCodes/main_revision1.tex` is not the final journal-layout source.

## Submission artifacts

- `PDF_for_Reviewers.pdf`: the 69-page review PDF supplied by the authors after submission. Preserve this file unchanged as the record of what reviewers receive.
- `Manuscript_clean.pdf` and `Manuscript_marked.pdf`: the two manuscript exports.
- `Response_to_reviewers.pdf`: the response letter.
- `Prior_Publication_Statement.pdf` and its `.tex` source.
- `AEPHORA_AI_ML-Based_Energy-Efficient_Proactive_Handover_and_Resource_Allocation.pdf`: the earlier conference publication.
- `Manuscript_LaTeX.zip`: the main-manuscript source archive.

The submitted PDFs are archived as supplied, not regenerated for this snapshot.
To verify their bytes and the principal working sources, run from this directory:

```bash
sha256sum -c SUBMISSION_SHA256SUMS
```

## Code and scope

The tagged commit preserves the current tracked program code, notebooks, manuscript sources, and figure assets, and adds the previously untracked analysis and plotting scripts, tests, and experiment reports outside result/cache directories.

This is a code and submission snapshot, not a complete experiment-data backup. Newly untracked raw simulation outputs, model checkpoints, large datasets, logs, LaTeX temporary files, `_garbage`, and unrelated editor settings are excluded. Experiment results already tracked in earlier commits remain part of the repository history. No files are deleted by this archival step.

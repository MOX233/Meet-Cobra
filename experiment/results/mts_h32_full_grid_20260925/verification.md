# Verification of the MTS H32-cross5 replacement

Date: 2026-09-25.

- All 54 cases completed. The runner independently recalculated power, violation probability, delay-proxy quantiles, association fractions, and pilot counts from the saved arrays. It checked RB capacity bounds, per-slot directional service, 32/5 probe accounting, frozen source and cache hashes, and paired traffic against the original grid.
- Six configuration-identical pilot cases were reused with hashes checked; 48 cases were newly simulated. The reuse manifest identifies every reused result.
- Figure replacement audit passed: 90 MTS rows replaced, all 630 rows for the seven other methods exactly unchanged. See `figure_replacement_audit.json`.
- Five PDF figures and their PNG previews regenerated and visually inspected. Existing axes, legends, colors, and three-seed min/max bands retained. The original submitted figures and earlier result directories were not changed.
- 35 unit tests passed with the command below. The tests cover hierarchical acquisition and local tracking, causal report inputs, bounded occupancy, directional service, O-MAPPO hierarchical search, and isolated replacement of each baseline's curves.
- Both LaTeX documents compiled successfully using `latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error`. The marked manuscript has 18 pages and the response letter 25 pages. A temporary preview with deletion text hidden has 15 pages. This preview is a layout check, not a final clean submission file.
- No undefined references or new overfull-box warnings. Existing manuscript warnings concern the notation table and struck-out equations; with deletions hidden, only the small notation-table warning remains.
- Section V-C is on page 12, Section V-D and Figs. 5–8 on pages 13–14, and the conclusion on page 14 in the hidden-deletion preview. The two updated reviewer replies use these locations.
- Changes are confined to the new MTS experiment, plotting replacement, baseline description, affected results and conclusion, R2C5, R3 Major C3, and revision policy. Existing user edits to the Fig. 4 plotting source and its two PDFs are preserved and excluded from this task's commits.

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python -m unittest test_mts_hierarchical_tracking test_mts_report test_mts_report_bounded test_revision_directional test_o_mappo_hierarchical test_plot_revision_mts_replacement test_plot_revision_o_mappo_replacement
```

Raw arrays (approximately 1.1 GB) and diagnostic logs are retained locally in this result directory; Git stores source code, protocol, summary, provenance, figure data, report, and the revised figures and text. The pre-update manuscript and figures remain recoverable through tag `pre-mts-h32-cross5-paper-20260925`; an additional local copy of the previous figures is in `pre_update_figures/`.

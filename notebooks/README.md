# Historical notebooks

The old interactive notebooks are preserved under `legacy/`, with their saved
outputs unchanged. Current training, experiment and plotting entrypoints are
Python scripts indexed in [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

Notebook code still contains paths relative to the repository root. Before using
a historical notebook, set the kernel working directory to the repository root;
do not assume the notebook's new directory is the data root. Some old notebooks
refer to deleted obsolete datasets and require their regeneration. These are not
the entrypoints for the submitted R1 results.

The old `utils/see_sionna_scene.ipynb` is under `legacy/utils/` to avoid a filename
collision with the root-level notebook. Notebook recovery copies are retained
locally under `archive/recovery/`, not discarded or bulk-added to Git.

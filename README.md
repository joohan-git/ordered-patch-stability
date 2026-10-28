# Ordered patches cap the stability of disordered frameworks

Code and data for J. Nam, *Ordered patches cap the stability of disordered frameworks*.

Archive: <https://doi.org/10.5281/zenodo.23682287>

## Requirements

Python 3.12 with the packages in `requirements.txt`:

```text
python -m pip install -r requirements.txt
```

Two external datasets are not redistributed:

- RCSR 3D nets (`rcsr3d_web.cgd`, <https://rcsr.net>). Set `RCSR_CGD` to the file.
- Meekel et al., *Science* 379, 357 (2023) ([doi:10.1126/science.ade5239](https://doi.org/10.1126/science.ade5239)), Data S5 and S6. Set `MEEKEL_ROOT` to a directory containing `data_Meekel2023_Science/raw/science.ade5239_data_s5/` and `.../science.ade5239_data_s6/`. The extracted cluster connectivities are included, so `MEEKEL_ROOT` is needed only to rebuild them from the CIF files (`common/examples.py trumof-inputs`).

## Quick test

```text
python -B smoke_test.py --cache work/smoke
```

This recomputes the checkerboard threshold, the atlas rows of apo and cco and the TRUMOF-1 *Pc* onset (0.746 GPa), and exits with a nonzero code if a check fails.

## Running calculations

Scripts are run through `run.py`, which works in a separate cache directory and writes results to `<cache>/outputs`:

```text
python -B run.py --cache work/atlas atlas/bootstrap.py --nets apo cco
python -B run.py --cache work/atlas atlas/run_atlas.py --nets apo cco --workers 1
python -B run.py --cache work/beam common/examples.py beam
python -B run.py --cache work/truchet common/examples.py truchet --L 8 --R 4
python -B run.py --cache work/inclusion common/examples.py inclusion --name cco --L 4
python -B run.py --cache work/cowley common/examples.py cowley
python -B run.py --cache work/trumof trumof/mechanics/verify.py --name trumof_1x1x1_7 --kind soft
python -B run.py --cache work/stats truchet_2d/size_statistics.py
```

The `atlas/` scripts and the `atlas-check`, `mesh`, `beam`, `inclusion` and `modes` examples read the RCSR file. The other commands need no external data.

`common/examples.py` provides further small examples: `atlas-check`, `mesh`, `beam`, `truchet`, `rule`, `compact3d`, `h2`, `ising`, `windows`, `cowley`, `inclusion`, `modes`, `trumof-inputs` and `trumof-correlated`. Full runs (the 876-net atlas, the sample cohorts and all TRUMOF-1 cases) take hours; they use the same scripts without the size limits shown above.

## Contents

| Folder (`computations/`) | Paper section |
|---|---|
| `common/` | Beam elements, frame assembly, Bloch and acoustic operators |
| `atlas/` | Crystal-net atlas (Fig. 5, Supplementary Tables 10–12) |
| `truchet_2d/` | Two-dimensional Truchet frames (Figs. 2–4) |
| `frame_3d/` | Three-dimensional frame (Fig. 2) |
| `inclusions/` | Soft inclusions (Fig. 6, Supplementary Table 1) |
| `prestress/` | Prestress transfer (Supplementary Tables 18–19) |
| `selection_rule/` | Selection rule (Fig. 6, Supplementary Tables 16–17) |
| `validation/` | Element and operator checks (Supplementary Tables 13–15) |
| `trumof/` | TRUMOF-1 (Supplementary Section S7) |

JSON files hold the computed results; NPZ files hold starting configurations and continuation states. `SHA256SUMS.json` lists file checksums.

## Citation

See `CITATION.cff`.

## License

MIT (`LICENSE`).

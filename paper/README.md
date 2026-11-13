# CLeaR 2027 submission draft

`main.tex` + `refs.bib` — the single-paper program (CVPR arm retired 2026-11-13).

**Before submission:**
1. Download the official CLeaR/PMLR author kit and swap `\documentclass` (content is sized for ~8 pages + unlimited references; the class file may be `jmlr.cls` with `sample-pmlr.tex`).
2. Fill in real author names / affiliations (currently Anonymous).
3. Resolve the `TODO-CHECK` bib entries (`veitch2021counterfactual`, `zargarbashi2024robust`, `burgess2018disentangling` author lists).
4. Verify page limit against the CLeaR 2027 CfP (the site is JS-only; earlier CLeaR editions were 8 pages + references in PMLR style).

**Every number in the paper** comes from these run logs (reproduce with the listed scripts):
- Table 1 (dose-response): `vault/04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261113_193524.md` — `python scripts/clear_paired.py --config configs/clear_paired_shapes3d_real.yaml --rhos 0.0,0.5,1.0 --models cbm`
- Table 2 (four-image + correction): `vault/04 - Results Logs/Four-Image_2x2_(wall_x_size)_20261113_200807.md` — `python scripts/four_image.py --config configs/clear_paired_shapes3d_real.yaml --rhos 0.0,1.0 --models cbm,residual --n_groups 400`
- First grid / Test-1 attribution / sanity nulls: `vault/04 - Results Logs/CLeaR_Paired_Pilot_clear_paired_shapes3d_real_20261111_094725.md`

Compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

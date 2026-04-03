# Thesis Materials — Navigation Index

## VLM-Guided Diffusion Policies for Legible Robot Motion
**Anudeep Gottapu — Arizona State University, May 2026**

---

## Folder Structure

```
thesis_materials/
├── README.md                    ← You are here
├── thesis_document/
│   └── thesis_main.md           ← Complete 7-chapter thesis (ASU format)
├── presentations/
│   ├── week1_environment_and_demos/
│   │   └── slides.md            ← Environment design & demo collection
│   ├── week2_diffusion_policy/
│   │   └── slides.md            ← U-Net architecture & training
│   ├── week3_vlm_evaluation/
│   │   └── slides.md            ← VLM legibility scoring pipeline
│   ├── week4_guidance_methods/
│   │   └── slides.md            ← 4 guidance methods comparison
│   ├── week5_full_pipeline/
│   │   └── slides.md            ← Integration & honest assessment
│   └── final_defense/
│       └── slides.md            ← Complete 19-slide thesis defense
├── figures/
│   ├── generate_all_figures.py  ← Run to regenerate all figures
│   └── generated/
│       ├── fig1_training_loss.png/.pdf
│       ├── fig2_guidance_sweep.png/.pdf
│       ├── fig3_best_of_n.png/.pdf
│       ├── fig4_method_comparison.png/.pdf
│       ├── fig5_pipeline_stages.png/.pdf
│       ├── fig6_vlo_distribution.png/.pdf
│       ├── fig7_baseline_vlo.png/.pdf
│       └── fig8_architecture.png/.pdf
└── tables/
    └── all_tables.md            ← 11 tables (Markdown + LaTeX template)
```

---

## Quick Access

| Deliverable | File | Status |
|-------------|------|--------|
| Full thesis | `thesis_document/thesis_main.md` | Complete |
| Week 1 slides | `presentations/week1_environment_and_demos/slides.md` | Complete |
| Week 2 slides | `presentations/week2_diffusion_policy/slides.md` | Complete |
| Week 3 slides | `presentations/week3_vlm_evaluation/slides.md` | Complete |
| Week 4 slides | `presentations/week4_guidance_methods/slides.md` | Complete |
| Week 5 slides | `presentations/week5_full_pipeline/slides.md` | Complete |
| Defense slides | `presentations/final_defense/slides.md` | Complete |
| All figures | `figures/generated/` (8 figures, PNG+PDF) | Complete |
| All tables | `tables/all_tables.md` (11 tables) | Complete |

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Base policy success | 84% (42/50) |
| Base policy VLO | 4.57/6 (mostly illegible) |
| VLM demo accuracy | 94.7% (v2: 97.5%) |
| Best guidance method | VLM reranking: L_early = 0.972 |
| Full pipeline improvement | 0.898 → 0.972 (+7.4pp, p=0.00042) |
| Task success maintained | 100% |

---

## To Regenerate Figures

```bash
py -3 thesis_materials/figures/generate_all_figures.py
```

Requires: matplotlib, numpy (standard Python 3.12 install)

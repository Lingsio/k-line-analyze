# SAK-Net: ECCV 2026 Submission

## Paper Information

- **Title**: SAK-Net: Sector-Adaptive K-Line Network for Visual Stock Trend Prediction
- **Venue**: ECCV 2026 (European Conference on Computer Vision)
- **Format**: Springer LNCS, 14 pages + references

## File Structure

```
paper/
├── main.tex              # Main LaTeX document
├── main.bib              # Bibliography file
├── eccv.sty              # ECCV formatting style
├── eccvabbrv.sty         # Abbreviation macros
├── llncs.cls             # Springer LNCS class
├── splncs04.bst          # Bibliography style
└── README.md             # This file
```

## Paper Structure

1. **Abstract** - Summary of contributions and results (57.81% accuracy)
2. **Introduction** - Motivation, challenges, and contributions
3. **Related Work** - Time series encoding, financial image analysis, domain adaptation
4. **Methodology** - Problem definition, encoding methods, architecture, sector-adaptive training
5. **Experiments** - Dataset, metrics, encoding comparison, baseline comparison, sector results, ablation studies
6. **Discussion** - Why visual methods work, sector adaptation insights, limitations
7. **Conclusion** - Summary of findings and future directions

## Compilation

### Using pdflatex + bibtex:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk:
```bash
latexmk -pdf main.tex
```

## Key Results

| Method | Accuracy | vs. Baseline |
|--------|----------|--------------|
| Random | 50.00% | - |
| LSTM | 49.68% | -0.32% |
| Candlestick RGB | 50.83% | +0.83% |
| Universal CNN | 51.26% | +1.26% |
| **SAK-Net (Ours)** | **57.81%** | **+7.81%** |

## Sector-Adaptive Results

| Sector | Accuracy | Improvement |
|--------|----------|-------------|
| Consumer | 60.37% | +9.11% |
| Industrials-Energy | 59.94% | +8.68% |
| Tech-Semiconductors | 59.88% | +8.62% |
| Tech-Software | 57.22% | +5.96% |
| Financials | 55.90% | +4.64% |
| Healthcare | 53.54% | +2.28% |

## Notes for Submission

1. **Blind Review**: Author information is anonymized for review
2. **Page Limit**: 14 pages (excluding references)
3. **Line Numbers**: Enabled for review version
4. **Paper ID**: Replace `*****` in `\usepackage[review,year=2026,ID=*****]{eccv}` with actual submission ID

## TODO Before Submission

- [ ] Update paper ID in main.tex
- [ ] Add author information for camera-ready version
- [ ] Add institutional affiliations
- [ ] Include ORCID links
- [ ] Add acknowledgments
- [ ] Generate high-quality figures
- [ ] Verify all citations compile correctly
- [ ] Check page count (max 14 pages)

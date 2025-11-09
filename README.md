# GeoECGAN

A reference implementation scaffold for **GAN-based geometry + NPI fusion** as in your figure:

- **Upper branch (NPI)**: virtual-perturbation effective connectivity (EC) from **HBN-EEG** windows.
- **Lower branch (Geometric Eigenmodes)**: geometry-constrained latent patterns from **HCP**.
- **Validation**: **ChineseEEG** character-level **CER(%)** and **BLEU**.

> ⚠️ This repo is a **clean-room scaffold**. Plug in the official GitHub code of the three papers (see `docs/REFERENCES.md`) where indicated, or wrap their modules as subpackages.

## Quickstart
```bash
pip install -e .
python train.py --device cpu
python -c "from evaluate import evaluate_chinese; print(evaluate_chinese(['我爱大脑研究'], ['我爱脑研究']))"
```

## Repo Layout
```
GeoECGAN/
├── src/geoecgan/                # python package
│   ├── models/                  # NPI, GEM, GAN
│   ├── metrics/                 # CER & BLEU
│   └── data/                    # dataset stubs
├── configs/                     # example YAMLs (TODO)
├── scripts/                     # shell helpers
├── tests/                       # unit tests (TODO)
├── docs/                        # method & dataset notes
├── .github/workflows/ci.yml     # lint + test CI
├── pyproject.toml               # packaging
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
└── CHANGELOG.md
```

## Datasets
- **HBN-EEG** → NPI branch input windows (shape `(B, 3, R)`)  
- **HCP** → geometry (mesh `(V,F)` *or* adjacency `A`) for eigenmodes  
- **ChineseEEG** → text decoding evaluation with CER / BLEU

See `docs/DATASETS.md` for download and preprocessing placeholders.

## Mapping to Papers
- **NPI branch** → integrate the official NPI code where `geoecgan/models/npi.py` is stubbed.  
- **GEM branch** → replace `gem.py` with the official geometry operator if available.  
- **ChineseEEG eval** → align tokenizer/normalizer with the paper for comparable CER/BLEU.

---
© 2025 MIT License.

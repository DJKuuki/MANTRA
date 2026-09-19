# Canonical FOMC Research Dataset (Trillion Dollar Words)

## 1. Overview
This dataset is adapted from the official research release of:
> **Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis**  
> Agam Shah, Suvan Paturi, Sudheer Chava  
> *ACL 2023 (Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics)*  
> DOI: [10.18653/v1/2023.acl-long.368](https://doi.org/10.18653/v1/2023.acl-long.368)  
> Repository: `gtfintechlab/fomc-hawkish-dovish`  
> License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

## 2. Manifest & Verification
- **Total Validated Samples**: 2281
- **Checksum**: `sha256:344f6cda7f59a6fcc2b088fd188dd03cc6dc53a8c25b862ce529e3e1218b073d`
- **Source Verified**: `True` (Federal Reserve communications verified by Shah et al.)
- **Annotation Verified**: `True` (Manually labeled by financial economics domain experts)
- **PIT Verified**: `False` (**Explicitly False**: Sentence-level corpus includes `year` metadata but lacks second-level publication timestamps)

## 3. Label Breakdown
- **Hawkish (+1)**: 571
- **Neutral (0)**: 1112
- **Dovish (-1)**: 598

## 4. Document Type Distribution
- **Minutes**: 1010
- **Press Conferences**: 309
- **Speeches**: 962

## 5. Temporal Splits
- **Train Split ($\le 2018$)**: 1729 samples
- **Dev Split ($2019$)**: 114 samples
- **Test Split ($\ge 2020$)**: 438 samples

## 6. Point-in-Time Safety Protocol Notice
Because publication intraday timestamps are absent in the upstream academic release, all samples are marked `availability_quality: unknown`.
Under `configs/fomc_formal_experiment.yaml`, these samples are rejected to uphold zero-tolerance exact Point-in-Time standards. They are utilized for real encoder pretraining corpus construction, representation baseline extraction, and CI/development smoke tests (`configs/fomc_ci.yaml`).

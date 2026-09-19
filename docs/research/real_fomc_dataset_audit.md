# Audit Report: Real FOMC Dataset Ingestion & Point-in-Time Verification

**Dataset Name**: Trillion Dollar Words (Shah et al., ACL 2023)  
**Upstream Repository**: [`gtfintechlab/fomc-hawkish-dovish`](https://github.com/gtfintechlab/fomc-hawkish-dovish)  
**Academic Venue**: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023), pp. 6664–6679  
**Canonical Identifier**: DOI [10.18653/v1/2023.acl-long.368](https://doi.org/10.18653/v1/2023.acl-long.368) / arXiv:2305.07972  
**Audit Date**: September 2026  
**Auditor**: MANTRA Research Engineering Team  

---

## 1. Executive Summary & Verification Matrix

In accordance with Section 4 of the research protocol, external datasets cannot be declared valid or point-in-time verified merely by file inclusion. Each dimension must be audited and verified independently:

| Verification Dimension | Status | Verification Evidence & Rationale |
| :--- | :--- | :--- |
| **Source Verification (`source_verified`)** | **VERIFIED (True)** | Text corpus extracted directly from official Federal Reserve releases: FOMC meeting minutes, FOMC press conference transcripts, and Federal Reserve Governor/President speeches spanning 1996 to 2022. Cross-referenced against official Federal Reserve website records by Shah et al. (ACL 2023). |
| **Annotation Verification (`annotation_verified`)** | **VERIFIED (True)** | Stance annotations (Hawkish / Neutral / Dovish) were manually created by financial domain experts and economics researchers using strict linguistic monetary policy definitions. Annotations represent human textual perception, fully decoupled from forward market returns ($Y_{\text{task}} \neq Y_{\text{econ}}$). |
| **Point-in-Time Verification (`pit_verified`)** | **UNVERIFIED (False)** | **Strict Audit Finding**: The upstream academic release provides `year` metadata for each annotated sentence, but does NOT release second-level or minute-level publication timestamps for individual sentences. In accordance with MANTRA PIT protocols (Section 5: "No Silent Exact"), all samples are marked `availability_quality: unknown`. Consequently, `is_formal_research_ready()` evaluates to `False` under zero-tolerance exact configurations (`configs/fomc_formal_experiment.yaml`), while being safely permitted for continued pretraining and baseline smoke runs (`configs/fomc_ci.yaml`). |

---

## 2. Corpus Statistics & Distributions

### 2.1 Total Volume & Time Horizon
- **Total Validated Samples**: 2,281 sentences (with valid labels $\in \{-1, 0, +1\}$; 199 unlabelled/discarded sentences from the raw 2,480 rows were strictly excluded).
- **Calendar Coverage**: 1996 to 2022 (27 calendar years).
- **Mean Annual Sample Density**: $\sim 84.5$ sentences per year.

### 2.2 Document Type Breakdown
The corpus samples represent three distinct communication channels of the Federal Reserve:

| Document Type | Sample Count | Percentage of Corpus | Publication Cadence |
| :--- | :--- | :--- | :--- |
| **Meeting Minutes (`minutes`)** | 1,010 | 44.28% | Historically released with statutory latency (typically 3 weeks post-meeting). |
| **Speeches & Testimony (`speech`)** | 962 | 42.17% | Delivered at diverse public and institutional speaking events by Federal Reserve officials. |
| **Press Conference Transcripts (`press_conference`)** | 309 | 13.55% | Initiated in 2011; delivered by Fed Chair (Bernanke, Yellen, Powell) post-meeting. |
| **Total** | **2,281** | **100.0%** | |

### 2.3 Stance Label Distribution ($Y_{\text{task}}$)
Labels are mapped to the standardized MANTRA 3-class scale $\{-1, 0, +1\}$:
- **Label +1 (Hawkish)**: 571 samples (25.03%) — emphasis on inflation containment, rate increases, policy tightening, and economic overheating.
- **Label 0 (Neutral)**: 1,112 samples (48.75%) — balanced factual reporting, description of current baseline conditions, and mandate restatements.
- **Label -1 (Dovish)**: 598 samples (26.22%) — emphasis on labor market slack, growth headwinds, accommodation preservation, and rate reductions.

The class balance reveals substantial neutral weight (48.75%), emphasizing why **Macro-averaged $F_1$** and **MCC** are the mandatory Task Competence metrics rather than raw classification accuracy.

---

## 3. Point-in-Time (PIT) Safety Audit

### 3.1 Quality Breakdown
| Quality Tier | Sample Count | Percent | Note |
| :--- | :--- | :--- | :--- |
| **Exact** | 0 | 0.0% | No sentence-level publication timestamps provided in upstream release. |
| **Heuristic** | 0 | 0.0% | Not assigned without documented daily release schedule. |
| **Unknown** | 2,281 | 100.0% | Conservative assignment per MANTRA Preflight rule. |

### 3.2 Audit on Release Timestamps & the "14:00 Assumption" Fallacy
A critical design requirement of MANTRA is:
> **Do not assume all FOMC documents are published at 14:00 America/New_York.**

- FOMC Statements are typically released at 14:00 EST/EDT.
- Press Conferences begin at 14:30 EST/EDT, with unedited transcripts published hours later or the following business morning.
- Meeting Minutes are published at 14:00 exactly 21 calendar days after the meeting ($T+21$).
- Speeches occur at irregular times throughout the day, and cannot be timestamped at 14:00 without look-ahead or look-behind errors.

Because the Trillion Dollar Words sentence files contain only `year` attributes, hardcoding `14:00:00-05:00` would constitute look-ahead fabrication. Setting `availability_quality: unknown` transparently records this uncertainty and prevents Look-Ahead Bias.

---

## 4. Partition Splits & Label Breakdown

Partitions are defined according to the frozen formal boundaries in [`configs/fomc_formal_experiment.yaml`](../../configs/fomc_formal_experiment.yaml):
- **Train Split ($\mathcal{D}_{\text{train}}$)**: $t \le \text{2018-12-31T23:59:59Z}$
- **Dev Split ($\mathcal{D}_{\text{dev}}$)**: $\text{2019-01-01T00:00:00Z} \le t \le \text{2019-12-31T23:59:59Z}$
- **Test Split ($\mathcal{D}_{\text{test}}$)**: $t \ge \text{2020-01-01T00:00:00Z}$

### 4.1 Sample Counts & Label Distribution by Partition
| Partition Split | Total Samples | Hawkish (+1) | Neutral (0) | Dovish (-1) | Regime Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Split ($\le 2018$)** | 1,729 (75.80%) | 434 (25.10%) | 825 (47.72%) | 470 (27.18%) | Multi-regime era: Dot-com bubble, GFC, ZLB, and 2015–2018 liftoff. |
| **Dev Split ($2019$)** | 114 (5.00%) | 31 (27.19%) | 50 (43.86%) | 33 (28.95%) | 2019 "mid-cycle adjustment" easing cycle. |
| **Test Split ($\ge 2020$)** | 438 (19.20%) | 106 (24.20%) | 237 (54.11%) | 95 (21.69%) | COVID-19 shock, 2021 transitory debate, 2022 rapid rate hike cycle. |
| **Total** | **2,281** | **571** | **1,112** | **598** | |

### 4.2 Split Integrity Analysis
1. The class proportions across Train (25.1% H / 47.7% N / 27.2% D), Dev (27.2% H / 43.9% N / 29.0% D), and Test (24.2% H / 54.1% N / 21.7% D) exhibit remarkable temporal stability, while also reflecting the heightened neutral and reactive tone during the 2020–2022 pandemic era.
2. The pre-cutoff partition contains 1,729 sentences, providing sufficient volume for downstream stance classifier calibration and pre-cutoff continued pretraining.
3. The post-cutoff test partition contains 438 sentences, providing an out-of-sample evaluation set for evaluating representational leakage ($L_{\text{repr}}$), behavioral masking sensitivity ($L_{\text{behavior}}$), and economic effect ($E_L^{\text{IC}}$).

---

## 5. Artifact Provenance & File Hashes

- **Canonical JSONL**: `data/research/fomc/fomc_temporal_dataset.jsonl`
- **SHA-256 Checksum**: `344f6cda7f59a6fcc2b088fd188dd03cc6dc53a8c25b862ce529e3e1218b073d`
- **Manifest**: [`data/research/fomc/manifest.json`](../../data/research/fomc/manifest.json)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- **Downloader & Converter Script**: [`scripts/download_trillion_dollar_words.py`](../../scripts/download_trillion_dollar_words.py)

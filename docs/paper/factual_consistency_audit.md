# Phase 6 Factual Consistency Audit Ledger

**Repository**: `DJKuuki/MANTRA`  
**Stage**: Phase 6 — Paper Factual Consistency & Reference Verification Patch  
**Scientific Code Freeze**: `5ec0f03f3a5393d90462aa78d018d54b08cce126`  
**Phase 4B Archive Closure**: `5d82a7a321f6ce2441e449b5fff7ec482681819e`  
**Locked Source Tree**: `02fe0ced0d03a920a7f56887f1d674282d17bc108986d19d2df495b59d330bb6`  
**Protocol Specification**: Version `1.2.4`  

---

## 1. Experimental and Execution Facts Audit

This audit ledger systematically contrasts claims in the initial Phase 6 full-paper assembly against frozen ground truth in `configs/phase4_confirmatory.yaml`, `configs/phase4_preregistration.yaml`, `experiments/phase4_confirmatory/manifests/*.json`, and production runtime code under `tradingagents/temporal_leakage/`.

| Paper Claim / Parameter | Frozen Source of Truth | Previous Paper Value | Corrected Value | Status |
| :--- | :--- | :--- | :--- | :---: |
| **MLM Token Budget** | `configs/phase4_confirmatory.yaml:45` (`token_budget: 256000`) | 256,000 tokens | 256,000 tokens constructed treatment stream | **VERIFIED** |
| **MLM Block Count** | `configs/phase4_confirmatory.yaml:46` (`num_blocks: 500`) | Omitted / conflated | 500 packed blocks per branch | **VERIFIED** |
| **MLM Block Length** | `configs/phase4_confirmatory.yaml:47` (`block_length: 512`) | "sequence length 160" | 512 tokens per packed block | **VERIFIED** |
| **MLM Batch Size** | `configs/phase4_confirmatory.yaml:48` (`batch_size: 16`) | Batch size 16 | Batch size 16 | **VERIFIED** |
| **MLM Optimizer Steps** | `configs/phase4_confirmatory.yaml:54` (`max_steps: 100`) | 100 steps ($16 \times 160 \times 100$) | 100 optimizer gradient steps | **VERIFIED** |
| **MLM Learning Rate** | `configs/phase4_confirmatory.yaml:49` (`learning_rate: 5.0e-5`) | $5 \times 10^{-5}$ | $5 \times 10^{-5}$ (AdamW) | **VERIFIED** |
| **MLM Weight Decay** | `configs/phase4_confirmatory.yaml:50` (`weight_decay: 0.01`) | 0.01 | 0.01 | **VERIFIED** |
| **MLM Masking Ratio** | `configs/phase4_confirmatory.yaml:51` (`mask_ratio: 0.15`) | 15% | 15% random dynamic masking | **VERIFIED** |
| **MLM Scheduler** | `configs/phase4_confirmatory.yaml:55` (`scheduler: "none"`) | "linear decay" | `none` (constant learning rate) | **VERIFIED** |
| **MLM Warmup Ratio** | `configs/phase4_confirmatory.yaml:56` (`warmup_ratio: 0.0`) | "linear warmup 10 steps" | `0.0` (0 warmup steps) | **VERIFIED** |
| **Downstream Training Scope** | `twin_pipeline.py:758,789` (`model.train()`, `model.parameters()`) | "Frozen encoder + trained head" | Full model sequence-classification fine-tuning | **VERIFIED** |
| **Downstream Epochs** | `configs/phase4_confirmatory.yaml:78` (`epochs: 3`) | 5 epochs | 3 epochs | **VERIFIED** |
| **Downstream Batch Size** | `configs/phase4_confirmatory.yaml:79` (`batch_size: 16`) | Batch size 16 | Batch size 16 | **VERIFIED** |
| **Downstream Learning Rate**| `configs/phase4_confirmatory.yaml:80` (`learning_rate: 2.0e-5`) | Unspecified / conflated | $2.0 \times 10^{-5}$ | **VERIFIED** |
| **Downstream Weight Decay** | `configs/phase4_confirmatory.yaml:81` (`weight_decay: 0.01`) | Unspecified | 0.01 | **VERIFIED** |
| **Downstream Optimizer** | `configs/phase4_confirmatory.yaml:82` (`optimizer: "AdamW"`) | AdamW | AdamW | **VERIFIED** |
| **Downstream Scheduler** | `configs/phase4_confirmatory.yaml:83` (`scheduler: "linear"`) | Unspecified | `linear` | **VERIFIED** |
| **Downstream Warmup** | `configs/phase4_confirmatory.yaml:84` (`warmup_ratio: 0.1`) | Unspecified | `0.1` (10% linear warmup) | **VERIFIED** |
| **Downstream Max Seq Len** | `configs/phase4_confirmatory.yaml:85` (`max_seq_length: 128`) | 160 tokens | 128 tokens | **VERIFIED** |
| **Downstream Realized Steps**| 1,729 TDW pre-2019 samples / batch 16 = 108 steps $\times$ 3 | Unreported | 3 epochs, yielding 324 realized optimizer steps | **VERIFIED** |
| **Downstream Head Init** | `configs/phase4_confirmatory.yaml:87` (`fresh_shared_within_seed`) | Matched initialization | Fresh 3-class linear head shared within seed | **VERIFIED** |
| **Downstream Sample Order** | `configs/phase4_confirmatory.yaml:88` (`paired_within_seed`) | Paired order | Identical batch ordering within seed (`DataLoader` seed) | **VERIFIED** |
| **Representation Probe** | `configs/phase4_confirmatory.yaml:62,63` (`ridge_regression`, `alpha: 1.0`) | Linear Ridge ($\alpha=1.0$) on representations | Linear Ridge ($\alpha=1.0$) on frozen extracted representations | **VERIFIED** |
| **Permutation Draws ($B$)** | `configs/phase4_confirmatory.yaml:69` (`paired_permutations: 2000`) | 2,000 draws | 2,000 draws (one-sided right-tailed paired sign-flip) | **VERIFIED** |
| **Finite-MC Resolution** | $k=0, B=2000 \implies p_{\mathrm{plus\_one}} = \frac{k+1}{B+1}$ | $p < 1/2001$ | $p_{\mathrm{plus\_one}} = 1/2001 \approx 0.00050$ (raw $p = 0.0000$) | **VERIFIED** |
| **Economic Bootstrap Draws** | `configs/phase4_confirmatory.yaml:70` (`bootstrap_draws: 1000`) | "clustered bootstrap by event" | 1,000-draw event-level stationary block bootstrap | **VERIFIED** |
| **Out-of-Sample Events** | `configs/phase4_confirmatory.yaml:68` (`oos_events: 32`) | 32 OOS events (2016–2019) | 32 out-of-sample temporal-CV meetings (Folds 2–4) | **VERIFIED** |
| **Total Evaluation Events** | `configs/phase4_confirmatory.yaml:15` (`expected_events: 40`) | 40 events | 40 scheduled FOMC calendar meetings (2015–2019) | **VERIFIED** |
| **Evaluation Anchors** | `configs/phase4_confirmatory.yaml:16` (`expected_anchors: 181`) | 181 anchors | 181 standardized paragraph anchors | **VERIFIED** |
| **Temporal CV Folds** | `configs/phase4_confirmatory.yaml:66` (`cv_folds: 4`) | 4 folds | 4-fold grouped expanding-window temporal CV | **VERIFIED** |
| **Cutoff Timestamp** | `configs/phase4_confirmatory.yaml:17` (`2019-12-31T23:59:59Z`) | `2019-12-31T23:59:59Z` | `2019-12-31T23:59:59Z` | **VERIFIED** |
| **Clean Sham Corpus** | `configs/phase4_confirmatory.yaml:23-25` (63 docs, 2015-01-28 to 2019-12-11) | 63 documents | 63 documents spanning 2015-01-28 to 2019-12-11 | **VERIFIED** |
| **Contamination Corpus** | `configs/phase4_confirmatory.yaml:31-33` (50 docs, 2020-01-29 to 2023-01-04) | 50 documents | 50 documents spanning 2020-01-29 to 2023-01-04 (48-day buffer) | **VERIFIED** |
| **Base Model Checkpoint** | `configs/phase4_confirmatory.yaml:39,40` (`ProsusAI/finbert`, revision `4556d13...`) | `ProsusAI/finbert` | `ProsusAI/finbert` (revision `4556d13015211d73dccd3fdd39d39232506f3e43`) | **VERIFIED** |
| **Total Corpus Volume** | 25 branches $\times$ 256,000 tokens = 6,400,000 tokens | "6.4M tokens processed by optimizer" | Aggregate constructed treatment-stream budget of 6.4M tokens | **VERIFIED** |
| **Hardware & Determinism** | Dedicated CUDA GPU; PyTorch | "strict hardware determinism" | Paired execution symmetry and reproducibility controls | **VERIFIED** |
| **Causal Attribution Language**| Controlled twin comparison vs regime shifts | "sole experimental variable" | Isolates incremental effect of temporal composition under matched compute | **VERIFIED** |
| **Downstream Decoupling Wording**| Statistical non-significance / null support | "attenuates/decouples completely" | Downstream endpoints did not support preregistered positive alternative | **VERIFIED** |

---

## 2. Archival Integrity Sign-Off
All entries in this ledger have been validated against frozen configurations, manifests, and production code. No empirical data, code, or preregistration parameters were modified.

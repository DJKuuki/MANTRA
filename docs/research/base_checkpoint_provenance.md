# Base Checkpoint Provenance: ProsusAI/finbert
 
**Status**: `bounded / uncertain` (NOT `verified`)  
**Evaluation Phase**: Phase 2.1 — Causal Twin Activation & Baseline Correction  
**Target Checkpoint**: `ProsusAI/finbert`  
**Pinned Revision**: `4556d13015211d73dccd3fdd39d39232506f3e43`  
 
---
 
## 1. Provenance Inventory
 
| Dimension | Audit Record | Status |
| :--- | :--- | :--- |
| **Model Repository** | `https://huggingface.co/ProsusAI/finbert` | Tracked |
| **Pinned Revision SHA** | `4556d13015211d73dccd3fdd39d39232506f3e43` | Pinned & Immutable |
| **Underlying Architecture** | `BertForSequenceClassification` (12 layers, 768 hidden dim, 12 attention heads, $\approx 110\text{M}$ parameters) | Standard BERT-base |
| **Tokenizer** | `BertTokenizerFast` (vocab size: 30,522, WordPiece) | Standard BERT-base |
| **Base Pretraining Corpus** | BooksCorpus (800M words) + English Wikipedia (2,500M words) by Devlin et al. (2018) | Pre-2018 general text |
| **Domain Adaptation Corpus** | Reuters TRC2-financial news corpus ($\approx 800\text{k}$ documents) by Dogu Araci (2019) | Financial domain news |
| **Task Fine-Tuning Corpus** | Financial PhraseBank (Malo et al., 2014) | Financial sentiment |
| **Original Task Output Classes** | 3 classes: `0: "positive"`, `1: "negative"`, `2: "neutral"` | Financial sentiment |
| **Latest Known Training Date** | Mid-2019 (release of FinBERT paper / repository) | Coarse temporal upper bound |
| **Cryptographic Cutoff Provenance** | Absent. No hash-verified dataset manifest or training log artifact exists from original authors. | **Unprovable** |
| **Temporal Cutoff Designation** | **`training_cutoff_status = bounded / uncertain`** | **Strict Audit Standard** |
 
---
 
## 2. Identified Uncertainties & Failure Modes
 
1. **Unverifiable Pretraining Boundary**: While the FinBERT publication occurred in 2019, third-party consumers cannot verify whether any web crawl updates or post-2018 articles entered the Reuters TRC2 adaptation corpus without raw corpus manifests. Therefore, claiming `training_cutoff_status = verified` for 2018-12-31 is scientifically unsupportable. It must remain `bounded / uncertain`.
2. **Task Semantics Mismatch**: The original FinBERT head was trained exclusively on sentence-level financial market sentiment (`positive`, `negative`, `neutral` for equity investors). This is semantically distinct from central bank monetary policy stance (`Dovish`, `Neutral`, `Hawkish`). Reinterpreting raw logits as monetary stance is invalid.
3. **Tokenizer Vocabulary Bias**: The WordPiece tokenizer was frozen during Google's 2018 BERT training. Monetary policy terms emerging after 2018 (e.g. specialized pandemic facilities) are sub-tokenized.
 
---
 
## 3. Methodological Protocol in MANTRA
 
To preserve causal validity despite uncertain upstream provenance:
1. **Head Discard**: The raw financial sentiment classification head of `ProsusAI/finbert` is discarded upon initialization.
2. **Body Re-use as Initialization Only**: The BERT encoder body is retained strictly as initial weights $\theta_0$ for:
   - Continued MLM pretraining under symmetric conditions ($M_C$ vs $M_L$).
   - Stance classifier construction with a freshly initialized, bit-identical FOMC 3-class classification head (`id2label={0: "Dovish", 1: "Neutral", 2: "Hawkish"}`).
3. **Causal Twin Symmetry**: Because both Clean ($M_C$) and Leak ($M_L$) twins start from the identical initial parameter state $\theta_0$, any upstream temporal contamination in $\theta_0$ affects both twins identically. The differential treatment:
   $$\Delta_L = M_L - M_C$$
   isolates strictly the post-cutoff temporal exposure administered in the MANTRA pipeline.

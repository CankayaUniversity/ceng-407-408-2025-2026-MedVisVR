# Clinical Implication Thresholds

> Single reference document for all threshold values used in `clinical_implications.py`.  
> **All thresholds are literature-anchored surrogates.** They support decision-support language only — not autonomous clinical diagnosis or treatment recommendation.

---

## 1. Midline Shift (MLS)

**Source metric:** `mass_effect.midline_shift_mm`  
**Implication type:** `herniation_risk`

| Threshold | Tier | Rationale |
|---|---|---|
| MLS < 5 mm | `none` | Below the >5 mm threshold for significant mass effect (Marmarou 1987) |
| 5 mm ≤ MLS < 7.5 mm | `warning` | At/above the commonly used >5 mm significant-shift threshold; clinically relevant mass effect warrants caution |
| MLS ≥ 7.5 mm | `critical` | At/above the ~7–7.5 mm range linked to effaced basal cisterns and abnormal pupillary findings (Ropper 1986) |

**Literature anchors:**
- Ropper AH (1986). *Lateral displacement of the brain and level of consciousness in patients with acute hemispheral mass.* NEJM 314:953–958.
- Marmarou A et al. (1987). *Contribution of CSF and vascular factors to elevation of ICP in severely head-injured patients.* J Neurosurg 66:883–890.
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6208863/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6651068/

---

## 2. Motor Cortex Distance

**Source metric:** `critical_proximity.motor_cortex_distance_mm`  
**Implication type:** `surgical_eloquence_risk`

| Threshold | Tier | Rationale |
|---|---|---|
| > 8 mm | `safe` | Above the ~8 mm separation used in motor-eloquence planning as lower-risk zone for post-op motor deficit |
| 5 mm < distance ≤ 8 mm | `caution` | Intermediate motor-eloquence corridor; resection warrants caution and functional mapping support |
| ≤ 5 mm | `high_risk` | Sub-5 mm eloquent interval; surgically high-risk; strong mapping caution required (awake craniotomy typically indicated) |

**Literature anchors:**
- Duffau H et al. (2003). *New insights into the anatomo-functional connectivity of the semantic system.* Brain.
- Chang EF et al. (2008). *Functional mapping-guided resection of low-grade gliomas in eloquent areas.* J Neurosurg.
- PubMed: https://pubmed.ncbi.nlm.nih.gov/20679917/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12954220/

---

## 3. Brainstem Distance

**Source metric:** `critical_proximity.brainstem_distance_mm`  
**Implication type:** `brainstem_compression_risk`

| Threshold | Tier | Rationale |
|---|---|---|
| > 10 mm | `safe` | >10 mm from brainstem; lower immediate concern for direct compression |
| 5 mm < distance ≤ 10 mm | `caution` | Within 5–10 mm of brainstem; caution for evolving mass effect or crowding |
| ≤ 5 mm | `high_risk` | Within 5 mm of brainstem; high-risk proximity tier for compression-related concern |

**Literature anchors:**
- Same as MLS references (brainstem proximity literature overlaps with herniation literature)
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6208863/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6651068/

---

## 4. Enhancing Tumor Volume

**Source metric:** `label_volumes_mm3.enhancing` (Label 4, ET)  
**Implication type:** `resection_consideration_tier`

| Threshold | Tier | Rationale |
|---|---|---|
| < 15 cm³ | `limited` | Below the larger-volume ranges discussed in GBM cytoreduction literature |
| 15–30 cm³ | `notable` | Moderate range where cytoreductive benefit becomes clinically meaningful if anatomy is favorable |
| > 30 cm³ | `substantial` | Substantial preoperative volume; strengthens resection-consideration language if safely accessible |

**Literature anchors:**
- Ellingson BM et al. (2016). *Volumetric response quantification using T1 subtraction predicts long-term survival benefit from bevacizumab in recurrent GBM.* Neuro-oncology.
- PubMed: https://pmc.ncbi.nlm.nih.gov/articles/PMC3264493/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/38337543/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/41265784/

---

## 5. Total Tumor Volume (Whole Tumor)

**Source metric:** `total_tumor_volume_mm3` (Labels 1+2+4)  
**Implication type:** `tumor_burden_tier`

| Threshold | Tier | Rationale |
|---|---|---|
| < 25 cm³ | `low` | Relatively limited volumetric burden |
| 25–75 cm³ | `moderate` | Moderate burden; may contribute to meaningful local mass effect depending on location |
| > 75 cm³ | `high_risk` | High burden; strengthens concern for clinically important intracranial mass effect |

**Literature anchors:**
- Same as enhancing volume references (volumetric burden cohort studies)

---

## 6. Basal Cisterns

**Source metric:** `mass_effect.cisternal_obliteration`, `mass_effect.cisternal_effacement`  
**Implication type:** `icp_surrogate_tier`

| Finding | Tier | Rationale |
|---|---|---|
| No effacement or obliteration | `none` | No cisternal compression recorded |
| Cisternal effacement present | `warning` | Warning-level surrogate for raised ICP |
| Cisternal obliteration present | `critical` | High-risk surrogate for raised ICP and transtentorial crowding |

**Literature anchors:**
- Same as MLS references (herniation/ICP literature)

---

## 7. Tier Hierarchy

```
none / safe / limited / low   →  rank 0  (informational)
warning / caution / moderate / notable  →  rank 1  (caution)
critical / high_risk / substantial      →  rank 2  (action-required)
```

`overall_tier` in `ClinicalImplication` is the **maximum rank** tier across all items.

---

## 8. BT-RADS Alignment

| Clinical tier | BT-RADS alignment note |
|---|---|
| `critical`, `high_risk`, `substantial` | Suggests BT-RADS 3B–4 features; multidisciplinary review warranted |
| `warning`, `caution`, `notable`, `moderate` | Consistent with BT-RADS 2–3A; short-interval follow-up or advanced imaging |
| `none`, `safe`, `limited`, `low` | Consistent with BT-RADS 1–2; routine surveillance |

---

## 9. Implementation Notes

- All thresholds are **surrogate markers** derived from population-level literature; individual variation exists.
- MLS is computed as `centroid_to_midline` distance (proxy, not direct cerebral midline measurement). Confidence is flagged in `distortion_context.atlas_reliability`.
- Motor cortex distance uses atlas-based surface proximity (no tractography). Sub-5 mm results have high clinical weight but should be confirmed with dedicated fMRI/DTI before surgical planning.
- These thresholds intentionally use conservative (lower) cutoffs to prioritize sensitivity over specificity in a decision-support context.
- Do not use these thresholds for autonomous clinical decision-making. Clinician confirmation is always required.

---

*Last updated: 2026-04-13 | Linked implementation: `ai_assistant/core/clinical_implications.py`*

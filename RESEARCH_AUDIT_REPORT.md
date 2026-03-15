# Research / Project Audit

**Subject:** IDT (Information-Dynamic Theory) — project documents and GPU test suite  
**Audit type:** Scientific review, experiment validation, AI/LLM relevance, business value  
**Rule:** Only claims and evidence explicitly present in the provided documents are used. No experiments, datasets, or results are invented.

---

## 1. Document inventory

### What is actually present

| Item | Status | Source / note |
|------|--------|----------------|
| **Problem definition** | Partially present | README/PDF: Test A checks whether M(IDT) predicts training slowdown better than 1/λ_min(H); Test B checks whether τ(small bs) > τ(large bs) when H is matched. "IDT" vs "CSD" predictions are stated. No formal problem statement or research question in auditable text. |
| **Research hypothesis / goal** | Implicit | Documents imply: (1) Fisher G is "active" and M(H,G) captures something H alone does not; (2) under matched H, small-batch runs need more iterations than large-batch to reduce loss by δL ("G-inversion"). Not stated as falsifiable hypotheses with priors. |
| **Methodology** | Present | Test A: ResNet-20, CIFAR-10, 160 epochs, bs=128; checkpoints every 5 epochs; H via Lanczos (R=50), G via Fisher (N_fim=200); τ_obs = 1/|ΔL|; Spearman ρ(tau, M) vs ρ(tau, 1/λ_min(H)); permutation null (500 perms) on G. Test B: three runs (bs=32, 128, 512) with lr ∝ sqrt(bs); match checkpoints by loss and top-5 H eigenvalues; measure τ = iters to reduce loss by 0.02; compare τ_ratio and sign accuracy. |
| **Datasets** | Present but single | Only **CIFAR-10** is used. No ImageNet, no other domains, no NLP. |
| **Experiments** | Described, not reported | Two experiments (A and B) are fully specified in code/README. **No run results** (no pctile_M, tau_ratio, FULL PASS/FAIL counts) are given in the documents. Results are to be produced by running the code. |
| **Evaluation metrics** | Present | Test A: ρ_M, ρ_H, pctile of ρ_M vs permutation null, C1–C4 (M>H, >p95, pctile≥90%, Δρ≥0.05). Test B: tau_ratio (median), sign_acc, ρ(M_ratio, tau_ratio), H_matched (MH_ratio ≈ 1). Verdicts: FULL/PARTIAL/AMBIGUOUS/FAIL. |
| **Baselines / comparisons** | Present but weak | Test A: baseline is 1/λ_min(H) and permutation null (placebo G). Test B: implicit baseline is "H-only" (MH_ratio ≈ 1). **No comparison to other published methods or to prior work** (e.g. sharpness, other curvature metrics, other theories). |
| **Error analysis** | Missing | No analysis of when/why tests fail, no per-checkpoint or per-seed error breakdown, no discussion of variance or outliers. |
| **Limitations** | Missing | No stated limitations (single dataset, single architecture, no statistical power, no sensitivity to hyperparameters). |
| **Reproducibility** | Partially present | Config is fixed (CFG in code); seeds=10; code was embedded in PDF only (no repo script until separately recreated). No Docker, no environment file, no published results to reproduce. |
| **Implementation details** | Present | Full pipeline in idt gpu tests.pdf (and in recreated idt_gpu_tests.py): ResNet-20, HVP, Lanczos, G_r via Fisher in subspace, M(ω) = mean(λ²)/(λ_min·mean(λ)), GEP. |
| **Economic / practical implications** | Missing | No discussion of cost, deployment, or product use. |

### What the documents claim the work proves or demonstrates

- **Test A:** If IDT is correct, M(H,G) should correlate with training slowdown (τ) better than 1/λ_min(H), and ρ_M should sit above the 95th percentile of the permutation null (aligned G).
- **Test B:** If IDT is correct, when H is matched across batch sizes, small-batch runs should need more iterations than large-batch to achieve the same loss drop (τ_small > τ_large); "CSD" would predict τ_small ≈ τ_large.
- **Overall:** Documents imply that "FULL PASS" on both tests would "confirm G-activity" (подтвердить G-активность). **No document provides actual run outcomes** (no table of seeds, pctile_M, tau_ratio, or verdicts). So the claim is conditional: *if* you run the tests and get FULL PASS, *then* the setup is said to support IDT.

---

## 2. Actual scientific contribution

### Classification: **Weak applied engineering work** (bordering on no scientific contribution without run evidence)

- **What is new in the auditable material:**  
  - A concrete operationalization of a metric M(ω) from generalized eigenvalues of H and G (Fisher) in a low-dimensional subspace (Lanczos + Fisher samples).  
  - A blinded placebo test (permutation of G across checkpoints) to test whether aligned M beats aligned H and random G.  
  - A matched-state design (same loss, same top-5 H eigenvalues) to compare τ across batch sizes.

- **Novelty assessment:**  
  - Hessian and Fisher in NN training are not new; Lanczos for top eigenvalues is standard; Spearman and permutation tests are standard. The **combination** (M(ω), placebo, matched-state τ) is a specific recipe. The theoretical content (IDT, "G-activity", "CSD") is referenced in other project documents (GAP3, NN01, Theorem M ≤ κ, etc.) but **those documents were not fully auditable** here; only the test specification and code were. So the **auditable** novelty is a particular evaluation protocol, not a new theory or algorithm.

- **Build-on potential:**  
  Other researchers could reuse the test protocol (same dataset, same model, same metrics) to compare alternative metrics or theories, **if** the code is available and run results are published. As of the provided materials, **no run results are published**, so there is nothing to build on except the protocol itself.

---

## 3. Quality of experiments and testing

| Criterion | Status | Comment |
|-----------|--------|---------|
| Clear experimental setup | Present and convincing | Dataset, model, hyperparameters, seeds, and metrics are specified. |
| Dataset description | Present but weak | Only "CIFAR-10" and standard augmentations; no discussion of suitability or limitations. |
| Baselines | Present but weak | H-only (1/λ_min) and permutation null; no comparison to other curvature/sharpness or optimization literature. |
| Quantitative metrics | Present and convincing | ρ_M, ρ_H, pctile, tau_ratio, sign_acc, verdict criteria are defined. |
| Comparison to existing approaches | Missing | No comparison to other methods or prior work. |
| Multiple scenarios or datasets | Missing | One dataset (CIFAR-10), one architecture (ResNet-20). |
| Ablation studies | Missing | No ablation on R, N_fim, δL, tolerance, or architecture. |
| Robustness analysis | Missing | No sensitivity to hyperparameters or to different optimizers/datasets. |
| Error analysis | Missing | No analysis of failures or variance. |
| Statistical reliability | Present but weak | 10 seeds and permutation null are mentioned; no power analysis, no confidence intervals or significance tests reported (and no results to attach them to). |

### Can this testing be considered comprehensive validation?

**Answer: No.**

- **No run results** are provided. Validation requires at least one full run with reported metrics and verdicts.
- **Single dataset and single model**: comprehensive validation would require multiple datasets and/or architectures.
- **No baselines** from the literature (e.g. other curvature or sharpness-based predictors).
- **No ablations** to justify design choices (R=50, N_fim=200, δL=0.02, etc.).
- **No statistical reporting**: even if results existed, the documents do not specify confidence intervals or significance tests for the verdicts.

The **design** of the tests is clear and implementable; the **execution and reporting** are missing in the provided documents.

---

## 4. Value for LLM / AI engineering

### Classification: **No practical value** for LLM/AI engineering in the auditable material

- The work is about **supervised image classification** (CIFAR-10, ResNet-20) and **optimization geometry** (Hessian, Fisher, batch size, loss landscape).  
- There is **no content** in the auditable test documents on: LLM system design, prompt engineering, RAG, evaluation frameworks, guardrails, or scalable AI infrastructure.  
- The theoretical documents (IDT, GAP3, NN01, etc.) may discuss broader "actualization" or control; **their content was not audited**; only the GPU test suite was.  
- **Conclusion:** For LLM programming and AI engineering as commonly understood (models, APIs, prompts, RAG, safety), the **auditable** material provides **no practical value**. Any claimed broader relevance would require evidence from the theoretical documents, which were not fully reviewed.

---

## 5. Potential business value

| Dimension | Supported by evidence in documents | Plausible but not proven | Claimed but unsupported |
|-----------|------------------------------------|---------------------------|--------------------------|
| Cost reduction | — | — | Not discussed |
| Productivity increase | — | — | Not discussed |
| Automation | — | Test suite could automate one type of audit | No deployment or integration described |
| Product differentiation | — | Possible use as "theory-backed training audit" | No product or customer described |
| Scalability | — | 8–14 h on one GPU for full run | No multi-GPU or scaling analysis |
| Deployment feasibility | — | Depends on PyTorch/CUDA | Not discussed |
| Operational complexity | — | High (training, HVP, Lanczos, many checkpoints) | Not discussed |

- **Value supported by evidence:** None. The documents describe a **research-oriented test suite**, not a product or service.  
- **Plausible but not proven:** If the tests were run and passed, the protocol could in principle be offered as a "training audit" service; no such results or business case are in the documents.  
- **Claimed but unsupported:** No business claims were found in the auditable test/README material.

---

## 6. Weak points and missing evidence

1. **No run results:** The documents define criteria for FULL PASS/FAIL but **do not provide any actual outcomes** (no pctile_M, tau_ratio, or verdict table). So it is **unknown** whether the tests pass or fail in practice.
2. **Claim without evidence:** The phrase "IDT G-активность подтверждена" (IDT G-activity confirmed) appears as a **possible** verdict text in the code, not as a reported finding. No document demonstrates that this verdict has been achieved.
3. **Single setup:** One dataset, one architecture, one optimizer family. No justification for generalizability.
4. **No statistical inference:** Verdicts are based on counts (e.g. ≥6/10 seeds) and thresholds; no p-values, confidence intervals, or power analysis.
5. **No comparison to literature:** No comparison to sharpness, other curvature metrics, or existing optimization/approximation theory.
6. **Theory–test link:** The link between the theoretical documents (IDT, GAP3, NN01, Theorem M ≤ κ) and the two tests is only implied (e.g. "if IDT is true"). The auditable material does not quote or derive the theory; it only implements a specific operationalization.
7. **Reproducibility:** Code was only in PDF form; no standard repo, environment, or data versioning described. A separate reconstruction (idt_gpu_tests.py) was created outside the original documents.
8. **Limitations and failure modes:** Not stated. No discussion of when the tests might fail or be uninformative.

---

## 7. Evidence table

| Claim | Evidence in document | Strength of evidence | Verdict |
|-------|----------------------|----------------------|---------|
| M(ω) = mean(λ²)/(λ_min·mean(λ)) predicts training slowdown better than 1/λ_min(H) | Formula and Test A design (ρ_M vs ρ_H, permutation null) | Protocol only; no run results | **Claim without evidence** |
| Under matched H, small-batch needs more iterations than large-batch to reduce loss (G-inversion) | Test B design (matched checkpoints, tau_ratio, sign_acc) | Protocol only; no run results | **Claim without evidence** |
| Permutation null (500 perms) is appropriate placebo | Description in README and code | Implementation present | **Present** |
| FULL PASS means "IDT G-activity confirmed" | Verdict text in code/README | Definitional only | **Claim without evidence** (no run shows FULL PASS) |
| ResNet-20 on CIFAR-10 is sufficient for validation | Only setup described | Single setup only | **Weak** (no justification for sufficiency) |
| 10 seeds and 500 permutations are sufficient for statistical conclusion | Mentioned in config | No power or CIs | **Weak** |
| Code is reproducible | Config and script in PDF; recreated .py exists | No env/container; results not reproduced | **Partially present** |

---

## 8. Scores (0–10)

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Scientific novelty | 2 | Recombination of known tools (H, G, Lanczos, Spearman, permutation). Specific protocol is new but no theoretical novelty auditable here. |
| Scientific rigor | 2 | Clear design but no reported results, no statistical inference, no baselines from literature, no ablations or limitations. |
| Testing completeness | 2 | Two tests, one dataset, one model; no ablations, no error analysis, no run results. |
| Engineering value for AI/LLM | 0 | Auditable material is only about CNNs on CIFAR-10; no LLM/AI engineering content. |
| Business value | 0 | No business case, no product, no cost/benefit; only a research test suite described. |
| Reproducibility | 3 | Protocol and config are specified; code was in PDF only; no published results to reproduce; run time and hardware noted. |

---

## 9. Final strict conclusion

### A. What is actually proven

- **Nothing** in the sense of empirical scientific proof. The documents specify **two evaluation protocols** (Test A and Test B) and implement them in code. They do **not** provide any run results (no tables of ρ_M, pctile_M, tau_ratio, or verdicts). Therefore, **no claim about IDT or G-activity is demonstrated** in the auditable material.

### B. What is suggested but not proven

- That M(H,G) is a better predictor of training slowdown than 1/λ_min(H) when G is correctly aligned (Test A).  
- That when H is matched across batch sizes, small-batch runs require more iterations than large-batch to reduce loss by δL (Test B).  
- That "FULL PASS" on both tests would support "IDT G-activity."  

All of the above are **conditional on running the tests and obtaining results**, which are **not** in the documents.

### C. What is missing for scientific credibility

- At least one full run with reported metrics and verdicts.  
- Multiple datasets or architectures, or a clear justification for CIFAR-10/ResNet-20 only.  
- Comparison to existing curvature/sharpness or optimization literature.  
- Ablations (e.g. R, N_fim, δL, tolerances).  
- Explicit limitations and failure modes.  
- Statistical reporting (e.g. confidence intervals or significance tests) tied to actual data.

### D. What is missing for business credibility

- Any business case, product description, or customer need.  
- Cost, scalability, or deployment analysis.  
- Evidence that the test outcomes matter for a concrete use case.

### E. Final judgement on whether the testing is truly comprehensive

**No.** The testing **design** is clearly described and implementable, but:

- **No run results** are provided, so the tests have not been validated in the documents.  
- Validation is **narrow** (one dataset, one model, no external baselines, no ablations).  
- There is **no statistical framework** for interpreting verdicts (e.g. power, Type I/II error, or CIs).  

Comprehensive validation would require: (1) at least one complete run with full reporting, (2) broader conditions (e.g. more datasets or architectures), (3) comparison to prior work, and (4) explicit limitations and statistical interpretation. None of these are satisfied in the auditable materials.

---

*Audit conducted only on documents whose content could be extracted (idt gpu tests.pdf, README GPU tests.pdf, and recreated idt_gpu_tests.py). Theoretical documents (IDT Abstract, GAP3, NN01, Theorem M ≤ κ, etc.) were not fully read; their titles and presence are noted but their claims are not part of this evidence table or conclusion.*

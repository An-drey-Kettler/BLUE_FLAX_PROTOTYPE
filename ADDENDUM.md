# ADDENDUM — 25.11.2026

Re-verification of claims in Letter #03 to Daniel Friedman (Active Inference Institute).

## A. Permutation baseline (corrected)

Re-run with fixed seeds (42, 1, 7), 200 permutations each, on 315 semantic chunks
(`test_text.txt`, max_chars=350, min_chars=60, multilingual encoder
`paraphrase-multilingual-MiniLM-L12-v2`).

| seed | perm_mean | perm_std | n_le  | p_emp |
|------|-----------|----------|-------|-------|
| 42   | 0.5183    | 0.0023   | 0/200 | 0.0000 |
| 1    | 0.5182    | 0.0025   | 0/200 | 0.0000 |
| 7    | 0.5182    | 0.0022   | 0/200 | 0.0000 |

Actual mean attractor drift = **0.4050**. Permuted baseline = **0.5182 ± 0.0023**.
Gap ≈ 0.113 at std 0.0023 (~49σ). The value 0.5016 reported in letter #03 was
seed-dependent and is corrected here. Conclusion unchanged and strengthened.

## B. Structural transitions (refined)

Of 14 events at Z ≥ 2.0, two (idx 5, 6) are warm-up artifacts (κ ≈ 640–703 from
near-identity of the running vector at initialization). Excluding them yields
**12 stable transitions**. Per-event Z ranges 2.01–4.32; max Z = 4.32 at idx 305
(drift = 0.9652, κ = 344.97), exhibiting the AXIS_DRIFT profile.

## C. Manual validation and caveat

Three transitions manually validated:

| idx | Z    | κ      | drift  | class         | manual annotation |
|-----|------|--------|--------|---------------|-------------------|
| 305 | 4.32 | 344.97 | 0.9652 | AXIS_DRIFT    | technical → personal/existential register shift, with self-correction |
| 238 | 3.53 | 390.99 | 0.8118 | AXIS_DRIFT    | technical/philosophical → emotional direct address |
| 236 | 3.40 | 539.58 | 0.6020 | RESONANT_LEAP | code → anomaly log → hallucination quote (coherent) |

**Caveat (honest):** chunks 237–238 contain embedded report artifacts
("CURRENT CHUNK (Trigger)"), which may partially inflate drift. The corpus
`test_text.txt` is a composite: it embeds code fragments (`vector_stabilizer_v2.py`,
`vector_stabilizer_v3.py`), anomaly logs (`anomaly_002.txt`), and the
`How_can_we_help_Gavin_Wood.txt` excerpt. Both AXIS_DRIFT cases correspond to
shifts into a personal/emotional register, whereas RESONANT_LEAP occurs within
technical context — a hypothesis to test on a larger, cleaned corpus.

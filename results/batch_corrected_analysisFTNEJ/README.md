# Batch-aware full-trace analysis

See `analysis/notebooks/batch_corrected_analysisFTNEJ.ipynb` for method decisions,
assumptions, executed plots, and interpretation. Inputs are the 90 individual
whole-trace CSVs in the three batch folders, with five analyzed features.

Repeated runs are averaged within batch/node/condition. Conditions are contrasted
within batch/node, differences are averaged across available batches within node,
and nodes receive equal weight. Primary exact sign-flip tests use one difference
per node; Bonferroni correction covers all 15 feature/pair combinations.

The first (joy-only) batch is descriptive only. E−N has 15 independent matched
nodes; each joy contrast has three, all in the second batch.

E−N spectral entropy is the only corrected significant comparison in the saved
run: mean difference 0.01736, corrected p=0.001831. Sparse joy support prevents
strong inference; nonsignificance is not equivalence. All results are exploratory.

Exports contain input provenance, node means, pairing support, individual matched
differences, node-level differences, primary tests, batch-specific descriptive
summaries, same-node E−N batch changes, and input hashes. This method estimates
matched contrasts; it does not produce universally batch-corrected trace values.

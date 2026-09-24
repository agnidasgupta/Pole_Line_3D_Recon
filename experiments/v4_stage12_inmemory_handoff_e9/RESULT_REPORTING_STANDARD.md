# Result reporting standard

Every experiment result must contain both raw measurements and a concise
numeric summary with an explicit **ACCEPTED** or **NOT_ACCEPTED** decision.

The summary must include every reported timing metric's count, mean, P50 and
P95 when available; run/session/slice counts; equivalence and output-comparison
counts; the measured percentage change; and the gate threshold. Raw CSV/JSON
data remains linked or embedded below the summary. A result may not be called
accepted merely because incomplete ground-truth labels show a metric change:
known positive labels must be preserved, while an unlabeled voxel is not proof
of a false positive.

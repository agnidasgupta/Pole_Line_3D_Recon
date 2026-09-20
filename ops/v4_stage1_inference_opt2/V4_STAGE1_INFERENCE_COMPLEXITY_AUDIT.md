# V4 Stage 1 inference-complexity audit

This audit separates optimizations that can be tested under the accepted V4
production contract from changes that require a new trained model. The current
checkpoint, calibration and incomplete annotation set are not changed.

## Contract boundary

The V4 production-preserving experiments keep the accepted checkpoint,
calibration, complete five-head network, active-GPU execution, BF16, batch 12,
channels-last, 64-cube inputs, 48-cube output cores, fusion, thresholds and
serialization. Exact saved-production comparison with `SCORE_ATOL=0` is an
implementation regression gate. It is not a claim that unlabelled voxels are
background; visually confirmed poles and lines may be absent from annotations.

No training change can keep the existing checkpoint's inference results exactly
unchanged. Training-level ideas below therefore belong to a separately versioned
future model and cannot be promoted as an optimization of accepted V4.

## Measured execution facts

The accepted E0 Nsight profile used a representative 200,634-row slice and five
measured iterations:

| Item | Result |
|---|---:|
| Mean profiled wall time | 176.667 ms |
| GPU model and score fusion | 151.343 ms |
| Core scheduling | 15.290 ms |
| D2H gather | 12.852 ms |
| Feature assembly | 5.175 ms |
| Ordinary kernel launches | 10,425 total / 2,085 per iteration |
| `cudaStreamSynchronize` | 70 calls / 520.297 ms API time |
| CUDA-event recording | 0.4% of CUDA API time |

The checkpoint has 341,518 parameters. Its structure is a three-level 3-D
encoder/decoder with ten 3×3×3 convolutions, two transposed convolutions,
GroupNorm and SiLU, followed by semantic, pole, line, objectness and embedding
1×1×1 heads.

The five heads contain only 238 parameters, but their dense output tensors span
the full 64-cube volume. The embedding head alone emits eight channels and is
not consumed by score fusion or output serialization. The current
`precision_recovery_loss()` also does not consume it, so the present training
entry point computes an embedding tensor without an embedding-loss term. It is
nevertheless kept at V4 inference because the contract explicitly requires
every checkpoint head.

## Contract-safe experiment candidates

### E5: cached coordinates and reusable model input — rejected

The x/y/z coordinate values are deterministic functions of the fixed grid,
patch size and core center. E5 caches the exact FP32 one-dimensional coordinate
vectors and writes occupancy, coordinates and distance directly into a reusable
channels-last input tensor. This removes repeated coordinate arithmetic,
`torch.cat`, and the subsequent channels-last materialization while keeping the
model input values and layout identical.

E5 passed representative exact-output tests and reduced the two-run Stage 1
mean by 1.39%. Its full-data gate stopped after 14 accepted sessions when a
pole score changed by `0.00019707530736923218`. A fresh E0 run reproduced the
complete failing session exactly. E5 therefore changed execution behavior and
is rejected; its full-data run must not be resumed.

### E5b: production-assembled reference coordinate cache — selected

E5b removes E5's reusable final model-input buffer. Coordinate values are
generated on cache misses by the exact production batched FP32 expression, but
production `torch.cat`, channels-last conversion and new final-input allocation
remain in place for every batch. Only immutable one-dimensional coordinate
values are cached.

Its CUDA self-test compares complete E0/E5b inputs with `torch.equal` for all
405 possible active-core centers and every partial fixed-batch padding count.
The first saved-production gate is the 157-slice session that rejected E5.

### Reuse the objectness sigmoid — suitable later

`fuse_scores()` computes `sigmoid(objectness)`, after which
`_run_model_scores()` computes the same sigmoid again for the saved objectness
output. Returning and reusing the first tensor should remove one dense sigmoid
without changing its value. This is likely exact and low risk, but it should be
an isolated experiment only after E5b is accepted or rejected.

### Evaluate all heads only on the consumed 48-cube core — conditional

Every output head is 1×1×1, so a voxel's head result depends only on the decoder
feature at that voxel. Cropping the decoder feature from 64³ to the consumed 48³
core before applying all five heads is mathematically identical for every saved
voxel and avoids head, softmax and sigmoid work on the discarded 57.8% border.

This preserves all heads and all saved results, but changes the internal forward
shape. It should be attempted only if the production contract explicitly allows
discarded border-head computation to be omitted, followed by exact CUDA and
full-data gates.

### Concatenate the five 1×1×1 heads — conditional

The five head weights and biases can be concatenated into one 14-channel 1×1×1
convolution and split into the same named outputs. This is mathematically
equivalent and reduces launches, but a different cuDNN kernel may change floating
point rounding. It is lower priority than core-only head evaluation and requires
an exact-output rejection gate.

### CUDA Graph replay — currently blocked

The trace contains more than two thousand ordinary launches per measured
iteration. CUDA Graph replay could reduce CPU launch overhead without changing
weights, but requires capture, static buffers and warm-up. The accepted contract
currently fixes eager execution and zero warm-up, so this is not an eligible V4
candidate unless that execution-only restriction is explicitly revised.

## Changes that are not production-preserving

The following can reduce inference cost only by creating a new model or runtime
contract. They must not be mixed into an E-series V4 optimization:

| Change | Why it is outside accepted V4 |
|---|---|
| Skip or remove the embedding head | Violates the complete-head contract even though the output is unused |
| Replace GroupNorm with BatchNorm or another foldable norm | Changes activations and requires training or recalibration |
| Reduce `base_channels` or encoder/decoder depth | Changes architecture and checkpoint shapes |
| Depthwise/separable or sparse 3-D convolutions | Changes operators, weights and outputs |
| Structured pruning | Changes the trained network |
| FP8/INT8 quantization | Changes numerical results and requires calibration/training |
| `torch.compile`, TensorRT or ONNX fusion | Changes the accepted eager runtime and may change rounding |
| Smaller patches, cores or batch size | Changes context, batching or the accepted production path |

## Training-level direction for a future model

If a separately versioned successor is allowed to produce different scores, the
highest-value training-for-inference design is:

1. train a core-output network that produces only the central 48³ predictions;
2. omit the unused embedding head or use it only in a training teacher;
3. replace GroupNorm with a deployment-foldable normalization strategy;
4. distill the current V4 pole, line, semantic and objectness outputs into a
   narrower decoder or separable-convolution student;
5. validate using visual review and positive-preservation criteria that respect
   incomplete labels, rather than treating every unlabelled prediction as false.

That work may improve inference substantially, but it is a new model-development
project—not a complexity reduction that can preserve every V4 inference result.

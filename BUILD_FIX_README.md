# V10 Unity native GPU build correction

## Diagnosed failures

The first Unity 6000.3.15f1 build log reported missing
`Unity.InferenceEngine` types. The supplied `Packages/manifest.json` and
`packages-lock.json` confirmed that `com.unity.ai.inference` was not installed.
The corrected installer pins `com.unity.ai.inference` to `2.6.1` before Unity
compilation.

After that dependency resolved, the latest full build log reported exactly one
unique compiler diagnostic: CS0136 at
`V10NativeStage2Reconstructor.cs(777,61)`. That source error is corrected in
this package.

The following compiler pass then exposed one unique Editor-only diagnostic:
CS0246 at `V10Stage12Build.cs(75,23)`. `BuildFailedException` is declared in
`UnityEditor.Build`, while `BuildReport` and `BuildResult` are declared in
`UnityEditor.Build.Reporting`. The build helper now imports both namespaces.

## Production corrections

- Standard Linux player with `BackendType.GPUCompute` is now the default.
- FP32 remains mandatory until FP16 passes the real-slice parity gate.
- Batch size is fixed to the accepted value of 12.
- GPU launch uses Vulkan and never adds `-nographics`.
- Runtime rejects a missing graphics device or missing compute-shader support.
- Input can be a Stage 1 manifest, one raw unlabeled CSV/CSV.GZ, or a raw
  directory. Raw files require `x,y,z`; `dist_center_ft` is optional.
- Stage 1 inference is saved and passed directly in memory to unchanged native
  Stage 2 reconstruction.
- The Stage 2 algorithm and ExtraTrees runtime are unchanged. One C# local
  variable was renamed to remove compiler error CS0136; this does not alter
  control flow, ordering, geometry, thresholds, or output values.

## Second compiler diagnostic

After Sentis 2.6.1 resolved, Unity compilation exposed one unique compiler
error: `V10NativeStage2Reconstructor.cs` used `values` for both an inner local
list and the enclosing component sequence. The inner list is now named
`componentRows` and the sequence is named `components`. This is a semantic
no-op and leaves reconstruction behavior unchanged.

## Third compiler diagnostic

The build helper already imported `UnityEditor.Build.Reporting`, but that does
not import its parent namespace. It now also imports `UnityEditor.Build`, which
resolves `BuildFailedException`. This affects only how an unsuccessful Unity
Editor build is reported; it is not compiled into or called by Stage 1 or
Stage 2 runtime processing.

## Mac upgrade and build

Extract this ZIP over the experimental Git worktree, rerun
`install_v10_unity_native_code_on_mac.sh`, and then rerun
`build_v10_unity_linux_server_on_mac.sh`. Existing model assets under
`Assets/V10Stage12/Models` are retained.

The build is not release-qualified until the generated Linux GPU player passes
the automatic Velasco ordinals 20-39 equivalence gate and then the complete
30-session gate against the accepted opt1-fix2 Stage 2 run.

## Confirmed build result

The user confirmed a successful Unity 6000.3.15f1 Linux GPU player build after
installing Linux Build Support (Mono and IL2CPP). The build log registered both
MacStandaloneSupport and LinuxStandaloneSupport and ended with
`BUILD_KIND=gpu-player`. Runtime output equivalence remains a separate required
acceptance gate.

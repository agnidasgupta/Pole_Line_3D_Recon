using Unity.InferenceEngine;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace VegetationAssurance.V10
{
    /// <summary>
    /// Unity 6.3 LTS / Sentis 2.6.1 implementation of the accepted V4 Stage-1
    /// sparse 64^3-patch, 48^3-core inference contract.
    /// </summary>
    public sealed class V10Stage12SentisInferenceManager : MonoBehaviour
    {
        private sealed class PreparedVoxel
        {
            public SparseVoxel voxel;
            public float distance;
            public int stableOrder;
        }

        private sealed class CoreGroup
        {
            public Int3Key key;
            public readonly List<int> rows = new List<int>();
        }

        [Header("Models")]
        [SerializeField] private ModelAsset fp32ModelAsset;
        [SerializeField] private ModelAsset fp16ModelAsset;
        [SerializeField] private TextAsset sidecarJson;

        [Header("Runtime")]
        [SerializeField] private bool requestFp16;
        [SerializeField] private bool unityFp16ParityApproved;
        [SerializeField, Range(1, 12)] private int batchSize = 1;
        // CPU is the safe default for a true Dedicated Server / -nographics
        // player. GPUCompute is selected only by the standard-player build.
        [SerializeField] private BackendType backend = BackendType.CPU;
        [SerializeField] private bool warmupOnStart = true;
        [SerializeField] private bool logDiagnostics = true;

        private const string PoleOutputName = "pole_score";
        private const string LineOutputName = "line_score";
        private const int Channels = 5;
        private const int Patch = 64;
        private const int Core = 48;
        private const int CoreVolume = Core * Core * Core;

        private Worker worker;
        private V10Sidecar config;
        private Task initializationTask;
        private readonly SemaphoreSlim workerLock = new SemaphoreSlim(1, 1);
        private bool destroyed;

        public bool IsReady { get; private set; }
        public V10Sidecar Configuration => config;

        private void Awake()
        {
            if (fp32ModelAsset != null && sidecarJson != null)
                initializationTask = InitializeAsync(warmupOnStart);
        }

        public Task InitializeAsync(bool runWarmup)
        {
            if (initializationTask == null || initializationTask.IsFaulted || initializationTask.IsCanceled)
                initializationTask = InitializeCoreAsync(runWarmup);
            return initializationTask;
        }

        private async Task InitializeCoreAsync(bool runWarmup)
        {
            await workerLock.WaitAsync();
            try
            {
                if (destroyed) throw new ObjectDisposedException(nameof(V10Stage12SentisInferenceManager));
                IsReady = false;
                worker?.Dispose();
                worker = null;
                if (fp32ModelAsset == null || sidecarJson == null)
                    throw new InvalidOperationException("FP32 model and sidecar are required.");
                config = JsonUtility.FromJson<V10Sidecar>(sidecarJson.text);
                ValidateSidecar(config);
                if (requestFp16 && (!config.onnx.fp16_deployable || !unityFp16ParityApproved))
                    throw new InvalidOperationException("FP16 requires both export and Unity real-slice parity approval.");
                ModelAsset selected = requestFp16 && fp16ModelAsset != null ? fp16ModelAsset : fp32ModelAsset;
                if (backend == BackendType.GPUCompute)
                {
                    if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
                        throw new InvalidOperationException(
                            "GPUCompute requires a graphics device. Do not launch with -nographics.");
                    if (!SystemInfo.supportsComputeShaders)
                        throw new InvalidOperationException(
                            "GPUCompute requires compute-shader support on the active graphics device.");
                }
                worker = new Worker(ModelLoader.Load(selected), backend);
                if (runWarmup) await WarmupLockedAsync();
                IsReady = true;
                if (logDiagnostics) Debug.Log(
                    "[V10Stage12] initialization complete; backend=" + backend +
                    " graphics_device=" + SystemInfo.graphicsDeviceName +
                    " graphics_api=" + SystemInfo.graphicsDeviceType +
                    " compute_shaders=" + SystemInfo.supportsComputeShaders +
                    " batch_size=" + batchSize +
                    " precision=" + (requestFp16 ? "fp16" : "fp32"));
            }
            catch
            {
                IsReady = false;
                throw;
            }
            finally
            {
                workerLock.Release();
            }
        }

        private static void ValidateSidecar(V10Sidecar value)
        {
            if (value == null || value.input == null || value.calibration == null || value.onnx == null ||
                value.stage2 == null || value.stage2.profile == null || value.stage2.refiner_tree_metadata == null)
                throw new InvalidOperationException("Invalid v10_stage12_sidecar.json.");
            if (value.format_version != 1)
                throw new InvalidOperationException("Unsupported sidecar format_version=" + value.format_version);
            if (value.input.grid_size_xyz == null || value.input.grid_size_xyz.Length != 3)
                throw new InvalidOperationException("sidecar grid_size_xyz must contain three values.");
            if (value.input.grid_size_xyz.Any(v => v <= 0))
                throw new InvalidOperationException("sidecar grid dimensions must be positive.");
            if (string.IsNullOrWhiteSpace(value.stage2.refiner_tree_metadata.sha256))
                throw new InvalidOperationException("sidecar Stage2 refiner tree SHA256 is missing.");
        }

        private async Task WarmupLockedAsync()
        {
            using var input = new Tensor<float>(
                new TensorShape(1, Channels, Patch, Patch, Patch),
                new float[Channels * Patch * Patch * Patch]);
            worker.Schedule(input);
            Tensor<float> pole = worker.PeekOutput(PoleOutputName) as Tensor<float>;
            Tensor<float> line = worker.PeekOutput(LineOutputName) as Tensor<float>;
            if (pole == null || line == null)
                throw new InvalidOperationException("ONNX outputs pole_score and line_score were not found.");
            using var poleCpu = await pole.ReadbackAndCloneAsync();
            using var lineCpu = await line.ReadbackAndCloneAsync();
            if (poleCpu.DownloadToArray().Length != CoreVolume || lineCpu.DownloadToArray().Length != CoreVolume)
                throw new InvalidOperationException("Unexpected Stage-1 warmup output shape.");
        }

        public async Task<Stage1SliceResult> InferSliceAsync(IReadOnlyList<SparseVoxel> inputVoxels)
        {
            if (initializationTask == null)
                initializationTask = InitializeAsync(warmupOnStart);
            await initializationTask;
            await workerLock.WaitAsync();
            try { return await InferLockedAsync(inputVoxels ?? Array.Empty<SparseVoxel>()); }
            finally { workerLock.Release(); }
        }

        private async Task<Stage1SliceResult> InferLockedAsync(IReadOnlyList<SparseVoxel> inputVoxels)
        {
            var result = new Stage1SliceResult();
            double started = Time.realtimeSinceStartupAsDouble;
            try
            {
                if (!IsReady || worker == null) throw new InvalidOperationException("Sentis worker is not ready.");
                int[] grid = config.input.grid_size_xyz;
                double prepareStart = Time.realtimeSinceStartupAsDouble;
                PreparedVoxel[] prepared = Prepare(inputVoxels, grid);
                result.voxels = prepared.Select(v => v.voxel).ToArray();
                result.normalizedDistance = prepared.Select(v => v.distance).ToArray();
                result.poleScore = new float[prepared.Length];
                result.lineScore = new float[prepared.Length];
                result.deployedLabel = new byte[prepared.Length];
                var index = new Dictionary<Int3Key, int>(prepared.Length);
                for (int i = 0; i < prepared.Length; i++)
                    index[new Int3Key(prepared[i].voxel.x, prepared[i].voxel.y, prepared[i].voxel.z)] = i;
                List<CoreGroup> cores = ActiveCores(prepared);
                result.activeCores = cores.Count;
                result.prepareSeconds = Time.realtimeSinceStartupAsDouble - prepareStart;

                int actualBatch = Math.Max(1, Math.Min(12, batchSize));
                for (int start = 0; start < cores.Count; start += actualBatch)
                {
                    int count = Math.Min(actualBatch, cores.Count - start);
                    List<CoreGroup> current = cores.GetRange(start, count);
                    float[] features = BuildBatch(current, prepared, index, grid);
                    using var tensor = new Tensor<float>(
                        new TensorShape(count, Channels, Patch, Patch, Patch), features);
                    double scheduleStart = Time.realtimeSinceStartupAsDouble;
                    worker.Schedule(tensor);
                    result.scheduleSeconds += Time.realtimeSinceStartupAsDouble - scheduleStart;
                    Tensor<float> poleTensor = worker.PeekOutput(PoleOutputName) as Tensor<float>;
                    Tensor<float> lineTensor = worker.PeekOutput(LineOutputName) as Tensor<float>;
                    if (poleTensor == null || lineTensor == null)
                        throw new InvalidOperationException("Expected pole_score and line_score outputs.");
                    double readStart = Time.realtimeSinceStartupAsDouble;
                    using var poleCpu = await poleTensor.ReadbackAndCloneAsync();
                    using var lineCpu = await lineTensor.ReadbackAndCloneAsync();
                    float[] pole = poleCpu.DownloadToArray();
                    float[] line = lineCpu.DownloadToArray();
                    result.readbackSeconds += Time.realtimeSinceStartupAsDouble - readStart;
                    if (pole.Length != count * CoreVolume || line.Length != count * CoreVolume)
                        throw new InvalidOperationException("Unexpected Stage-1 output length.");
                    for (int b = 0; b < count; b++)
                    {
                        CoreGroup group = current[b];
                        int ox = group.key.x * Core;
                        int oy = group.key.y * Core;
                        int oz = group.key.z * Core;
                        foreach (int row in group.rows)
                        {
                            SparseVoxel voxel = prepared[row].voxel;
                            int local = ((voxel.z - oz) * Core + voxel.y - oy) * Core + voxel.x - ox;
                            result.poleScore[row] = pole[b * CoreVolume + local];
                            result.lineScore[row] = line[b * CoreVolume + local];
                        }
                    }
                }

                for (int i = 0; i < prepared.Length; i++)
                    result.deployedLabel[i] = Label(result.poleScore[i], result.lineScore[i]);
                result.success = true;
            }
            catch (Exception exception)
            {
                result.success = false;
                result.error = exception.ToString();
                if (logDiagnostics) Debug.LogError("[V10Stage12] inference failed: " + exception);
            }
            result.totalSeconds = Time.realtimeSinceStartupAsDouble - started;
            return result;
        }

        private byte Label(float pole, float line)
        {
            bool p = pole >= config.calibration.pole_threshold;
            bool l = line >= config.calibration.line_threshold;
            if (p && (!l || pole >= line)) return 1;
            return l ? (byte)2 : (byte)0;
        }

        private static PreparedVoxel[] Prepare(IReadOnlyList<SparseVoxel> source, int[] grid)
        {
            var retained = new Dictionary<Int3Key, PreparedVoxel>();
            // Match build_sparse_item_from_dataframe exactly: normalize over all
            // valid source rows first, then retain the last duplicate coordinate.
            float maxDistance = 1f;
            for (int i = 0; i < source.Count; i++)
            {
                SparseVoxel voxel = source[i];
                if (voxel.x < 0 || voxel.x >= grid[0] || voxel.y < 0 || voxel.y >= grid[1] ||
                    voxel.z < 0 || voxel.z >= grid[2]) continue;
                if (!float.IsNaN(voxel.distCenterFt))
                    maxDistance = Math.Max(maxDistance, Math.Abs(voxel.distCenterFt));
                retained[new Int3Key(voxel.x, voxel.y, voxel.z)] =
                    new PreparedVoxel { voxel = voxel, stableOrder = i };
            }
            PreparedVoxel[] output = retained.Values
                .OrderBy(v => v.voxel.z)
                .ThenBy(v => v.stableOrder)
                .ToArray();
            foreach (PreparedVoxel item in output)
            {
                float normalized = item.voxel.distCenterFt / maxDistance;
                item.distance = float.IsNaN(normalized) || float.IsInfinity(normalized) ? 0f : normalized;
            }
            return output;
        }

        private static List<CoreGroup> ActiveCores(PreparedVoxel[] voxels)
        {
            var groups = new Dictionary<Int3Key, CoreGroup>();
            for (int row = 0; row < voxels.Length; row++)
            {
                SparseVoxel voxel = voxels[row].voxel;
                var key = new Int3Key(voxel.x / Core, voxel.y / Core, voxel.z / Core);
                if (!groups.TryGetValue(key, out CoreGroup group))
                {
                    group = new CoreGroup { key = key };
                    groups.Add(key, group);
                }
                group.rows.Add(row);
            }
            List<CoreGroup> output = groups.Values.ToList();
            output.Sort((a, b) => a.key.CompareTo(b.key));
            return output;
        }

        private static float[] BuildBatch(List<CoreGroup> cores, PreparedVoxel[] voxels,
            Dictionary<Int3Key, int> lookup, int[] grid)
        {
            int patchVolume = Patch * Patch * Patch;
            float[] output = new float[cores.Count * Channels * patchVolume];
            for (int b = 0; b < cores.Count; b++)
            {
                int cx = cores[b].key.x * Core + Core / 2;
                int cy = cores[b].key.y * Core + Core / 2;
                int cz = cores[b].key.z * Core + Core / 2;
                int x0 = cx - Patch / 2;
                int y0 = cy - Patch / 2;
                int z0 = cz - Patch / 2;
                int batchBase = b * Channels * patchVolume;
                for (int z = 0; z < Patch; z++)
                for (int y = 0; y < Patch; y++)
                for (int x = 0; x < Patch; x++)
                {
                    int flat = (z * Patch + y) * Patch + x;
                    int gx = x0 + x;
                    int gy = y0 + y;
                    int gz = z0 + z;
                    if (lookup.TryGetValue(new Int3Key(gx, gy, gz), out int row))
                    {
                        output[batchBase + flat] = 1f;
                        output[batchBase + 4 * patchVolume + flat] = voxels[row].distance;
                    }
                    output[batchBase + patchVolume + flat] = NormalizeCoordinate(gx, grid[0]);
                    output[batchBase + 2 * patchVolume + flat] = NormalizeCoordinate(gy, grid[1]);
                    output[batchBase + 3 * patchVolume + flat] = NormalizeCoordinate(gz, grid[2]);
                }
            }
            return output;
        }

        private static float NormalizeCoordinate(int value, int size) =>
            Mathf.Clamp(value / (float)Math.Max(size - 1, 1) * 2f - 1f, -1.5f, 1.5f);

        public static void WriteStage1Csv(string path, Stage1SliceResult result)
        {
            if (result == null || !result.success)
                throw new InvalidOperationException("Cannot write a failed Stage-1 result.");
            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full) ?? ".");
            string temporary = full + ".tmp." + Guid.NewGuid().ToString("N");
            try
            {
                using (var writer = new StreamWriter(temporary, false, new UTF8Encoding(false)))
                {
                    writer.WriteLine("source_row,x,y,z,dist_values,pole,line,deployed_label");
                    for (int i = 0; i < result.voxels.Length; i++)
                    {
                        SparseVoxel voxel = result.voxels[i];
                        writer.Write(voxel.sourceRow.ToString(CultureInfo.InvariantCulture)); writer.Write(',');
                        writer.Write(voxel.x); writer.Write(','); writer.Write(voxel.y); writer.Write(','); writer.Write(voxel.z); writer.Write(',');
                        writer.Write(result.normalizedDistance[i].ToString("R", CultureInfo.InvariantCulture)); writer.Write(',');
                        writer.Write(result.poleScore[i].ToString("R", CultureInfo.InvariantCulture)); writer.Write(',');
                        writer.Write(result.lineScore[i].ToString("R", CultureInfo.InvariantCulture)); writer.Write(',');
                        writer.WriteLine(result.deployedLabel[i]);
                    }
                }
                AtomicReplace(temporary, full);
            }
            finally
            {
                if (File.Exists(temporary)) File.Delete(temporary);
            }
        }

        internal static void AtomicReplace(string temporary, string destination)
        {
            if (!File.Exists(destination))
            {
                File.Move(temporary, destination);
                return;
            }
            string backup = destination + ".bak." + Guid.NewGuid().ToString("N");
            try
            {
                File.Replace(temporary, destination, backup, true);
                if (File.Exists(backup)) File.Delete(backup);
            }
            catch (PlatformNotSupportedException)
            {
                File.Move(destination, backup);
                try { File.Move(temporary, destination); }
                catch
                {
                    if (!File.Exists(destination) && File.Exists(backup)) File.Move(backup, destination);
                    throw;
                }
                if (File.Exists(backup)) File.Delete(backup);
            }
        }

        private void OnDestroy()
        {
            destroyed = true;
            IsReady = false;
            worker?.Dispose();
            worker = null;
            workerLock.Dispose();
        }
    }
}

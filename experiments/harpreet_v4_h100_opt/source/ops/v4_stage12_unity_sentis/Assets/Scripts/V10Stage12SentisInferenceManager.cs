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

/// <summary>
/// Unity 6.3 LTS / Sentis 2.6.1 manager for the accepted V4 Stage-1 model.
///
/// Input is one slice of sparse local voxels. The manager reproduces the Python
/// 64^3 patch / 48^3 core scheduler and five input channels and returns calibrated
/// Stage-1 pole/line scores. Exact V10 Stage-2 reconstruction remains in the
/// packaged Python reference and consumes WriteStage1Csv output inside Docker.
/// Keeping the accepted Stage-2 code avoids an unvalidated C# geometry rewrite.
/// </summary>
public sealed class V10Stage12SentisInferenceManager : MonoBehaviour
{
    [Serializable]
    public struct SparseVoxel
    {
        public int x;
        public int y;
        public int z;
        public float distCenterFt;
        public long sourceRow;

        public SparseVoxel(int x, int y, int z, float distCenterFt = 0f, long sourceRow = -1)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.distCenterFt = distCenterFt;
            this.sourceRow = sourceRow;
        }
    }

    public sealed class Stage1Result
    {
        public bool success;
        public string error;
        public SparseVoxel[] voxels;
        public float[] normalizedDistance;
        public float[] poleScore;
        public float[] lineScore;
        public byte[] deployedLabel;
        public int activeCores;
        public double prepareSeconds;
        public double scheduleSeconds;
        public double readbackSeconds;
        public double totalSeconds;
    }

    [Serializable]
    private sealed class GridConfig
    {
        public int[] grid_size_xyz = { 400, 400, 200 };
    }

    [Serializable]
    private sealed class CalibrationConfig
    {
        public float pole_threshold = .2125f;
        public float line_threshold = .2125f;
    }

    [Serializable]
    private sealed class OnnxConfig
    {
        public bool fp16_deployable = false;
    }

    [Serializable]
    private sealed class SidecarConfig
    {
        public OnnxConfig onnx = new OnnxConfig();
        public GridConfig input = new GridConfig();
        public CalibrationConfig calibration = new CalibrationConfig();
    }

    private readonly struct Key : IEquatable<Key>, IComparable<Key>
    {
        public readonly int x, y, z;
        public Key(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }
        public bool Equals(Key other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is Key other && Equals(other);
        public override int GetHashCode()
        {
            unchecked { return ((x * 397) ^ y) * 397 ^ z; }
        }
        public int CompareTo(Key other)
        {
            int c = z.CompareTo(other.z);
            if (c != 0) return c;
            c = y.CompareTo(other.y);
            return c != 0 ? c : x.CompareTo(other.x);
        }
    }

    private sealed class PreparedVoxel
    {
        public SparseVoxel voxel;
        public float dist;
        public int stableOrder;
    }

    private sealed class CoreGroup
    {
        public Key key;
        public readonly List<int> rows = new List<int>();
    }

    [Header("Models")]
    [SerializeField] private ModelAsset fp32ModelAsset;
    [Tooltip("Optional. Used only when the sidecar parity gate says FP16 is deployable.")]
    [SerializeField] private ModelAsset fp16ModelAsset;
    [SerializeField] private TextAsset sidecarJson;

    [Header("Runtime")]
    [SerializeField] private bool requestFp16 = false;
    [Tooltip("Set true only after Unity FP16 output has passed the real-slice parity gate.")]
    [SerializeField] private bool unityFp16ParityApproved = false;
    [SerializeField, Range(1, 12)] private int batchSize = 1;
    [SerializeField] private BackendType backend = BackendType.GPUCompute;
    [SerializeField] private bool warmupOnStart = true;
    [SerializeField] private bool logDiagnostics = true;

    private const string InputName = "volume";
    private const string PoleOutputName = "pole_score";
    private const string LineOutputName = "line_score";
    private const int Channels = 5;
    private const int Patch = 64;
    private const int Core = 48;
    private const int CoreVolume = Core * Core * Core;
    private Worker _worker;
    private SidecarConfig _config;
    private readonly SemaphoreSlim _inferenceLock = new SemaphoreSlim(1, 1);

    public bool IsReady { get; private set; }

    private void Awake()
    {
        if (fp32ModelAsset != null && sidecarJson != null)
            Initialize(warmupOnStart);
    }

    public void Initialize(bool runWarmup)
    {
        IsReady = false;
        _worker?.Dispose();
        _worker = null;
        if (fp32ModelAsset == null || sidecarJson == null)
            throw new InvalidOperationException("FP32 model and sidecar are required.");
        _config = JsonUtility.FromJson<SidecarConfig>(sidecarJson.text);
        if (_config == null || _config.input == null || _config.calibration == null)
            throw new InvalidOperationException("Invalid v10_stage12_sidecar.json.");
        if (requestFp16 && (_config.onnx == null || !_config.onnx.fp16_deployable || !unityFp16ParityApproved))
            throw new InvalidOperationException("FP16 requires both Nebius and Unity real-slice parity approval.");
        ModelAsset selected = requestFp16 && fp16ModelAsset != null ? fp16ModelAsset : fp32ModelAsset;
        _worker = new Worker(ModelLoader.Load(selected), backend);
        IsReady = true;
        if (runWarmup) _ = WarmupAsync();
    }

    private async Task WarmupAsync()
    {
        try
        {
            using var input = new Tensor<float>(new TensorShape(1, Channels, Patch, Patch, Patch),
                new float[Channels * Patch * Patch * Patch]);
            _worker.Schedule(input);
            var pole = _worker.PeekOutput(PoleOutputName) as Tensor<float>;
            var line = _worker.PeekOutput(LineOutputName) as Tensor<float>;
            if (pole == null || line == null) throw new InvalidOperationException("ONNX outputs not found.");
            using var poleCpu = await pole.ReadbackAndCloneAsync();
            using var lineCpu = await line.ReadbackAndCloneAsync();
            if (poleCpu.DownloadToArray().Length != CoreVolume || lineCpu.DownloadToArray().Length != CoreVolume)
                throw new InvalidOperationException("Unexpected warmup output shape.");
            if (logDiagnostics) Debug.Log("[V10Stage12] Sentis warmup passed.");
        }
        catch (Exception exception)
        {
            Debug.LogError("[V10Stage12] warmup failed: " + exception);
            IsReady = false;
        }
    }

    public async Task<Stage1Result> InferSliceAsync(IReadOnlyList<SparseVoxel> inputVoxels)
    {
        await _inferenceLock.WaitAsync();
        try { return await InferLockedAsync(inputVoxels); }
        finally { _inferenceLock.Release(); }
    }

    private async Task<Stage1Result> InferLockedAsync(IReadOnlyList<SparseVoxel> inputVoxels)
    {
        var result = new Stage1Result();
        double started = Time.realtimeSinceStartupAsDouble;
        try
        {
            if (!IsReady) throw new InvalidOperationException("Sentis worker is not ready.");
            int[] grid = _config.input.grid_size_xyz;
            if (grid == null || grid.Length != 3) throw new InvalidOperationException("sidecar grid_size_xyz invalid");
            double prepStart = Time.realtimeSinceStartupAsDouble;
            PreparedVoxel[] voxels = Prepare(inputVoxels, grid);
            result.voxels = voxels.Select(v => v.voxel).ToArray();
            result.normalizedDistance = voxels.Select(v => v.dist).ToArray();
            result.poleScore = new float[voxels.Length];
            result.lineScore = new float[voxels.Length];
            result.deployedLabel = new byte[voxels.Length];
            var index = new Dictionary<Key, int>(voxels.Length);
            for (int i = 0; i < voxels.Length; i++)
                index[new Key(voxels[i].voxel.x, voxels[i].voxel.y, voxels[i].voxel.z)] = i;
            var cores = ActiveCores(voxels);
            result.activeCores = cores.Count;
            result.prepareSeconds = Time.realtimeSinceStartupAsDouble - prepStart;

            for (int batchStart = 0; batchStart < cores.Count; batchStart += batchSize)
            {
                int count = Math.Min(batchSize, cores.Count - batchStart);
                var current = cores.GetRange(batchStart, count);
                float[] features = BuildBatch(current, voxels, index, grid);
                using var tensor = new Tensor<float>(new TensorShape(count, Channels, Patch, Patch, Patch), features);
                double scheduleStart = Time.realtimeSinceStartupAsDouble;
                _worker.Schedule(tensor);
                result.scheduleSeconds += Time.realtimeSinceStartupAsDouble - scheduleStart;
                var poleTensor = _worker.PeekOutput(PoleOutputName) as Tensor<float>;
                var lineTensor = _worker.PeekOutput(LineOutputName) as Tensor<float>;
                if (poleTensor == null || lineTensor == null)
                    throw new InvalidOperationException("Expected pole_score and line_score outputs.");
                double readStart = Time.realtimeSinceStartupAsDouble;
                using var poleCpu = await poleTensor.ReadbackAndCloneAsync();
                using var lineCpu = await lineTensor.ReadbackAndCloneAsync();
                float[] pole = poleCpu.DownloadToArray();
                float[] line = lineCpu.DownloadToArray();
                result.readbackSeconds += Time.realtimeSinceStartupAsDouble - readStart;
                if (pole.Length != count * CoreVolume || line.Length != count * CoreVolume)
                    throw new InvalidOperationException("Unexpected Stage-1 ONNX output length.");
                for (int b = 0; b < count; b++)
                {
                    CoreGroup group = current[b];
                    Key core = group.key;
                    int ox = core.x * Core, oy = core.y * Core, oz = core.z * Core;
                    foreach (int row in group.rows)
                    {
                        SparseVoxel v = voxels[row].voxel;
                        int local = ((v.z - oz) * Core + (v.y - oy)) * Core + (v.x - ox);
                        result.poleScore[row] = pole[b * CoreVolume + local];
                        result.lineScore[row] = line[b * CoreVolume + local];
                    }
                }
            }
            for (int i = 0; i < voxels.Length; i++)
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
        bool p = pole >= _config.calibration.pole_threshold;
        bool l = line >= _config.calibration.line_threshold;
        if (p && (!l || pole >= line)) return 1;
        if (l) return 2;
        return 0;
    }

    private static PreparedVoxel[] Prepare(IReadOnlyList<SparseVoxel> source, int[] grid)
    {
        var last = new Dictionary<Key, PreparedVoxel>();
        float maxDistance = 1f;
        for (int i = 0; i < source.Count; i++)
        {
            SparseVoxel v = source[i];
            if (v.x < 0 || v.x >= grid[0] || v.y < 0 || v.y >= grid[1] || v.z < 0 || v.z >= grid[2])
                continue;
            if (float.IsNaN(v.distCenterFt) || float.IsInfinity(v.distCenterFt)) v.distCenterFt = 0f;
            maxDistance = Math.Max(maxDistance, Math.Abs(v.distCenterFt));
            last[new Key(v.x, v.y, v.z)] = new PreparedVoxel { voxel = v, stableOrder = i };
        }
        PreparedVoxel[] output = last.Values.OrderBy(v => v.voxel.z).ThenBy(v => v.stableOrder).ToArray();
        foreach (PreparedVoxel item in output) item.dist = item.voxel.distCenterFt / maxDistance;
        return output;
    }

    private static List<CoreGroup> ActiveCores(PreparedVoxel[] voxels)
    {
        var groups = new Dictionary<Key, CoreGroup>();
        for (int row = 0; row < voxels.Length; row++)
        {
            PreparedVoxel item = voxels[row];
            var key = new Key(item.voxel.x / Core, item.voxel.y / Core, item.voxel.z / Core);
            if (!groups.TryGetValue(key, out CoreGroup group))
            {
                group = new CoreGroup { key = key };
                groups.Add(key, group);
            }
            group.rows.Add(row);
        }
        var output = groups.Values.ToList();
        output.Sort((a, b) => a.key.CompareTo(b.key));
        return output;
    }

    private static float[] BuildBatch(List<CoreGroup> cores, PreparedVoxel[] voxels,
        Dictionary<Key, int> lookup, int[] grid)
    {
        int patchVolume = Patch * Patch * Patch;
        float[] output = new float[cores.Count * Channels * patchVolume];
        for (int b = 0; b < cores.Count; b++)
        {
            int cx = cores[b].key.x * Core + Core / 2;
            int cy = cores[b].key.y * Core + Core / 2;
            int cz = cores[b].key.z * Core + Core / 2;
            int x0 = cx - Patch / 2, y0 = cy - Patch / 2, z0 = cz - Patch / 2;
            int batchBase = b * Channels * patchVolume;
            for (int z = 0; z < Patch; z++)
            for (int y = 0; y < Patch; y++)
            for (int x = 0; x < Patch; x++)
            {
                int flat = (z * Patch + y) * Patch + x;
                int gx = x0 + x, gy = y0 + y, gz = z0 + z;
                if (lookup.TryGetValue(new Key(gx, gy, gz), out int row))
                {
                    output[batchBase + flat] = 1f;
                    output[batchBase + 4 * patchVolume + flat] = voxels[row].dist;
                }
                output[batchBase + patchVolume + flat] = NormalizeCoord(gx, grid[0]);
                output[batchBase + 2 * patchVolume + flat] = NormalizeCoord(gy, grid[1]);
                output[batchBase + 3 * patchVolume + flat] = NormalizeCoord(gz, grid[2]);
            }
        }
        return output;
    }

    private static float NormalizeCoord(int value, int size)
        => Mathf.Clamp((value / (float)Math.Max(size - 1, 1)) * 2f - 1f, -1.5f, 1.5f);

    /// <summary>Writes the portable handoff consumed by run_unity_stage1_csv_stage2.py.</summary>
    public static void WriteStage1Csv(string path, Stage1Result result)
    {
        if (result == null || !result.success) throw new InvalidOperationException("Cannot write failed inference.");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
        string temporary = path + ".tmp";
        using (var writer = new StreamWriter(temporary, false, new UTF8Encoding(false)))
        {
            writer.WriteLine("source_row,x,y,z,dist_values,pole,line,deployed_label");
            for (int i = 0; i < result.voxels.Length; i++)
            {
                SparseVoxel v = result.voxels[i];
                writer.Write(v.sourceRow.ToString(CultureInfo.InvariantCulture)); writer.Write(',');
                writer.Write(v.x); writer.Write(','); writer.Write(v.y); writer.Write(','); writer.Write(v.z); writer.Write(',');
                writer.Write(result.normalizedDistance[i].ToString("R", CultureInfo.InvariantCulture)); writer.Write(',');
                writer.Write(result.poleScore[i].ToString("R", CultureInfo.InvariantCulture)); writer.Write(',');
                writer.Write(result.lineScore[i].ToString("R", CultureInfo.InvariantCulture)); writer.Write(',');
                writer.WriteLine(result.deployedLabel[i]);
            }
        }
        if (File.Exists(path)) File.Delete(path);
        File.Move(temporary, path);
    }

    private void OnDestroy()
    {
        _worker?.Dispose();
        _inferenceLock.Dispose();
    }
}

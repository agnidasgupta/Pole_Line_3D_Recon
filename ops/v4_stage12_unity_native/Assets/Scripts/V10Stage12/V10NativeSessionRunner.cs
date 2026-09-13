using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace VegetationAssurance.V10
{
    /// <summary>Fault-tolerant standalone Stage1 Sentis + native Stage2 runner.</summary>
    public sealed class V10NativeSessionRunner : MonoBehaviour
    {
        private sealed class ManifestRow
        {
            public string groupId, id, relativePath, sourceCsv, source, geography, session, manifestDirectory;
            public int sliceSeq;
            public double centerX, centerY, centerZ;
        }

        private sealed class SessionTiming
        {
            public int slices;
            public int poles, lines, lineVoxels;
            public double stage1Prepare, stage1Schedule, stage1Readback, stage1Total;
            public double stage2Components, stage2Refiner, lineComponents, lineTrace, lineAudit, lineGeometry, stage2Total, wall;
        }

        [SerializeField] private V10Stage12SentisInferenceManager inferenceManager;
        [SerializeField] private TextAsset stage2TreeBinary;
        [SerializeField] private bool runWhenCommandLinePresent = true;

        private Timer heartbeat;
        private readonly object heartbeatLock = new object();
        private string heartbeatPath;
        private volatile string heartbeatState = "starting";

        private async void Start()
        {
            if (!runWhenCommandLinePresent || !HasArgument("--v10-manifest")) return;
            int code = 1;
            try { code = await RunFromCommandLineAsync(); }
            catch (Exception exception)
            {
                UnityEngine.Debug.LogError("[V10Stage12] fatal: " + exception);
                TryWriteFatal(exception);
            }
            finally
            {
                heartbeat?.Dispose();
                heartbeat = null;
                Application.Quit(code);
            }
        }

        public async Task<int> RunFromCommandLineAsync()
        {
            Dictionary<string, string> options = Options(Environment.GetCommandLineArgs());
            string manifestPath = Required(options, "--v10-manifest");
            string runRoot = Path.GetFullPath(Required(options, "--v10-run-root"));
            bool resume = !options.TryGetValue("--v10-resume", out string resumeText) || resumeText != "0";
            int expectedSessions = options.TryGetValue("--v10-expected-sessions", out string countText)
                ? int.Parse(countText, CultureInfo.InvariantCulture) : 0;
            if (inferenceManager == null || stage2TreeBinary == null)
                throw new InvalidOperationException("Inference manager and Stage2 tree binary are required in the server scene.");

            Directory.CreateDirectory(runRoot);
            Directory.CreateDirectory(Path.Combine(runRoot, "status"));
            Directory.CreateDirectory(Path.Combine(runRoot, "timing"));
            heartbeatPath = Path.Combine(runRoot, "RUN_HEARTBEAT.txt");
            heartbeat = new Timer(_ => WriteHeartbeat(), null, TimeSpan.Zero, TimeSpan.FromSeconds(15));
            V10NativeOutputWriter.AtomicText(Path.Combine(runRoot, "RUNNING.txt"),
                "pid=" + Process.GetCurrentProcess().Id + "\nstarted_utc=" + DateTime.UtcNow.ToString("O") + "\n");

            List<ManifestRow> rows = ReadManifest(manifestPath);
            string[] sessions = rows.Select(v => v.groupId).Distinct(StringComparer.Ordinal).ToArray();
            if (expectedSessions > 0 && sessions.Length != expectedSessions)
                throw new InvalidDataException("Session count mismatch: expected=" + expectedSessions + " actual=" + sessions.Length);

            await inferenceManager.InitializeAsync(true);
            var trees = V10ExtraTreesBundle.Load(stage2TreeBinary.bytes,
                inferenceManager.Configuration.stage2.refiner_tree_metadata.sha256);
            var reconstructor = new V10NativeStage2Reconstructor(trees, inferenceManager.Configuration);
            string timingCsv = Path.Combine(runRoot, "timing", "STAGE12_TIMING_PER_SLICE.csv");
            EnsureTimingHeader(timingCsv);
            bool anyFailure = false;
            int ordinal = 0;

            foreach (IGrouping<string, ManifestRow> group in rows.GroupBy(v => v.groupId, StringComparer.Ordinal))
            {
                string sid = SafeSessionId(group.Key);
                int failed = 0, completed = 0;
                var sessionTiming = new SessionTiming();
                foreach (ManifestRow row in group.OrderBy(v => v.sliceSeq))
                {
                    ordinal++;
                    heartbeatState = "session=" + group.Key + " slice_seq=" + row.sliceSeq + " item=" + ordinal + "/" + rows.Count;
                    string objectDirectory = V10NativeOutputWriter.ObjectDirectory(runRoot, sid, row.relativePath);
                    string stem = Stem(row.relativePath);
                    if (resume && SliceReady(objectDirectory, stem))
                    {
                        completed++;
                        UnityEngine.Debug.Log("[V10Stage12] reuse " + heartbeatState);
                        continue;
                    }

                    Stopwatch wall = Stopwatch.StartNew();
                    try
                    {
                        string input = ResolveInput(row.sourceCsv, row.manifestDirectory);
                        List<SparseVoxel> voxels = V10SparseVoxelCsvReader.Read(input);
                        Stage1SliceResult stage1 = await inferenceManager.InferSliceAsync(voxels);
                        if (!stage1.success) throw new InvalidOperationException(stage1.error);
                        string stage1Directory = Stage1Directory(runRoot, sid, row.relativePath);
                        Directory.CreateDirectory(stage1Directory);
                        V10Stage12SentisInferenceManager.WriteStage1Csv(
                            Path.Combine(stage1Directory, stem + "_stage1.csv"), stage1);

                        Stage2SliceResult stage2 = reconstructor.Reconstruct(stage1, row.id, row.sliceSeq);
                        if (!stage2.success) throw new InvalidOperationException(stage2.error);
                        var metadata = new Dictionary<string, object> { ["group_id"] = row.groupId };
                        V10NativeOutputWriter.WriteSlice(runRoot, sid, row.relativePath, stage1, stage2, metadata);
                        wall.Stop();
                        AppendTiming(timingCsv, row, sid, stage1, stage2, wall.Elapsed.TotalMilliseconds);
                        AddTiming(sessionTiming, stage1, stage2, wall.Elapsed.TotalMilliseconds);
                        AppendInferenceManifest(runRoot, row, sid, stage2);
                        string oldFailure = Path.Combine(runRoot, "status", sid + ".slice_" + row.sliceSeq + ".failed");
                        if (File.Exists(oldFailure)) File.Delete(oldFailure);
                        completed++;
                        UnityEngine.Debug.Log("[V10Stage12] completed " + heartbeatState);
                    }
                    catch (Exception exception)
                    {
                        failed++;
                        anyFailure = true;
                        string failure = Path.Combine(runRoot, "status", sid + ".slice_" + row.sliceSeq + ".failed");
                        V10NativeOutputWriter.AtomicText(failure, exception + "\n");
                        UnityEngine.Debug.LogError("[V10Stage12] rejected " + heartbeatState + "\n" + exception);
                    }
                }

                string statusSuffix = failed == 0 ? ".stage2.ok" : ".stage2.failed";
                string opposite = Path.Combine(runRoot, "status", sid + (failed == 0 ? ".stage2.failed" : ".stage2.ok"));
                if (File.Exists(opposite)) File.Delete(opposite);
                V10NativeOutputWriter.AtomicText(Path.Combine(runRoot, "status", sid + statusSuffix),
                    "group_id=" + group.Key + "\ncompleted=" + completed + "\nfailed=" + failed + "\nutc=" + DateTime.UtcNow.ToString("O") + "\n");
                string stage2SessionRoot = Path.Combine(runRoot, "stage2", sid);
                Directory.CreateDirectory(stage2SessionRoot);
                var completion = new Dictionary<string, object>
                {
                    ["group_id"] = group.Key, ["completed_slices"] = completed, ["failed_slices"] = failed,
                    ["status"] = failed == 0 ? "completed" : "failed", ["completed_utc"] = DateTime.UtcNow.ToString("O")
                };
                V10NativeOutputWriter.AtomicText(Path.Combine(stage2SessionRoot, "STAGE2_COMPLETED.json"),
                    V10Json.Serialize(completion, true));
                var summary = new Dictionary<string, object>
                {
                    ["group_id"] = group.Key, ["slices"] = completed,
                    ["accepted_poles"] = CountRows(stage2SessionRoot, "*_poles.csv"),
                    ["accepted_line_segments"] = CountRows(stage2SessionRoot, "*_line_segments.csv"),
                    ["accepted_stage1_line_voxels"] = CountRows(stage2SessionRoot, "*_accepted_line_voxels.csv"),
                    ["stage1_to_stage2_voxel_preservation"] = 1.0,
                    ["geometry_stage1_voxel_support_fraction"] = 1.0,
                    ["disconnected_fragment_bridges_allowed"] = false,
                    ["runtime_gt_usage"] = false, ["synthetic_line_voxels"] = 0
                };
                V10NativeOutputWriter.AtomicText(Path.Combine(stage2SessionRoot,
                    "STAGE2_STAGE1_ELECTRICAL_TRACK_SUMMARY.json"), V10Json.Serialize(summary, true));
                WriteSessionTiming(runRoot, group.Key, sessionTiming);
            }

            heartbeatState = anyFailure ? "finished_with_failures" : "complete";
            WriteAllSessionTiming(runRoot);
            if (options.TryGetValue("--v10-reference-root", out string referenceRoot) &&
                !string.IsNullOrWhiteSpace(referenceRoot))
            {
                heartbeatState = "automated_equivalence_gate";
                bool equivalent = V10NativeEquivalenceValidator.Validate(
                    runRoot, referenceRoot, Path.Combine(runRoot, "UNITY_NATIVE_EQUIVALENCE.txt"));
                if (!equivalent) anyFailure = true;
            }
            heartbeatState = anyFailure ? "finished_with_failures" : "complete";
            string running = Path.Combine(runRoot, "RUNNING.txt");
            if (File.Exists(running)) File.Delete(running);
            string marker = anyFailure ? "STAGE12_FAILED.txt" : "STAGE12_COMPLETE.txt";
            string oppositeMarker = Path.Combine(runRoot, anyFailure ? "STAGE12_COMPLETE.txt" : "STAGE12_FAILED.txt");
            if (File.Exists(oppositeMarker)) File.Delete(oppositeMarker);
            V10NativeOutputWriter.AtomicText(Path.Combine(runRoot, marker),
                "sessions=" + sessions.Length + "\nslices=" + rows.Count + "\ncompleted_utc=" + DateTime.UtcNow.ToString("O") + "\n");
            if (!anyFailure)
            {
                V10NativeOutputWriter.AtomicText(Path.Combine(runRoot, "PHASE2_STAGE2_OK.txt"), "UNITY_NATIVE_STAGE2_OK\n");
                V10NativeOutputWriter.AtomicText(Path.Combine(runRoot, "STAGE2_ONLY_COMPLETE.txt"), "UNITY_NATIVE_STAGE2_COMPLETE\n");
            }
            else
            {
                foreach (string stale in new[] { "PHASE2_STAGE2_OK.txt", "STAGE2_ONLY_COMPLETE.txt" })
                {
                    string path = Path.Combine(runRoot, stale);
                    if (File.Exists(path)) File.Delete(path);
                }
            }
            WriteHeartbeat();
            return anyFailure ? 2 : 0;
        }

        private static List<ManifestRow> ReadManifest(string path)
        {
            string full = Path.GetFullPath(path);
            if (Directory.Exists(full))
            {
                string[] manifests = Directory.GetFiles(full, "stage1_manifest.csv", SearchOption.AllDirectories)
                    .OrderBy(v => v, StringComparer.Ordinal).ToArray();
                if (manifests.Length == 0) throw new FileNotFoundException("No stage1_manifest.csv files under " + full);
                return manifests.SelectMany(ReadManifestFile).OrderBy(v => v.groupId, StringComparer.Ordinal)
                    .ThenBy(v => v.sliceSeq).ToList();
            }
            return ReadManifestFile(full);
        }

        private static List<ManifestRow> ReadManifestFile(string path)
        {
            using var reader = new StreamReader(path, Encoding.UTF8, true);
            string first = reader.ReadLine();
            if (first == null) throw new InvalidDataException("Manifest is empty: " + path);
            Dictionary<string, int> header = V10Csv.Header(V10Csv.ParseLine(first));
            foreach (string name in new[] { "group_id", "slice_seq", "relative_path" })
                if (!header.ContainsKey(name)) throw new InvalidDataException("Manifest lacks " + name);
            if (!header.ContainsKey("source_csv") && !header.ContainsKey("source"))
                throw new InvalidDataException("Manifest needs source_csv or source: " + path);
            var output = new List<ManifestRow>();
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                List<string> row = V10Csv.ParseLine(line);
                string relative = V10Csv.At(row, header, "relative_path") ?? "";
                string source = V10Csv.At(row, header, "source") ?? "";
                string sourceCsv = V10Csv.At(row, header, "source_csv") ?? source;
                output.Add(new ManifestRow
                {
                    groupId = V10Csv.At(row, header, "group_id") ?? "",
                    sliceSeq = V10Csv.RequiredInt(row, header, "slice_seq"),
                    relativePath = relative,
                    sourceCsv = sourceCsv,
                    id = V10Csv.At(row, header, "id") ?? Stem(relative),
                    source = source,
                    geography = V10Csv.At(row, header, "geography") ?? "",
                    session = V10Csv.At(row, header, "session") ?? "",
                    centerX = OptionalDouble(row, header, "center_x"),
                    centerY = OptionalDouble(row, header, "center_y"),
                    centerZ = OptionalDouble(row, header, "center_z"),
                    manifestDirectory = Path.GetDirectoryName(path) ?? "."
                });
            }
            if (output.Count == 0 || output.Any(v => v.groupId.Length == 0 || v.relativePath.Length == 0 || v.sourceCsv.Length == 0))
                throw new InvalidDataException("Manifest contains no usable rows or blank required fields.");
            return output.OrderBy(v => v.groupId, StringComparer.Ordinal).ThenBy(v => v.sliceSeq).ToList();
        }

        private static double OptionalDouble(IReadOnlyList<string> row, Dictionary<string, int> header, string name)
        {
            string text = V10Csv.At(row, header, name);
            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ? value : 0.0;
        }

        private static int CountRows(string root, string pattern)
        {
            if (!Directory.Exists(root)) return 0;
            int total = 0;
            foreach (string path in Directory.GetFiles(root, pattern, SearchOption.AllDirectories))
                total += Math.Max(0, File.ReadLines(path).Count() - 1);
            return total;
        }

        private static bool SliceReady(string directory, string stem)
        {
            string[] suffixes = {
                "_poles.csv", "_line_segments.csv", "_line_vertices.csv", "_components.csv",
                "_stage1_electrical_tracks.csv", "_pole_attachments.csv", "_selected_fragment_bridges.csv",
                "_fragment_bridge_candidates.csv", "_stage1_line_voxels.csv", "_accepted_line_voxels.csv",
                "_stage1_electrical_track_audit.json"
            };
            return suffixes.All(suffix => File.Exists(Path.Combine(directory, stem + suffix)));
        }

        private static string ResolveInput(string path, string baseDirectory) =>
            Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(baseDirectory, path));
        private static string Stem(string relativePath)
        {
            string name = Path.GetFileName(relativePath ?? "slice.csv");
            return name.EndsWith(".csv.gz", StringComparison.OrdinalIgnoreCase)
                ? name.Substring(0, name.Length - 7) : Path.GetFileNameWithoutExtension(name);
        }
        private static string Stage1Directory(string root, string sid, string relativePath)
        {
            string parent = Path.GetDirectoryName((relativePath ?? "").Replace('\\', '/')) ?? "";
            return Path.Combine(root, "stage1", sid, "stage1_scores", parent);
        }
        public static string SafeSessionId(string groupId)
        {
            var output = new StringBuilder(); bool separator = false;
            foreach (char c in groupId ?? "")
            {
                bool keep = char.IsLetterOrDigit(c) || c == '_' || c == '.' || c == '-';
                if (keep) { output.Append(c); separator = false; }
                else if (!separator) { output.Append("__"); separator = true; }
            }
            return output.Length == 0 ? "session" : output.ToString();
        }

        private static void EnsureTimingHeader(string path)
        {
            if (File.Exists(path)) return;
            V10NativeOutputWriter.AtomicText(path,
                "group_id,sid,slice_seq,id,stage1_prepare_ms,stage1_schedule_ms,stage1_readback_ms,stage1_total_ms," +
                "production_component_ms,production_refiner_parametric_ms,line_connected_components_ms,line_trace_ms," +
                "line_assignment_audit_ms,line_geometry_ms,stage2_total_ms,slice_wall_ms\n");
        }

        private static readonly object AppendLock = new object();
        private static void AppendTiming(string path, ManifestRow row, string sid, Stage1SliceResult a, Stage2SliceResult b, double wall)
        {
            string[] values = { row.groupId, sid, row.sliceSeq.ToString(CultureInfo.InvariantCulture), row.id,
                (a.prepareSeconds * 1000).ToString("R", CultureInfo.InvariantCulture),
                (a.scheduleSeconds * 1000).ToString("R", CultureInfo.InvariantCulture),
                (a.readbackSeconds * 1000).ToString("R", CultureInfo.InvariantCulture),
                (a.totalSeconds * 1000).ToString("R", CultureInfo.InvariantCulture),
                b.timing.production_component_ms.ToString("R", CultureInfo.InvariantCulture),
                b.timing.production_refiner_parametric_ms.ToString("R", CultureInfo.InvariantCulture),
                b.timing.line_connected_components_ms.ToString("R", CultureInfo.InvariantCulture),
                b.timing.line_trace_ms.ToString("R", CultureInfo.InvariantCulture),
                b.timing.line_assignment_audit_ms.ToString("R", CultureInfo.InvariantCulture),
                b.timing.line_geometry_ms.ToString("R", CultureInfo.InvariantCulture),
                b.timing.stage2_total_ms.ToString("R", CultureInfo.InvariantCulture), wall.ToString("R", CultureInfo.InvariantCulture) };
            string replacement = string.Join(",", values.Select(value => V10Csv.Escape(value)));
            lock (AppendLock)
            {
                string[] lines = File.ReadAllLines(path);
                var output = new List<string> { lines[0] };
                Dictionary<string, int> header = V10Csv.Header(V10Csv.ParseLine(lines[0]));
                foreach (string existing in lines.Skip(1))
                {
                    if (string.IsNullOrWhiteSpace(existing)) continue;
                    List<string> parsed = V10Csv.ParseLine(existing);
                    string gid = V10Csv.At(parsed, header, "group_id") ?? "";
                    string seq = V10Csv.At(parsed, header, "slice_seq") ?? "";
                    if (gid == row.groupId && seq == row.sliceSeq.ToString(CultureInfo.InvariantCulture)) continue;
                    output.Add(existing);
                }
                output.Add(replacement);
                V10NativeOutputWriter.AtomicText(path, string.Join("\n", output) + "\n");
            }
        }

        private static void AddTiming(SessionTiming s, Stage1SliceResult a, Stage2SliceResult b, double wall)
        {
            s.slices++; s.stage1Prepare += a.prepareSeconds * 1000; s.stage1Schedule += a.scheduleSeconds * 1000;
            s.poles += b.poles.Count; s.lines += b.lines.Count; s.lineVoxels += b.acceptedLineRows.Count;
            s.stage1Readback += a.readbackSeconds * 1000; s.stage1Total += a.totalSeconds * 1000;
            s.stage2Components += b.timing.production_component_ms; s.stage2Refiner += b.timing.production_refiner_parametric_ms;
            s.lineComponents += b.timing.line_connected_components_ms; s.lineTrace += b.timing.line_trace_ms;
            s.lineAudit += b.timing.line_assignment_audit_ms; s.lineGeometry += b.timing.line_geometry_ms;
            s.stage2Total += b.timing.stage2_total_ms; s.wall += wall;
        }

        private static void WriteSessionTiming(string root, string groupId, SessionTiming s)
        {
            double n = Math.Max(1, s.slices);
            string text = "group_id=" + groupId + "\ncompleted_slices=" + s.slices + "\n" +
                "average_stage1_prepare_ms=" + (s.stage1Prepare / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_stage1_schedule_ms=" + (s.stage1Schedule / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_stage1_readback_ms=" + (s.stage1Readback / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_stage1_total_ms=" + (s.stage1Total / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_production_component_ms=" + (s.stage2Components / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_production_refiner_parametric_ms=" + (s.stage2Refiner / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_line_connected_components_ms=" + (s.lineComponents / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_line_trace_ms=" + (s.lineTrace / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_line_assignment_audit_ms=" + (s.lineAudit / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_line_geometry_ms=" + (s.lineGeometry / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_stage2_total_ms=" + (s.stage2Total / n).ToString("R", CultureInfo.InvariantCulture) + "\n" +
                "average_slice_wall_ms=" + (s.wall / n).ToString("R", CultureInfo.InvariantCulture) + "\n";
            V10NativeOutputWriter.AtomicText(Path.Combine(root, "timing", SafeSessionId(groupId) + ".timing.txt"), text);
        }

        private static void WriteAllSessionTiming(string root)
        {
            string directory = Path.Combine(root, "timing");
            string csv = Path.Combine(directory, "STAGE12_TIMING_PER_SLICE.csv");
            var aggregate = new Dictionary<string, SessionTiming>(StringComparer.Ordinal);
            using (var reader = new StreamReader(csv, Encoding.UTF8, true))
            {
                string first = reader.ReadLine();
                if (first == null) throw new InvalidDataException("Timing CSV has no header.");
                Dictionary<string, int> header = V10Csv.Header(V10Csv.ParseLine(first));
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    List<string> row = V10Csv.ParseLine(line);
                    string gid = V10Csv.At(row, header, "group_id") ?? "";
                    if (!aggregate.TryGetValue(gid, out SessionTiming s))
                    { s = new SessionTiming(); aggregate.Add(gid, s); }
                    s.slices++;
                    s.stage1Prepare += TimingValue(row, header, "stage1_prepare_ms");
                    s.stage1Schedule += TimingValue(row, header, "stage1_schedule_ms");
                    s.stage1Readback += TimingValue(row, header, "stage1_readback_ms");
                    s.stage1Total += TimingValue(row, header, "stage1_total_ms");
                    s.stage2Components += TimingValue(row, header, "production_component_ms");
                    s.stage2Refiner += TimingValue(row, header, "production_refiner_parametric_ms");
                    s.lineComponents += TimingValue(row, header, "line_connected_components_ms");
                    s.lineTrace += TimingValue(row, header, "line_trace_ms");
                    s.lineAudit += TimingValue(row, header, "line_assignment_audit_ms");
                    s.lineGeometry += TimingValue(row, header, "line_geometry_ms");
                    s.stage2Total += TimingValue(row, header, "stage2_total_ms");
                    s.wall += TimingValue(row, header, "slice_wall_ms");
                }
            }
            foreach (KeyValuePair<string, SessionTiming> item in aggregate)
                WriteSessionTiming(root, item.Key, item.Value);
            var output = new StringBuilder("V10 UNITY NATIVE STAGE1+STAGE2 SESSION AVERAGES\n\n");
            foreach (string path in Directory.GetFiles(directory, "*.timing.txt").OrderBy(v => v, StringComparer.Ordinal))
                output.Append(File.ReadAllText(path)).Append('\n');
            V10NativeOutputWriter.AtomicText(Path.Combine(root, "STAGE12_TIMING_SESSION_AVERAGES.txt"), output.ToString());
        }

        private static double TimingValue(IReadOnlyList<string> row, Dictionary<string, int> header, string name)
        {
            string text = V10Csv.At(row, header, name);
            if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
                throw new InvalidDataException("Invalid timing value " + name + "=" + text);
            return value;
        }

        private static void AppendInferenceManifest(string root, ManifestRow row, string sid, Stage2SliceResult result)
        {
            string sessionRoot = Path.Combine(root, "stage2", sid);
            Directory.CreateDirectory(sessionRoot);
            string path = Path.Combine(sessionRoot, "inference_manifest.csv");
            lock (AppendLock)
            {
                string headerLine = "contract_version,id,source,relative_path,geography,session,slice_seq,group_id,center_x,center_y,center_z,accepted_poles,accepted_line_segments,status";
                object[] values = { "v4-stage12-unity-native-1", row.id, row.source, row.relativePath, row.geography, row.session,
                    row.sliceSeq, row.groupId, row.centerX, row.centerY, row.centerZ, result.poles.Count, result.lines.Count, "completed" };
                var output = new List<string> { headerLine };
                if (File.Exists(path))
                {
                    string[] lines = File.ReadAllLines(path);
                    Dictionary<string, int> header = V10Csv.Header(V10Csv.ParseLine(lines[0]));
                    foreach (string existing in lines.Skip(1))
                    {
                        if (string.IsNullOrWhiteSpace(existing)) continue;
                        List<string> parsed = V10Csv.ParseLine(existing);
                        if ((V10Csv.At(parsed, header, "group_id") ?? "") == row.groupId &&
                            (V10Csv.At(parsed, header, "slice_seq") ?? "") == row.sliceSeq.ToString(CultureInfo.InvariantCulture)) continue;
                        output.Add(existing);
                    }
                }
                output.Add(string.Join(",", values.Select(value => V10Csv.Escape(value))));
                V10NativeOutputWriter.AtomicText(path, string.Join("\n", output) + "\n");
                V10NativeOutputWriter.AtomicText(Path.Combine(sessionRoot, "stage2_manifest.csv"),
                    string.Join("\n", output) + "\n");
            }
        }

        private void WriteHeartbeat()
        {
            try
            {
                if (string.IsNullOrEmpty(heartbeatPath)) return;
                lock (heartbeatLock)
                    V10NativeOutputWriter.AtomicText(heartbeatPath,
                        "utc=" + DateTime.UtcNow.ToString("O") + "\npid=" + Process.GetCurrentProcess().Id + "\nstate=" + heartbeatState + "\n");
            }
            catch { }
        }
        private void TryWriteFatal(Exception exception)
        {
            try
            {
                if (!string.IsNullOrEmpty(heartbeatPath))
                    V10NativeOutputWriter.AtomicText(Path.Combine(Path.GetDirectoryName(heartbeatPath) ?? ".", "FATAL_ERROR.txt"), exception + "\n");
            }
            catch { }
        }
        private static bool HasArgument(string key) => Environment.GetCommandLineArgs().Contains(key);
        private static string Required(Dictionary<string, string> values, string key) =>
            values.TryGetValue(key, out string value) && !string.IsNullOrWhiteSpace(value)
                ? value : throw new ArgumentException("Missing command-line option " + key);
        private static Dictionary<string, string> Options(string[] args)
        {
            var output = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < args.Length; i++)
                if (args[i].StartsWith("--v10-", StringComparison.Ordinal))
                    output[args[i]] = i + 1 < args.Length && !args[i + 1].StartsWith("--v10-", StringComparison.Ordinal) ? args[++i] : "1";
            return output;
        }
    }
}

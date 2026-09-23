using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace VegetationAssurance.V10
{
    /// <summary>
    /// Native C# port of the active V10 opt1-fix2 Stage-2 path. It preserves
    /// production pole extraction/refinement and replaces production line output
    /// with graph-connected, Stage-1-voxel-supported open tracks.
    /// </summary>
    public sealed class V10NativeStage2Reconstructor
    {
        private readonly V10ExtraTreesBundle trees;
        private readonly V10Stage2Profile profile;
        private readonly int[] grid;
        private readonly double voxelSizeFt;

        private static readonly Int3Key[] Neighbor26 = BuildNeighborOffsets(false);
        private static readonly Int3Key[] Forward26 = BuildNeighborOffsets(true);

        private sealed class PoleRecord
        {
            public string id;
            public double baseX, baseY, baseZ;
            public double topX, topY, topZ;
            public double radiusP90Ft;
        }

        private sealed class ComponentGeometry
        {
            public int[] rows;
            public Dictionary<string, double> features;
            public D3[] points;
            public D3 principal;
            public D3 endpoint1;
            public D3 endpoint2;
        }

        private readonly struct Edge
        {
            public readonly int node;
            public readonly double weight;
            public Edge(int node, double weight) { this.node = node; this.weight = weight; }
        }

        private sealed class Trace
        {
            public int sourceComponent;
            public int[] voxelRows;
            public int[] pathRows;
        }

        private readonly struct IndexedPoleVoxel
        {
            public readonly int order;
            public readonly D3 point;
            public IndexedPoleVoxel(int order, D3 point) { this.order = order; this.point = point; }
        }

        private readonly struct HeapEntry
        {
            public readonly double distance;
            public readonly int node;
            public HeapEntry(double distance, int node) { this.distance = distance; this.node = node; }
        }

        private sealed class MinHeap
        {
            private readonly List<HeapEntry> data = new List<HeapEntry>();
            public int Count => data.Count;
            private static bool Less(HeapEntry a, HeapEntry b) =>
                a.distance < b.distance || (a.distance == b.distance && a.node < b.node);
            public void Push(HeapEntry value)
            {
                data.Add(value);
                int i = data.Count - 1;
                while (i > 0)
                {
                    int p = (i - 1) / 2;
                    if (!Less(data[i], data[p])) break;
                    (data[i], data[p]) = (data[p], data[i]);
                    i = p;
                }
            }
            public HeapEntry Pop()
            {
                HeapEntry result = data[0];
                HeapEntry last = data[data.Count - 1];
                data.RemoveAt(data.Count - 1);
                if (data.Count == 0) return result;
                data[0] = last;
                int i = 0;
                while (true)
                {
                    int left = i * 2 + 1;
                    if (left >= data.Count) break;
                    int right = left + 1;
                    int child = right < data.Count && Less(data[right], data[left]) ? right : left;
                    if (!Less(data[child], data[i])) break;
                    (data[i], data[child]) = (data[child], data[i]);
                    i = child;
                }
                return result;
            }
        }

        private sealed class UnionFind
        {
            private readonly int[] parent;
            private readonly byte[] rank;
            public UnionFind(int count)
            {
                parent = new int[count];
                rank = new byte[count];
                for (int i = 0; i < count; i++) parent[i] = i;
            }
            public int Find(int value)
            {
                int current = value;
                while (parent[current] != current)
                {
                    parent[current] = parent[parent[current]];
                    current = parent[current];
                }
                return current;
            }
            public void Union(int left, int right)
            {
                left = Find(left);
                right = Find(right);
                if (left == right) return;
                if (rank[left] < rank[right]) (left, right) = (right, left);
                parent[right] = left;
                if (rank[left] == rank[right]) rank[left]++;
            }
        }

        public V10NativeStage2Reconstructor(V10ExtraTreesBundle trees, V10Sidecar sidecar,
            double voxelSizeFt = 0.5)
        {
            this.trees = trees ?? throw new ArgumentNullException(nameof(trees));
            if (sidecar == null || sidecar.stage2 == null || sidecar.stage2.profile == null)
                throw new ArgumentException("Sidecar Stage-2 profile is missing.", nameof(sidecar));
            profile = sidecar.stage2.profile;
            grid = sidecar.input.grid_size_xyz;
            this.voxelSizeFt = voxelSizeFt;
            if (grid == null || grid.Length != 3 || grid.Any(v => v <= 0))
                throw new ArgumentException("Sidecar grid is invalid.", nameof(sidecar));
            if (profile.disconnected_fragment_bridge_allowed ||
                !profile.line_geometry_must_stay_inside_stage1_voxel_cells ||
                !profile.open_line_endpoints_preserved ||
                !profile.pole_attachment_requires_stage1_voxel_contact)
                throw new InvalidOperationException("Sidecar violates the accepted strict V10 Stage-2 contract.");
        }

        public Stage2SliceResult Reconstruct(Stage1SliceResult stage1, string fileId, int sliceSeq)
        {
            var result = new Stage2SliceResult { fileId = fileId, sliceSeq = sliceSeq };
            var all = Stopwatch.StartNew();
            try
            {
                ValidateStage1(stage1);
                List<PoleRecord> poles = ExtractProductionPoles(stage1, result);
                BuildStrictLines(stage1, poles, result);
                result.timing.stage2_total_ms = all.Elapsed.TotalMilliseconds;
                result.success = true;
            }
            catch (Exception exception)
            {
                result.success = false;
                result.error = exception.ToString();
            }
            return result;
        }

        private static void ValidateStage1(Stage1SliceResult stage1)
        {
            if (stage1 == null || !stage1.success) throw new InvalidOperationException("Stage 1 did not succeed.");
            int count = stage1.voxels?.Length ?? -1;
            if (count < 0 || stage1.poleScore == null || stage1.lineScore == null || stage1.deployedLabel == null ||
                stage1.poleScore.Length != count || stage1.lineScore.Length != count || stage1.deployedLabel.Length != count)
                throw new InvalidOperationException("Stage-1 arrays do not align.");
            for (int i = 0; i < count; i++)
            {
                if (stage1.deployedLabel[i] > 2) throw new InvalidOperationException("Stage-1 label is outside 0..2.");
                if (float.IsNaN(stage1.poleScore[i]) || float.IsInfinity(stage1.poleScore[i]) ||
                    float.IsNaN(stage1.lineScore[i]) || float.IsInfinity(stage1.lineScore[i]))
                    throw new InvalidOperationException("Stage-1 score is non-finite.");
            }
        }

        private List<PoleRecord> ExtractProductionPoles(Stage1SliceResult stage1, Stage2SliceResult output)
        {
            var timer = Stopwatch.StartNew();
            var candidates = new List<int>();
            for (int i = 0; i < stage1.voxels.Length; i++)
                if (stage1.poleScore[i] >= 0.15f && stage1.poleScore[i] >= stage1.lineScore[i] * 0.8f)
                    candidates.Add(i);
            // scipy.sparse.csgraph assigns production component IDs in first-row
            // order. Preserve that ordering because the pole IDs are part of the
            // accepted CSV contract and feed attachment ranking.
            List<int[]> components = ConnectedComponents(stage1.voxels, candidates, false);
            output.timing.production_component_ms = timer.Elapsed.TotalMilliseconds;
            timer.Restart();
            var accepted = new List<PoleRecord>();
            int componentNumber = 0;
            foreach (int[] rows in components)
            {
                componentNumber++;
                if (rows.Length < 4) continue;
                string componentId = "P" + componentNumber.ToString("D5", CultureInfo.InvariantCulture);
                ComponentGeometry geometry = AnalyzeComponent(stage1, rows, componentId, "pole");
                double probability = trees.PredictPole(geometry.features);
                bool edge = geometry.features["touches_xy_edge"] != 0.0;
                double zSpan = geometry.features["z_span_ft"];
                double minHeight = edge ? 4.0 : 10.0;
                bool physical = zSpan >= minHeight && zSpan <= 90.0 &&
                    geometry.features["radius_p90_ft"] <= 3.5 &&
                    geometry.features["principal_verticality"] >= 0.65 &&
                    geometry.features["horizontal_span_ft"] / Math.Max(zSpan, 1e-6) <= 0.65;
                bool componentAccepted = probability >= trees.PoleThreshold && physical;
                var componentRow = FeatureRow(geometry.features);
                componentRow["component_id"] = componentId;
                componentRow["class_name"] = "pole";
                componentRow["refiner_probability"] = probability;
                componentRow["physical_ok"] = physical;
                componentRow["component_accept"] = componentAccepted;
                componentRow["accept_mode"] = componentAccepted ? "refiner_plus_loose_hard_gate" : "rejected";
                componentRow["file_id"] = output.fileId;
                componentRow["slice_seq"] = output.sliceSeq;
                output.components.Add(componentRow);
                if (!componentAccepted) continue;

                Dictionary<string, double> parameter = PoleParameters(geometry.points);
                var record = new PoleRecord
                {
                    id = componentId,
                    baseX = parameter["base_x"], baseY = parameter["base_y"], baseZ = parameter["base_z"],
                    topX = parameter["top_x"], topY = parameter["top_y"], topZ = parameter["top_z"],
                    radiusP90Ft = geometry.features["radius_p90_ft"]
                };
                accepted.Add(record);
                output.poles.Add(new Dictionary<string, object>
                {
                    ["file_id"] = output.fileId, ["component_id"] = componentId,
                    ["slice_seq"] = output.sliceSeq, ["refiner_probability"] = probability,
                    ["touches_xy_edge"] = edge, ["radius_p90_ft"] = geometry.features["radius_p90_ft"],
                    ["verticality"] = geometry.features["principal_verticality"],
                    ["base_x"] = record.baseX, ["base_y"] = record.baseY, ["base_z"] = record.baseZ,
                    ["top_x"] = record.topX, ["top_y"] = record.topY, ["top_z"] = record.topZ,
                    ["height_ft"] = parameter["height_ft"], ["tilt_ft"] = parameter["tilt_ft"]
                });
            }
            output.timing.production_refiner_parametric_ms = timer.Elapsed.TotalMilliseconds;
            return accepted;
        }

        private ComponentGeometry AnalyzeComponent(Stage1SliceResult stage1, int[] rows,
            string componentId, string className)
        {
            D3[] points = rows.Select(i => ToD3(stage1.voxels[i])).ToArray();
            double[] scores = rows.Select(i => className == "pole" ? (double)stage1.poleScore[i] : stage1.lineScore[i]).ToArray();
            int minX = rows.Min(i => stage1.voxels[i].x), maxX = rows.Max(i => stage1.voxels[i].x);
            int minY = rows.Min(i => stage1.voxels[i].y), maxY = rows.Max(i => stage1.voxels[i].y);
            int minZ = rows.Min(i => stage1.voxels[i].z), maxZ = rows.Max(i => stage1.voxels[i].z);
            D3 median = new D3(Median(points.Select(p => p.x)), Median(points.Select(p => p.y)), Median(points.Select(p => p.z)));
            D3 principal = Principal3(points, median, out double[] eigenvalues);
            double e1 = Math.Max(0, eigenvalues[0]);
            double e2 = Math.Max(0, eigenvalues[1]);
            double e3 = Math.Max(0, eigenvalues[2]);
            double verticality = Math.Abs(principal.z);
            double horizontalHead = Math.Sqrt(Math.Max(0.0, 1.0 - verticality * verticality));
            double[] radial = points.Select(p =>
            {
                D3 centered = p - median;
                D3 fitted = median + principal * centered.Dot(principal);
                return (p - fitted).Norm() * voxelSizeFt;
            }).ToArray();
            D3 centerXy = new D3(Median(points.Select(p => p.x)), Median(points.Select(p => p.y)), 0);
            D3 axisXy = Principal2(points, centerXy);
            var projected = points.Select((point, index) => new
            {
                point,
                index,
                value = (point.x - centerXy.x) * axisXy.x + (point.y - centerXy.y) * axisXy.y
            }).OrderBy(v => v.value).ThenBy(v => v.index).ToArray();
            D3 endpoint1 = projected[0].point;
            D3 endpoint2 = projected[projected.Length - 1].point;
            double xyPath = 0.0;
            for (int i = 1; i < projected.Length; i++)
            {
                double dx = projected[i].point.x - projected[i - 1].point.x;
                double dy = projected[i].point.y - projected[i - 1].point.y;
                xyPath += Math.Sqrt(dx * dx + dy * dy) * voxelSizeFt;
            }
            double endpointDistance = Math.Sqrt(
                Square(endpoint2.x - endpoint1.x) + Square(endpoint2.y - endpoint1.y)) * voxelSizeFt;
            double quadraticRmse = 0.0, sag = 0.0;
            QuadraticGeometry(projected.Select(v => v.value * voxelSizeFt).ToArray(),
                projected.Select(v => v.point.z * voxelSizeFt).ToArray(), ref quadraticRmse, ref sag);
            double edgeDistance = points.Min(p => Math.Min(Math.Min(p.x, grid[0] - 1 - p.x), Math.Min(p.y, grid[1] - 1 - p.y)));
            bool touchesEdge = edgeDistance <= 10.0;
            double mean = scores.Average();
            double variance = scores.Select(v => Square(v - mean)).Average();
            double xSpan = (maxX - minX + 1) * voxelSizeFt;
            double ySpan = (maxY - minY + 1) * voxelSizeFt;
            double zSpan = (maxZ - minZ + 1) * voxelSizeFt;
            double bbox = (double)(maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1);
            var f = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["n_voxels"] = rows.Length, ["score_mean"] = mean, ["score_std"] = Math.Sqrt(variance),
                ["score_p10"] = Quantile(scores, 0.1), ["score_p50"] = Quantile(scores, 0.5),
                ["score_p90"] = Quantile(scores, 0.9), ["score_max"] = scores.Max(),
                ["vertical_head_mean"] = verticality, ["horizontal_head_mean"] = horizontalHead,
                ["x_span_ft"] = xSpan, ["y_span_ft"] = ySpan, ["z_span_ft"] = zSpan,
                ["horizontal_span_ft"] = Math.Sqrt(xSpan * xSpan + ySpan * ySpan),
                ["bbox_density"] = rows.Length / Math.Max(bbox, 1.0),
                ["center_x"] = points.Average(p => p.x), ["center_y"] = points.Average(p => p.y),
                ["center_z"] = points.Average(p => p.z), ["center_z_ft"] = points.Average(p => p.z) * voxelSizeFt,
                ["min_z_ft"] = minZ * voxelSizeFt, ["max_z_ft"] = maxZ * voxelSizeFt,
                ["principal_dx"] = principal.x, ["principal_dy"] = principal.y, ["principal_dz"] = principal.z,
                ["principal_verticality"] = verticality,
                ["linearity"] = (e1 - e2) / Math.Max(e1, 1e-9),
                ["planarity"] = (e2 - e3) / Math.Max(e1, 1e-9),
                ["scattering"] = e3 / Math.Max(e1, 1e-9),
                ["radius_p50_ft"] = Quantile(radial, 0.5), ["radius_p90_ft"] = Quantile(radial, 0.9),
                ["xy_path_length_ft"] = xyPath, ["xy_endpoint_distance_ft"] = endpointDistance,
                ["xy_tortuosity"] = xyPath / Math.Max(endpointDistance, 1e-6),
                ["quadratic_rmse_ft"] = quadraticRmse, ["sag_estimate_ft"] = sag,
                ["endpoint1_x"] = endpoint1.x, ["endpoint1_y"] = endpoint1.y, ["endpoint1_z"] = endpoint1.z,
                ["endpoint2_x"] = endpoint2.x, ["endpoint2_y"] = endpoint2.y, ["endpoint2_z"] = endpoint2.z,
                ["exact_gt_fraction"] = 0.0, ["near_gt_fraction"] = 0.0,
                ["edge_distance_vox"] = edgeDistance, ["touches_xy_edge"] = touchesEdge ? 1.0 : 0.0,
                ["partial_pole_candidate"] = className == "pole" && touchesEdge && verticality >= 0.65 ? 1.0 : 0.0
            };
            return new ComponentGeometry
            {
                rows = rows, features = f, points = points, principal = principal,
                endpoint1 = endpoint1, endpoint2 = endpoint2
            };
        }

        private void BuildStrictLines(Stage1SliceResult stage1, List<PoleRecord> poles, Stage2SliceResult output)
        {
            var timer = Stopwatch.StartNew();
            var lineRows = new List<int>();
            var poleSupport = new Dictionary<Int3Key, List<IndexedPoleVoxel>>();
            int poleOrder = 0;
            for (int i = 0; i < stage1.voxels.Length; i++)
            {
                if (stage1.deployedLabel[i] == 2) lineRows.Add(i);
                if (stage1.deployedLabel[i] == 1)
                {
                    Int3Key key = Key(stage1.voxels[i]);
                    if (!poleSupport.TryGetValue(key, out List<IndexedPoleVoxel> list))
                    {
                        list = new List<IndexedPoleVoxel>();
                        poleSupport.Add(key, list);
                    }
                    list.Add(new IndexedPoleVoxel(poleOrder++, ToD3(stage1.voxels[i])));
                }
            }
            output.stage1LineRows.AddRange(lineRows);
            List<int[]> components = ConnectedComponents(stage1.voxels, lineRows, true);
            output.timing.line_connected_components_ms = timer.Elapsed.TotalMilliseconds;
            timer.Restart();
            var traces = new List<Trace>();
            for (int i = 0; i < components.Count; i++) traces.AddRange(TraceComponent(i, components[i], stage1.voxels));
            output.timing.line_trace_ms = timer.Elapsed.TotalMilliseconds;
            timer.Restart();
            int[] assigned = traces.SelectMany(v => v.voxelRows).ToArray();
            if (assigned.Length != lineRows.Count || assigned.Distinct().Count() != lineRows.Count ||
                !new HashSet<int>(assigned).SetEquals(lineRows))
                throw new InvalidOperationException("Stage1 line assignment changed voxel identity.");
            output.acceptedLineRows.AddRange(assigned);
            output.timing.line_assignment_audit_ms = timer.Elapsed.TotalMilliseconds;
            timer.Restart();

            var globalSupport = new HashSet<Int3Key>(lineRows.Select(i => Key(stage1.voxels[i])));
            var usedAttachments = new HashSet<string>(StringComparer.Ordinal);
            int geometrySupported = 0, geometryTotal = 0, points = 0;
            double maximumTurn = 0.0, maximumRadius = 0.0;
            for (int track = 0; track < traces.Count; track++)
            {
                Trace trace = traces[track];
                string componentId = "S1E" + (track + 1).ToString("D5", CultureInfo.InvariantCulture);
                D3[] path = trace.pathRows.Select(i => ToD3(stage1.voxels[i])).ToArray();
                D3[] vertices = SimplifySupported(path);
                if (vertices.Length == 1) points++;
                List<Dictionary<string, object>> attachments = Attach(vertices, poleSupport, poles, componentId, usedAttachments);
                output.attachments.AddRange(attachments);
                CountPolylineSupport(vertices, globalSupport, out int supported, out int total);
                if (supported != total)
                    throw new InvalidOperationException("Stage2 line geometry left Stage1 voxel support: " + componentId);
                geometrySupported += supported;
                geometryTotal += total;
                double maxTurn = PolylineMaximumTurn(vertices);
                maximumTurn = Math.Max(maximumTurn, maxTurn);
                double horizontalSpan = 0, verticalSpan = 0, verticality = 0, tortuosity = 1;
                if (vertices.Length >= 2)
                {
                    double length = 0.0;
                    for (int i = 1; i < vertices.Length; i++) length += (vertices[i] - vertices[i - 1]).Norm() * voxelSizeFt;
                    D3 direct = (vertices[vertices.Length - 1] - vertices[0]) * voxelSizeFt;
                    double directLength = direct.Norm();
                    horizontalSpan = Math.Sqrt(direct.x * direct.x + direct.y * direct.y);
                    verticalSpan = Math.Abs(direct.z);
                    verticality = verticalSpan / Math.Max(directLength, 1e-12);
                    tortuosity = length / Math.Max(directLength, 1e-12);
                }
                D3[] voxelPoints = trace.voxelRows.Select(i => ToD3(stage1.voxels[i])).ToArray();
                double radius = TrackRadiusP90(voxelPoints);
                maximumRadius = Math.Max(maximumRadius, radius);
                double scoreMean = trace.voxelRows.Average(i => (double)stage1.lineScore[i]);
                output.lines.Add(new Dictionary<string, object>
                {
                    ["file_id"] = output.fileId, ["component_id"] = componentId,
                    ["slice_seq"] = output.sliceSeq, ["refiner_probability"] = double.NaN,
                    ["horizontal_span_ft"] = horizontalSpan, ["vertical_span_ft"] = verticalSpan,
                    ["verticality"] = verticality, ["tortuosity"] = tortuosity,
                    ["vertex_count"] = vertices.Length
                });
                for (int i = 0; i < vertices.Length; i++)
                    output.vertices.Add(new Dictionary<string, object>
                    {
                        ["file_id"] = output.fileId, ["component_id"] = componentId,
                        ["slice_seq"] = output.sliceSeq, ["vertex_index"] = i,
                        ["x"] = vertices[i].x, ["y"] = vertices[i].y, ["z"] = vertices[i].z
                    });
                var component = new Dictionary<string, object>
                {
                    ["component_id"] = componentId, ["class_name"] = "line",
                    ["n_voxels"] = trace.voxelRows.Length, ["score_mean"] = scoreMean,
                    ["component_accept"] = true,
                    ["accept_mode"] = "stage1_inferred_voxel_supported_open_track",
                    ["stage1_line_label_fraction"] = 1.0, ["raw_fragment_count"] = 1,
                    ["bridge_count"] = 0, ["pole_attachment_count"] = attachments.Count,
                    ["max_bridge_gap_ft"] = 0.0, ["track_radius_p95_ft"] = radius,
                    ["synthetic_line_voxels"] = 0, ["runtime_gt_usage"] = false,
                    ["source_component_index"] = trace.sourceComponent,
                    ["file_id"] = output.fileId, ["slice_seq"] = output.sliceSeq
                };
                output.components.Add(component);
                output.tracks.Add(new Dictionary<string, object>
                {
                    ["component_id"] = componentId, ["source_component_index"] = trace.sourceComponent,
                    ["raw_fragment_count"] = 1, ["n_voxels"] = trace.voxelRows.Length,
                    ["vertex_count"] = vertices.Length, ["bridge_count"] = 0,
                    ["pole_attachment_count"] = attachments.Count, ["max_bridge_gap_ft"] = 0.0,
                    ["track_radius_p95_ft"] = radius, ["horizontal_span_ft"] = horizontalSpan,
                    ["vertical_span_ft"] = verticalSpan, ["score_mean"] = scoreMean,
                    ["max_turn_deg"] = maxTurn, ["geometry_support_fraction"] = 1.0
                });
            }
            output.timing.line_geometry_ms = timer.Elapsed.TotalMilliseconds;
            output.audit.stage1_inferred_line_voxels = lineRows.Count;
            output.audit.accepted_stage1_line_voxels = assigned.Length;
            output.audit.stage1_to_stage2_voxel_preservation = lineRows.Count == 0 ? 1.0 : assigned.Length / (double)lineRows.Count;
            output.audit.raw_stage1_line_components = components.Count;
            output.audit.joined_stage2_tracks = traces.Count;
            output.audit.line_path_fragments = traces.Count;
            output.audit.point_line_fragments = points;
            output.audit.pole_attachments = output.attachments.Count;
            output.audit.geometry_support_samples = geometryTotal;
            output.audit.geometry_supported_samples = geometrySupported;
            output.audit.geometry_stage1_voxel_support_fraction = geometryTotal == 0 ? 1.0 : geometrySupported / (double)geometryTotal;
            output.audit.geometry_outside_stage1_voxel_samples = geometryTotal - geometrySupported;
            output.audit.max_output_turn_deg = maximumTurn;
            output.audit.max_track_radius_p95_ft = maximumRadius;
        }

        private List<Trace> TraceComponent(int componentIndex, int[] rows, SparseVoxel[] voxels)
        {
            Dictionary<int, List<Edge>> adjacency = Adjacency(rows, voxels, out Dictionary<Int3Key, int> coordinateRows);
            var remaining = new HashSet<int>(rows);
            var output = new List<Trace>();
            int radius = Math.Max(0, profile.centerline_coverage_radius_vox);
            while (remaining.Count > 0)
            {
                int seed = remaining.Min();
                HashSet<int> sub = RestrictedComponent(seed, remaining, adjacency);
                List<int> path = DiameterPath(sub, adjacency);
                if (path.Count == 0) throw new InvalidOperationException("Empty path for nonempty line component.");
                var covered = new HashSet<int>();
                foreach (int row in path)
                {
                    SparseVoxel voxel = voxels[row];
                    for (int dz = -radius; dz <= radius; dz++)
                    for (int dy = -radius; dy <= radius; dy++)
                    for (int dx = -radius; dx <= radius; dx++)
                        if (coordinateRows.TryGetValue(new Int3Key(voxel.x + dx, voxel.y + dy, voxel.z + dz), out int other) &&
                            sub.Contains(other) && remaining.Contains(other)) covered.Add(other);
                }
                covered.UnionWith(path);
                remaining.ExceptWith(covered);
                List<List<int>> parts = SplitSharpTurns(path, voxels);
                List<int[]> assigned = PartitionAssigned(covered, parts, voxels);
                for (int i = 0; i < parts.Count; i++)
                    if (assigned[i].Length > 0)
                        output.Add(new Trace
                        {
                            sourceComponent = componentIndex,
                            voxelRows = assigned[i],
                            pathRows = parts[i].ToArray()
                        });
            }
            int[] all = output.SelectMany(v => v.voxelRows).ToArray();
            if (all.Length != rows.Length || all.Distinct().Count() != rows.Length)
                throw new InvalidOperationException("Line path cover mismatch for component " + componentIndex);
            return output;
        }

        private static Dictionary<int, List<Edge>> Adjacency(int[] rows, SparseVoxel[] voxels,
            out Dictionary<Int3Key, int> coordinateRows)
        {
            coordinateRows = rows.ToDictionary(i => Key(voxels[i]), i => i);
            var adjacency = rows.ToDictionary(i => i, i => new List<Edge>());
            foreach (int row in rows)
            {
                SparseVoxel voxel = voxels[row];
                foreach (Int3Key delta in Neighbor26)
                {
                    if (!coordinateRows.TryGetValue(new Int3Key(voxel.x + delta.x, voxel.y + delta.y, voxel.z + delta.z), out int other) || other <= row)
                        continue;
                    double weight = Math.Sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
                    adjacency[row].Add(new Edge(other, weight));
                    adjacency[other].Add(new Edge(row, weight));
                }
            }
            return adjacency;
        }

        private static HashSet<int> RestrictedComponent(int seed, HashSet<int> allowed,
            Dictionary<int, List<Edge>> adjacency)
        {
            var found = new HashSet<int> { seed };
            var stack = new Stack<int>();
            stack.Push(seed);
            while (stack.Count > 0)
            {
                int current = stack.Pop();
                foreach (Edge edge in adjacency[current])
                    if (allowed.Contains(edge.node) && found.Add(edge.node)) stack.Push(edge.node);
            }
            return found;
        }

        private static int GraphFarthest(int start, HashSet<int> allowed,
            Dictionary<int, List<Edge>> adjacency, out Dictionary<int, int> previous)
        {
            var distance = new Dictionary<int, double> { [start] = 0.0 };
            previous = new Dictionary<int, int>();
            var heap = new MinHeap();
            heap.Push(new HeapEntry(0.0, start));
            while (heap.Count > 0)
            {
                HeapEntry entry = heap.Pop();
                if (!distance.TryGetValue(entry.node, out double current) || entry.distance != current) continue;
                foreach (Edge edge in adjacency[entry.node])
                {
                    if (!allowed.Contains(edge.node)) continue;
                    double next = entry.distance + edge.weight;
                    if (!distance.TryGetValue(edge.node, out double old) || next + 1e-12 < old)
                    {
                        distance[edge.node] = next;
                        previous[edge.node] = entry.node;
                        heap.Push(new HeapEntry(next, edge.node));
                    }
                }
            }
            return distance.OrderByDescending(v => v.Value).ThenBy(v => v.Key).First().Key;
        }

        private static List<int> DiameterPath(HashSet<int> nodes, Dictionary<int, List<Edge>> adjacency)
        {
            if (nodes.Count <= 1) return nodes.ToList();
            int first = GraphFarthest(nodes.Min(), nodes, adjacency, out _);
            int last = GraphFarthest(first, nodes, adjacency, out Dictionary<int, int> previous);
            var path = new List<int> { last };
            while (path[path.Count - 1] != first)
            {
                if (!previous.TryGetValue(path[path.Count - 1], out int parent))
                    throw new InvalidOperationException("Voxel graph diameter reconstruction failed.");
                path.Add(parent);
            }
            path.Reverse();
            return path;
        }

        private List<List<int>> SplitSharpTurns(List<int> path, SparseVoxel[] voxels)
        {
            if (path.Count < 3) return new List<List<int>> { path };
            int window = Math.Max(1, profile.turn_tangent_window_vox);
            var cuts = new List<int>();
            int lastCut = 0;
            for (int i = 1; i < path.Count - 1; i++)
            {
                int w = Math.Min(window, Math.Min(i, path.Count - 1 - i));
                D3 current = ToD3(voxels[path[i]]);
                double sustained = TurnAngle(current - ToD3(voxels[path[i - w]]), ToD3(voxels[path[i + w]]) - current);
                double immediate = TurnAngle(current - ToD3(voxels[path[i - 1]]), ToD3(voxels[path[i + 1]]) - current);
                if (sustained <= profile.max_local_turn_deg && immediate <= 120.0) continue;
                if (i - lastCut < 2) continue;
                cuts.Add(i);
                lastCut = i + 1;
            }
            if (cuts.Count == 0) return new List<List<int>> { path };
            var output = new List<List<int>>();
            int start = 0;
            foreach (int cut in cuts)
            {
                output.Add(path.GetRange(start, cut + 1 - start));
                start = cut + 1;
            }
            if (start < path.Count) output.Add(path.GetRange(start, path.Count - start));
            return output;
        }

        private static List<int[]> PartitionAssigned(HashSet<int> assigned, List<List<int>> parts, SparseVoxel[] voxels)
        {
            if (parts.Count == 1) return new List<int[]> { assigned.OrderBy(v => v).ToArray() };
            var buckets = Enumerable.Range(0, parts.Count).Select(_ => new List<int>()).ToArray();
            foreach (int row in assigned.OrderBy(v => v))
            {
                D3 point = ToD3(voxels[row]);
                int bestPart = 0;
                double best = double.PositiveInfinity;
                for (int p = 0; p < parts.Count; p++)
                {
                    double distance = parts[p].Min(candidate => DistanceSquared(point, ToD3(voxels[candidate])));
                    if (distance < best) { best = distance; bestPart = p; }
                }
                buckets[bestPart].Add(row);
            }
            return buckets.Select(v => v.ToArray()).ToList();
        }

        private D3[] SimplifySupported(D3[] path)
        {
            if (path.Length <= 2) return path.ToArray();
            var support = new HashSet<Int3Key>(path.Select(Key));
            double maxSpan = Math.Max(1.0, profile.max_supported_chord_vox);
            int lookahead = Math.Max(2, profile.max_supported_chord_lookahead);
            var output = new List<D3> { path[0] };
            int i = 0;
            while (i < path.Length - 1)
            {
                int upper = Math.Min(path.Length - 1, i + lookahead);
                int chosen = i + 1;
                for (int j = upper; j > i; j--)
                {
                    if ((path[j] - path[i]).Norm() > maxSpan) continue;
                    SegmentSupport(path[i], path[j], support, out int good, out int total);
                    if (good == total) { chosen = j; break; }
                }
                output.Add(path[chosen]);
                i = chosen;
            }
            return output.ToArray();
        }

        private List<Dictionary<string, object>> Attach(D3[] vertices,
            Dictionary<Int3Key, List<IndexedPoleVoxel>> poleSupport, List<PoleRecord> poles,
            string componentId, HashSet<string> used)
        {
            var output = new List<Dictionary<string, object>>();
            if (vertices.Length < 2 || poles.Count == 0 || poleSupport.Count == 0) return output;
            foreach (string end in new[] { "start", "end" })
            {
                D3 endpoint = end == "start" ? vertices[0] : vertices[vertices.Length - 1];
                D3 outward = EndpointDirection(vertices, end);
                Dictionary<string, object> candidate = AttachmentCandidate(endpoint, outward, poleSupport, poles);
                if (candidate == null) continue;
                string key = Convert.ToString(candidate["pole_component_id"], CultureInfo.InvariantCulture) + "|" +
                    Math.Round((double)candidate["anchor_x"], 6, MidpointRounding.ToEven).ToString("R", CultureInfo.InvariantCulture) + "|" +
                    Math.Round((double)candidate["anchor_y"], 6, MidpointRounding.ToEven).ToString("R", CultureInfo.InvariantCulture) + "|" +
                    Math.Round((double)candidate["anchor_z"], 6, MidpointRounding.ToEven).ToString("R", CultureInfo.InvariantCulture);
                if (!used.Add(key)) continue;
                candidate["component_id"] = componentId;
                candidate["track_end"] = end;
                output.Add(candidate);
            }
            return output;
        }

        private Dictionary<string, object> AttachmentCandidate(D3 endpoint, D3 outward,
            Dictionary<Int3Key, List<IndexedPoleVoxel>> poleSupport, List<PoleRecord> poles)
        {
            double contactMaximum = profile.pole_contact_max_chebyshev_vox;
            int radius = Math.Max(1, (int)Math.Ceiling(contactMaximum));
            int px = (int)endpoint.x, py = (int)endpoint.y, pz = (int)endpoint.z;
            var indexedContacts = new List<IndexedPoleVoxel>();
            for (int dz = -radius; dz <= radius; dz++)
            for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++)
                if (poleSupport.TryGetValue(new Int3Key(px + dx, py + dy, pz + dz), out List<IndexedPoleVoxel> values))
                    indexedContacts.AddRange(values);
            D3[] contacts = indexedContacts.OrderBy(v => v.order).Select(v => v.point)
                .Where(q => Chebyshev(q - endpoint) <= contactMaximum + 1e-12 &&
                            (q - endpoint).Norm() > 1e-12).ToArray();
            Dictionary<string, object> best = null;
            Tuple<double, double, double, string> bestRank = null;
            foreach (PoleRecord pole in poles)
            foreach (D3 contact in contacts)
            {
                double zLow = Math.Min(pole.baseZ, pole.topZ), zHigh = Math.Max(pole.baseZ, pole.topZ);
                if (zHigh - zLow <= 1e-9) continue;
                double fraction = (contact.z - zLow) / (zHigh - zLow);
                if (fraction < profile.pole_attach_min_height_fraction || fraction > 1.10) continue;
                double u = Math.Max(0.0, Math.Min(1.0, (contact.z - pole.baseZ) / (pole.topZ - pole.baseZ)));
                double poleX = pole.baseX + u * (pole.topX - pole.baseX);
                double poleY = pole.baseY + u * (pole.topY - pole.baseY);
                double axisDistance = Math.Sqrt(Square(contact.x - poleX) + Square(contact.y - poleY)) * voxelSizeFt;
                double surfaceRadius = Math.Max(profile.pole_surface_standoff_min_ft, pole.radiusP90Ft);
                if (axisDistance > surfaceRadius + Math.Sqrt(2.0) * voxelSizeFt) continue;
                D3 toward = new D3(contact.x - endpoint.x, contact.y - endpoint.y, 0);
                double angle = toward.Norm() <= 1e-12 ? 0.0 : AxisAngle(outward, toward);
                if (angle > profile.pole_attach_max_angle_deg) continue;
                double contactDistance = (contact - endpoint).Norm();
                var rank = Tuple.Create(contactDistance, axisDistance, angle, pole.id);
                if (bestRank != null && CompareRank(rank, bestRank) >= 0) continue;
                bestRank = rank;
                best = new Dictionary<string, object>
                {
                    ["pole_component_id"] = pole.id,
                    ["endpoint_distance_to_pole_axis_ft"] = axisDistance,
                    ["endpoint_to_pole_angle_deg"] = angle,
                    ["attachment_height_fraction"] = fraction,
                    ["anchor_x"] = endpoint.x, ["anchor_y"] = endpoint.y, ["anchor_z"] = endpoint.z,
                    ["pole_surface_radius_ft"] = surfaceRadius,
                    ["contact_pole_voxel_x"] = contact.x, ["contact_pole_voxel_y"] = contact.y,
                    ["contact_pole_voxel_z"] = contact.z, ["contact_distance_vox"] = contactDistance,
                    ["attachment_support_mode"] = "direct_stage1_line_pole_voxel_contact"
                };
            }
            return best;
        }

        private static int CompareRank(Tuple<double, double, double, string> a,
            Tuple<double, double, double, string> b)
        {
            int c = a.Item1.CompareTo(b.Item1); if (c != 0) return c;
            c = a.Item2.CompareTo(b.Item2); if (c != 0) return c;
            c = a.Item3.CompareTo(b.Item3); if (c != 0) return c;
            return StringComparer.Ordinal.Compare(a.Item4, b.Item4);
        }

        private static D3 EndpointDirection(D3[] vertices, string end)
        {
            if (vertices.Length < 2) return new D3(1, 0, 0);
            int window = Math.Min(4, vertices.Length - 1);
            D3 value = end == "start" ? vertices[0] - vertices[window] :
                vertices[vertices.Length - 1] - vertices[vertices.Length - 1 - window];
            return new D3(value.x, value.y, 0).Unit(new D3(1, 0, 0));
        }

        private static List<int[]> ConnectedComponents(SparseVoxel[] voxels, List<int> rows, bool sortBySize)
        {
            if (rows.Count == 0) return new List<int[]>();
            var localByCoordinate = new Dictionary<Int3Key, int>(rows.Count);
            for (int i = 0; i < rows.Count; i++) localByCoordinate[Key(voxels[rows[i]])] = i;
            var union = new UnionFind(rows.Count);
            for (int local = 0; local < rows.Count; local++)
            {
                SparseVoxel voxel = voxels[rows[local]];
                foreach (Int3Key delta in Forward26)
                    if (localByCoordinate.TryGetValue(new Int3Key(voxel.x + delta.x, voxel.y + delta.y, voxel.z + delta.z), out int other))
                        union.Union(local, other);
            }
            var groups = new Dictionary<int, List<int>>();
            for (int local = 0; local < rows.Count; local++)
            {
                int root = union.Find(local);
                if (!groups.TryGetValue(root, out List<int> componentRows))
                {
                    componentRows = new List<int>();
                    groups.Add(root, componentRows);
                }
                componentRows.Add(rows[local]);
            }
            IEnumerable<int[]> components = groups.Values.Select(v => v.ToArray());
            return sortBySize
                ? components.OrderByDescending(v => v.Length).ThenBy(v => v.Min()).ToList()
                : components.OrderBy(v => v.Min()).ToList();
        }

        private static Int3Key[] BuildNeighborOffsets(bool forwardOnly)
        {
            var values = new List<Int3Key>();
            for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (forwardOnly && !(dz > 0 || dz == 0 && (dy > 0 || dy == 0 && dx > 0))) continue;
                values.Add(new Int3Key(dx, dy, dz));
            }
            return values.ToArray();
        }

        private void SegmentSupport(D3 a, D3 b, HashSet<Int3Key> support, out int good, out int total)
        {
            double length = (b - a).Norm();
            int samples = Math.Max(1, (int)Math.Ceiling(length / Math.Max(profile.support_sample_step_vox, 1e-6f)));
            good = 0;
            total = samples + 1;
            for (int i = 0; i <= samples; i++)
                if (PointInsideSupport(a + (b - a) * (i / (double)samples), support)) good++;
        }

        private static bool PointInsideSupport(D3 point, HashSet<Int3Key> support)
        {
            const double tolerance = 0.500001;
            int x0 = (int)Math.Ceiling(point.x - tolerance), x1 = (int)Math.Floor(point.x + tolerance);
            int y0 = (int)Math.Ceiling(point.y - tolerance), y1 = (int)Math.Floor(point.y + tolerance);
            int z0 = (int)Math.Ceiling(point.z - tolerance), z1 = (int)Math.Floor(point.z + tolerance);
            for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
                if (support.Contains(new Int3Key(x, y, z))) return true;
            return false;
        }

        private void CountPolylineSupport(D3[] vertices, HashSet<Int3Key> support, out int good, out int total)
        {
            good = total = 0;
            if (vertices.Length == 0) return;
            if (vertices.Length == 1)
            {
                good = PointInsideSupport(vertices[0], support) ? 1 : 0;
                total = 1;
                return;
            }
            for (int i = 1; i < vertices.Length; i++)
            {
                SegmentSupport(vertices[i - 1], vertices[i], support, out int localGood, out int localTotal);
                good += localGood;
                total += localTotal;
            }
        }

        private static double PolylineMaximumTurn(D3[] vertices)
        {
            double maximum = 0.0;
            for (int i = 1; i < vertices.Length - 1; i++)
                maximum = Math.Max(maximum, TurnAngle(vertices[i] - vertices[i - 1], vertices[i + 1] - vertices[i]));
            return maximum;
        }

        private double TrackRadiusP90(D3[] points)
        {
            if (points.Length == 0) return 0.0;
            D3 center = new D3(points.Average(v => v.x), points.Average(v => v.y), 0);
            D3 axis = Principal2(points, center);
            D3 normal = new D3(-axis.y, axis.x, 0);
            return Quantile(points.Select(p => Math.Abs((p - center).Dot(normal)) * voxelSizeFt), 0.9);
        }

        private Dictionary<string, double> PoleParameters(D3[] points)
        {
            double[] z = points.Select(v => v.z).ToArray();
            double low = Quantile(z, 0.15), high = Quantile(z, 0.85);
            D3 bottom = MedianPoint(points.Where(v => v.z <= low));
            D3 top = MedianPoint(points.Where(v => v.z >= high));
            double minZ = z.Min(), maxZ = z.Max();
            return new Dictionary<string, double>
            {
                ["base_x"] = bottom.x, ["base_y"] = bottom.y, ["base_z"] = minZ,
                ["top_x"] = top.x, ["top_y"] = top.y, ["top_z"] = maxZ,
                ["height_ft"] = (maxZ - minZ + 1.0) * voxelSizeFt,
                ["tilt_ft"] = Math.Sqrt(Square(top.x - bottom.x) + Square(top.y - bottom.y)) * voxelSizeFt
            };
        }

        private static Dictionary<string, object> FeatureRow(Dictionary<string, double> features)
        {
            var output = new Dictionary<string, object>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, double> item in features) output[item.Key] = item.Value;
            return output;
        }

        private static D3 Principal3(D3[] points, D3 center, out double[] eigenvalues)
        {
            if (points.Length < 2)
            {
                eigenvalues = new[] { 0.0, 0.0, 0.0 };
                return new D3(0, 0, 1);
            }
            // NumPy uses the median-centred matrix for SVD direction, but np.cov
            // recentres that matrix by its mean for the reported eigenvalues.
            var directionScatter = new double[3, 3];
            foreach (D3 point in points)
            {
                D3 q = point - center;
                AddOuter(directionScatter, q);
            }
            CompleteSymmetric(directionScatter);
            EigenSymmetric3(directionScatter, out _, out D3[] vectors);

            D3 mean = new D3(points.Average(v => v.x), points.Average(v => v.y), points.Average(v => v.z));
            var covariance = new double[3, 3];
            foreach (D3 point in points) AddOuter(covariance, point - mean);
            CompleteSymmetric(covariance);
            EigenSymmetric3(covariance, out eigenvalues, out _);
            if (points.Length > 2)
                for (int i = 0; i < eigenvalues.Length; i++) eigenvalues[i] /= points.Length - 1.0;
            return Canonical(vectors[0]).Unit(new D3(0, 0, 1));
        }

        private static void AddOuter(double[,] matrix, D3 q)
        {
            matrix[0, 0] += q.x * q.x; matrix[0, 1] += q.x * q.y; matrix[0, 2] += q.x * q.z;
            matrix[1, 1] += q.y * q.y; matrix[1, 2] += q.y * q.z; matrix[2, 2] += q.z * q.z;
        }

        private static void CompleteSymmetric(double[,] matrix)
        {
            matrix[1, 0] = matrix[0, 1]; matrix[2, 0] = matrix[0, 2]; matrix[2, 1] = matrix[1, 2];
        }

        private static D3 Principal2(D3[] points, D3 center)
        {
            if (points.Length < 2) return new D3(1, 0, 0);
            double xx = 0, xy = 0, yy = 0;
            foreach (D3 p in points)
            {
                double x = p.x - center.x, y = p.y - center.y;
                xx += x * x; xy += x * y; yy += y * y;
            }
            double angle = 0.5 * Math.Atan2(2.0 * xy, xx - yy);
            D3 axis = new D3(Math.Cos(angle), Math.Sin(angle), 0).Unit(new D3(1, 0, 0));
            if (axis.x < 0 || Math.Abs(axis.x) < 1e-12 && axis.y < 0) axis = axis * -1.0;
            return axis;
        }

        private static void EigenSymmetric3(double[,] a, out double[] eigenvalues, out D3[] eigenvectors)
        {
            var v = new double[3, 3];
            for (int i = 0; i < 3; i++) v[i, i] = 1.0;
            for (int iteration = 0; iteration < 64; iteration++)
            {
                int p = 0, q = 1;
                double maximum = Math.Abs(a[0, 1]);
                if (Math.Abs(a[0, 2]) > maximum) { p = 0; q = 2; maximum = Math.Abs(a[0, 2]); }
                if (Math.Abs(a[1, 2]) > maximum) { p = 1; q = 2; maximum = Math.Abs(a[1, 2]); }
                if (maximum < 1e-14) break;
                double angle = 0.5 * Math.Atan2(2.0 * a[p, q], a[q, q] - a[p, p]);
                double c = Math.Cos(angle), s = Math.Sin(angle);
                double app = a[p, p], aqq = a[q, q], apq = a[p, q];
                a[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                a[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                a[p, q] = a[q, p] = 0.0;
                for (int k = 0; k < 3; k++)
                {
                    if (k == p || k == q) continue;
                    double akp = a[k, p], akq = a[k, q];
                    a[k, p] = a[p, k] = c * akp - s * akq;
                    a[k, q] = a[q, k] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++)
                {
                    double vkp = v[k, p], vkq = v[k, q];
                    v[k, p] = c * vkp - s * vkq;
                    v[k, q] = s * vkp + c * vkq;
                }
            }
            var items = Enumerable.Range(0, 3).Select(i => new
            {
                value = a[i, i], vector = new D3(v[0, i], v[1, i], v[2, i])
            }).OrderByDescending(item => item.value).ToArray();
            eigenvalues = items.Select(item => item.value).ToArray();
            eigenvectors = items.Select(item => item.vector).ToArray();
        }

        private static D3 Canonical(D3 value)
        {
            double ax = Math.Abs(value.x), ay = Math.Abs(value.y), az = Math.Abs(value.z);
            double sign = ax >= ay && ax >= az ? Math.Sign(value.x) : ay >= az ? Math.Sign(value.y) : Math.Sign(value.z);
            return sign < 0 ? value * -1.0 : value;
        }

        private void QuadraticGeometry(double[] s, double[] z, ref double rmse, ref double sag)
        {
            if (s.Length < 6 || s.Any(v => double.IsNaN(v) || double.IsInfinity(v)) ||
                z.Any(v => double.IsNaN(v) || double.IsInfinity(v))) return;
            double min = s.Min(), max = s.Max(), span = max - min;
            if (span < Math.Max(voxelSizeFt * 2.0, 1e-6)) return;
            int unique = s.Select(v => Math.Round(v, 6, MidpointRounding.ToEven)).Distinct().Count();
            if (unique < 3) return;
            double midpoint = 0.5 * (min + max), half = Math.Max(0.5 * span, 1e-9);
            double[,] normal = new double[3, 3];
            double[] rhs = new double[3];
            for (int i = 0; i < s.Length; i++)
            {
                double u = (s[i] - midpoint) / half;
                double[] row = { u * u, u, 1.0 };
                for (int a = 0; a < 3; a++)
                {
                    rhs[a] += row[a] * z[i];
                    for (int b = 0; b < 3; b++) normal[a, b] += row[a] * row[b];
                }
            }
            if (!Solve3(normal, rhs, out double[] coefficient)) return;
            double error = 0.0;
            for (int i = 0; i < s.Length; i++)
            {
                double u = (s[i] - midpoint) / half;
                double fitted = coefficient[0] * u * u + coefficient[1] * u + coefficient[2];
                error += Square(fitted - z[i]);
            }
            rmse = Math.Sqrt(error / s.Length);
            sag = Math.Max(0.0, coefficient[0]);
        }

        private static bool Solve3(double[,] matrix, double[] rhs, out double[] output)
        {
            var a = new double[3, 4];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++) a[i, j] = matrix[i, j];
                a[i, 3] = rhs[i];
            }
            for (int column = 0; column < 3; column++)
            {
                int pivot = column;
                for (int row = column + 1; row < 3; row++)
                    if (Math.Abs(a[row, column]) > Math.Abs(a[pivot, column])) pivot = row;
                if (Math.Abs(a[pivot, column]) < 1e-12) { output = null; return false; }
                if (pivot != column)
                    for (int j = column; j < 4; j++) (a[column, j], a[pivot, j]) = (a[pivot, j], a[column, j]);
                double divisor = a[column, column];
                for (int j = column; j < 4; j++) a[column, j] /= divisor;
                for (int row = 0; row < 3; row++)
                {
                    if (row == column) continue;
                    double factor = a[row, column];
                    for (int j = column; j < 4; j++) a[row, j] -= factor * a[column, j];
                }
            }
            output = new[] { a[0, 3], a[1, 3], a[2, 3] };
            return true;
        }

        private static double Quantile(IEnumerable<double> source, double q)
        {
            double[] values = source.OrderBy(v => v).ToArray();
            if (values.Length == 0) return double.NaN;
            double position = (values.Length - 1) * q;
            int lower = (int)Math.Floor(position), upper = (int)Math.Ceiling(position);
            if (lower == upper) return values[lower];
            double weight = position - lower;
            return values[lower] * (1.0 - weight) + values[upper] * weight;
        }

        private static double Median(IEnumerable<double> source) => Quantile(source, 0.5);
        private static D3 MedianPoint(IEnumerable<D3> source)
        {
            D3[] values = source.ToArray();
            return new D3(Median(values.Select(v => v.x)), Median(values.Select(v => v.y)), Median(values.Select(v => v.z)));
        }
        private static double TurnAngle(D3 incoming, D3 outgoing)
        {
            double left = incoming.Norm(), right = outgoing.Norm();
            if (left <= 1e-12 || right <= 1e-12) return 0.0;
            double dot = Math.Max(-1.0, Math.Min(1.0, incoming.Dot(outgoing) / (left * right)));
            return Math.Acos(dot) * 180.0 / Math.PI;
        }
        private static double AxisAngle(D3 left, D3 right)
        {
            double a = left.Norm(), b = right.Norm();
            if (a <= 1e-12 || b <= 1e-12) return 90.0;
            double dot = Math.Max(-1.0, Math.Min(1.0, Math.Abs(left.Dot(right) / (a * b))));
            return Math.Acos(dot) * 180.0 / Math.PI;
        }
        private static double Chebyshev(D3 value) => Math.Max(Math.Abs(value.x), Math.Max(Math.Abs(value.y), Math.Abs(value.z)));
        private static double DistanceSquared(D3 a, D3 b) => Square(a.x - b.x) + Square(a.y - b.y) + Square(a.z - b.z);
        private static double Square(double value) => value * value;
        private static D3 ToD3(SparseVoxel value) => new D3(value.x, value.y, value.z);
        private static Int3Key Key(SparseVoxel value) => new Int3Key(value.x, value.y, value.z);
        private static Int3Key Key(D3 value) => new Int3Key((int)value.x, (int)value.y, (int)value.z);
    }
}

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace VegetationAssurance.V10
{
    public static class V10NativeOutputWriter
    {
        public static readonly string[] PoleColumns = {
            "file_id","component_id","slice_seq","refiner_probability","touches_xy_edge","radius_p90_ft",
            "verticality","base_x","base_y","base_z","top_x","top_y","top_z","height_ft","tilt_ft"
        };
        public static readonly string[] LineColumns = {
            "file_id","component_id","slice_seq","refiner_probability","horizontal_span_ft","vertical_span_ft",
            "verticality","tortuosity","vertex_count"
        };
        public static readonly string[] VertexColumns = {
            "file_id","component_id","slice_seq","vertex_index","x","y","z"
        };
        public static readonly string[] VoxelColumns = {
            "x","y","z","v4_pole_score","v4_line_score","v4_semantic_head","v4_deployed_label"
        };
        public static readonly string[] TrackColumns = {
            "component_id","source_component_index","raw_fragment_count","n_voxels","vertex_count","bridge_count",
            "pole_attachment_count","max_bridge_gap_ft","track_radius_p95_ft","horizontal_span_ft","vertical_span_ft",
            "score_mean","max_turn_deg","geometry_support_fraction"
        };
        public static readonly string[] AttachmentColumns = {
            "component_id","track_end","pole_component_id","endpoint_distance_to_pole_axis_ft",
            "endpoint_to_pole_angle_deg","attachment_height_fraction","anchor_x","anchor_y","anchor_z",
            "pole_surface_radius_ft","contact_pole_voxel_x","contact_pole_voxel_y","contact_pole_voxel_z",
            "contact_distance_vox","attachment_support_mode"
        };
        public static readonly string[] BridgeColumns = {
            "fragment_a","fragment_b","a_endpoint_index","b_endpoint_index","gap_ft","longitudinal_gap_ft",
            "longitudinal_overlap_ft","lane_center_offset_ft","endpoint_lateral_jump_ft","axis_angle_deg",
            "bridge_angle_a_deg","bridge_angle_b_deg","vertical_gap_ft","near_pole_bridge_guard",
            "stage1_voxel_support_fraction","stage1_voxel_support_samples","stage1_voxel_support_total_samples",
            "passed","reject_reason","all_failed_reasons","selected","selection_reject_reason"
        };

        private static readonly string[] ComponentBaseColumns = {
            "component_id","class_name","n_voxels","score_mean","score_std","score_p10","score_p50","score_p90",
            "score_max","vertical_head_mean","horizontal_head_mean","x_span_ft","y_span_ft","z_span_ft",
            "horizontal_span_ft","bbox_density","center_x","center_y","center_z","center_z_ft","min_z_ft","max_z_ft",
            "principal_dx","principal_dy","principal_dz","principal_verticality","linearity","planarity","scattering",
            "radius_p50_ft","radius_p90_ft","xy_path_length_ft","xy_endpoint_distance_ft","xy_tortuosity",
            "quadratic_rmse_ft","sag_estimate_ft","endpoint1_x","endpoint1_y","endpoint1_z","endpoint2_x","endpoint2_y",
            "endpoint2_z","exact_gt_fraction","near_gt_fraction","edge_distance_vox","touches_xy_edge",
            "partial_pole_candidate","refiner_probability","physical_ok","component_accept","accept_mode","file_id","slice_seq",
            "stage1_line_label_fraction","raw_fragment_count","bridge_count","pole_attachment_count","max_bridge_gap_ft",
            "track_radius_p95_ft","synthetic_line_voxels","runtime_gt_usage","source_component_index"
        };

        public static string ObjectDirectory(string runRoot, string sid, string relativePath)
        {
            string normalized = (relativePath ?? "slice.csv").Replace('\\', '/').TrimStart('/');
            string parent = Path.GetDirectoryName(normalized)?.Replace('\\', Path.DirectorySeparatorChar) ?? "";
            return Path.Combine(runRoot, "stage2", sid, "stage2_objects", parent);
        }

        public static void WriteSlice(string runRoot, string sid, string relativePath,
            Stage1SliceResult stage1, Stage2SliceResult stage2, IReadOnlyDictionary<string, object> metadata)
        {
            if (stage1 == null || !stage1.success) throw new InvalidOperationException("Stage 1 failed.");
            if (stage2 == null || !stage2.success) throw new InvalidOperationException("Stage 2 failed: " + stage2?.error);
            string directory = ObjectDirectory(runRoot, sid, relativePath);
            Directory.CreateDirectory(directory);
            string stem = Path.GetFileNameWithoutExtension(relativePath ?? "slice.csv");

            AtomicCsv(Path.Combine(directory, stem + "_poles.csv"), PoleColumns, stage2.poles);
            AtomicCsv(Path.Combine(directory, stem + "_line_segments.csv"), LineColumns, stage2.lines);
            AtomicCsv(Path.Combine(directory, stem + "_line_vertices.csv"), VertexColumns, stage2.vertices);
            AtomicCsv(Path.Combine(directory, stem + "_components.csv"), ComponentBaseColumns, stage2.components);
            AtomicCsv(Path.Combine(directory, stem + "_stage1_electrical_tracks.csv"), TrackColumns, stage2.tracks);
            AtomicCsv(Path.Combine(directory, stem + "_pole_attachments.csv"), AttachmentColumns, stage2.attachments);
            AtomicCsv(Path.Combine(directory, stem + "_selected_fragment_bridges.csv"), BridgeColumns,
                Array.Empty<Dictionary<string, object>>());
            AtomicCsv(Path.Combine(directory, stem + "_fragment_bridge_candidates.csv"), BridgeColumns,
                Array.Empty<Dictionary<string, object>>());
            AtomicCsv(Path.Combine(directory, stem + "_stage1_line_voxels.csv"), VoxelColumns,
                VoxelRows(stage1, stage2.stage1LineRows));
            AtomicCsv(Path.Combine(directory, stem + "_accepted_line_voxels.csv"), VoxelColumns,
                VoxelRows(stage1, stage2.acceptedLineRows));

            var audit = new Dictionary<string, object>(StringComparer.Ordinal);
            foreach (FieldInfo field in typeof(Stage2ElectricalAudit).GetFields(BindingFlags.Instance | BindingFlags.Public))
                audit[field.Name] = field.GetValue(stage2.audit);
            audit["contract_version"] = "v4-stage12-unity-native-1";
            audit["stage"] = 2;
            audit["group_id"] = metadata != null && metadata.TryGetValue("group_id", out object gid) ? gid : "";
            audit["id"] = stage2.fileId;
            audit["slice_seq"] = stage2.sliceSeq;
            audit["runtime"] = "Unity 6.3 LTS / Sentis 2.6.1 / native C# Stage2";
            AtomicText(Path.Combine(directory, stem + "_stage1_electrical_track_audit.json"), V10Json.Serialize(audit, true));
        }

        private static IEnumerable<Dictionary<string, object>> VoxelRows(Stage1SliceResult result, IEnumerable<int> indices)
        {
            foreach (int i in indices)
            {
                SparseVoxel v = result.voxels[i];
                yield return new Dictionary<string, object>
                {
                    ["x"] = v.x, ["y"] = v.y, ["z"] = v.z,
                    ["v4_pole_score"] = result.poleScore[i], ["v4_line_score"] = result.lineScore[i],
                    ["v4_semantic_head"] = 0, ["v4_deployed_label"] = result.deployedLabel[i]
                };
            }
        }

        public static void AtomicCsv(string path, IReadOnlyList<string> columns,
            IEnumerable<Dictionary<string, object>> rows)
        {
            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full) ?? ".");
            string temporary = full + ".tmp." + Guid.NewGuid().ToString("N");
            try
            {
                using (var writer = new StreamWriter(temporary, false, new UTF8Encoding(false)))
                {
                    writer.WriteLine(string.Join(",", columns.Select(value => V10Csv.Escape(value))));
                    foreach (Dictionary<string, object> row in rows)
                    {
                        for (int i = 0; i < columns.Count; i++)
                        {
                            if (i > 0) writer.Write(',');
                            row.TryGetValue(columns[i], out object value);
                            writer.Write(V10Csv.Escape(value));
                        }
                        writer.WriteLine();
                    }
                }
                V10Stage12SentisInferenceManager.AtomicReplace(temporary, full);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }

        public static void AtomicText(string path, string text)
        {
            string full = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(full) ?? ".");
            string temporary = full + ".tmp." + Guid.NewGuid().ToString("N");
            try
            {
                File.WriteAllText(temporary, text, new UTF8Encoding(false));
                V10Stage12SentisInferenceManager.AtomicReplace(temporary, full);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
    }

    internal static class V10Json
    {
        public static string Serialize(object value, bool pretty = false)
        {
            var output = new StringBuilder();
            Write(output, value, pretty, 0);
            output.Append('\n');
            return output.ToString();
        }

        private static void Write(StringBuilder output, object value, bool pretty, int depth)
        {
            if (value == null) { output.Append("null"); return; }
            if (value is string s) { Quote(output, s); return; }
            if (value is bool b) { output.Append(b ? "true" : "false"); return; }
            if (value is float f) { Number(output, f); return; }
            if (value is double d) { Number(output, d); return; }
            if (value is decimal || value is byte || value is sbyte || value is short || value is ushort ||
                value is int || value is uint || value is long || value is ulong)
            { output.Append(Convert.ToString(value, CultureInfo.InvariantCulture)); return; }
            if (value is IDictionary dictionary)
            {
                output.Append('{'); bool first = true;
                foreach (DictionaryEntry item in dictionary)
                {
                    if (!first) output.Append(','); Newline(output, pretty, depth + 1); first = false;
                    Quote(output, Convert.ToString(item.Key, CultureInfo.InvariantCulture)); output.Append(pretty ? ": " : ":");
                    Write(output, item.Value, pretty, depth + 1);
                }
                if (!first) Newline(output, pretty, depth);
                output.Append('}'); return;
            }
            if (value is IEnumerable enumerable)
            {
                output.Append('['); bool first = true;
                foreach (object item in enumerable)
                {
                    if (!first) output.Append(','); Newline(output, pretty, depth + 1); first = false;
                    Write(output, item, pretty, depth + 1);
                }
                if (!first) Newline(output, pretty, depth);
                output.Append(']'); return;
            }
            var fields = new Dictionary<string, object>();
            foreach (FieldInfo field in value.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public))
                fields[field.Name] = field.GetValue(value);
            Write(output, fields, pretty, depth);
        }

        private static void Number(StringBuilder output, double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) output.Append("null");
            else output.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }
        private static void Quote(StringBuilder output, string text)
        {
            output.Append('"');
            foreach (char c in text ?? "")
            {
                switch (c)
                {
                    case '"': output.Append("\\\""); break;
                    case '\\': output.Append("\\\\"); break;
                    case '\n': output.Append("\\n"); break;
                    case '\r': output.Append("\\r"); break;
                    case '\t': output.Append("\\t"); break;
                    default: if (c < 32) output.Append("\\u" + ((int)c).ToString("x4")); else output.Append(c); break;
                }
            }
            output.Append('"');
        }
        private static void Newline(StringBuilder output, bool pretty, int depth)
        {
            if (!pretty) return;
            output.Append('\n').Append(' ', depth * 2);
        }
    }
}

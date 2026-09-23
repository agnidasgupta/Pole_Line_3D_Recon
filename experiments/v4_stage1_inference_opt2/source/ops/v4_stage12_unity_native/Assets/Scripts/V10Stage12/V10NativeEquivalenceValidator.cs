using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace VegetationAssurance.V10
{
    /// <summary>Automated release gate against accepted opt1-fix2 outputs.</summary>
    public static class V10NativeEquivalenceValidator
    {
        private static readonly string[] ComparedSuffixes = {
            "_poles.csv", "_line_segments.csv", "_line_vertices.csv",
            "_stage1_line_voxels.csv", "_accepted_line_voxels.csv",
            "_stage1_electrical_tracks.csv", "_pole_attachments.csv",
            "_selected_fragment_bridges.csv", "_fragment_bridge_candidates.csv"
        };

        public static bool Validate(string actualRoot, string referenceRoot, string reportPath,
            double numericTolerance = 3e-5)
        {
            actualRoot = Path.GetFullPath(actualRoot);
            referenceRoot = Path.GetFullPath(referenceRoot);
            var errors = new List<string>();
            int comparedFiles = 0, comparedRows = 0, audits = 0;
            if (!Directory.Exists(referenceRoot)) errors.Add("Reference root does not exist: " + referenceRoot);
            else
            {
                foreach (string reference in Directory.GetFiles(referenceRoot, "*", SearchOption.AllDirectories)
                    .Where(p => ComparedSuffixes.Any(s => p.EndsWith(s, StringComparison.Ordinal))))
                {
                    string relative = Path.GetRelativePath(referenceRoot, reference);
                    string actual = Path.Combine(actualRoot, relative);
                    if (!File.Exists(actual)) { errors.Add("missing: " + relative); continue; }
                    comparedFiles++;
                    try { comparedRows += CompareCsv(reference, actual, relative, numericTolerance, errors); }
                    catch (Exception exception) { errors.Add(relative + ": comparison error: " + exception.Message); }
                }
            }

            foreach (string audit in Directory.Exists(actualRoot)
                ? Directory.GetFiles(actualRoot, "*_stage1_electrical_track_audit.json", SearchOption.AllDirectories)
                : Array.Empty<string>())
            {
                audits++;
                string text = File.ReadAllText(audit);
                RequireNumber(text, audit, "stage1_to_stage2_voxel_preservation", 1.0, errors);
                RequireNumber(text, audit, "geometry_stage1_voxel_support_fraction", 1.0, errors);
                RequireNumber(text, audit, "geometry_outside_stage1_voxel_samples", 0.0, errors);
                RequireNumber(text, audit, "selected_fragment_bridges", 0.0, errors);
                RequireBool(text, audit, "disconnected_fragment_bridges_allowed", false, errors);
                RequireBool(text, audit, "open_line_endpoints_preserved", true, errors);
                RequireBool(text, audit, "pole_attachment_requires_stage1_voxel_contact", true, errors);
            }
            ValidateAttachmentUniqueness(actualRoot, errors);
            if (comparedFiles == 0) errors.Add("No reference Stage2 CSV files were compared.");
            if (audits == 0) errors.Add("No Unity Stage2 audit JSON files were found.");

            var report = new StringBuilder();
            report.AppendLine("V10 UNITY NATIVE EQUIVALENCE GATE");
            report.AppendLine("actual_root=" + actualRoot);
            report.AppendLine("reference_root=" + referenceRoot);
            report.AppendLine("numeric_tolerance=" + numericTolerance.ToString("R", CultureInfo.InvariantCulture));
            report.AppendLine("compared_files=" + comparedFiles);
            report.AppendLine("compared_rows=" + comparedRows);
            report.AppendLine("electrical_audits=" + audits);
            report.AppendLine("errors=" + errors.Count);
            report.AppendLine("status=" + (errors.Count == 0 ? "PASS" : "FAIL"));
            foreach (string error in errors.Take(500)) report.AppendLine("ERROR " + error);
            if (errors.Count > 500) report.AppendLine("ERROR ... " + (errors.Count - 500) + " additional errors omitted");
            V10NativeOutputWriter.AtomicText(reportPath, report.ToString());
            return errors.Count == 0;
        }

        private static int CompareCsv(string expectedPath, string actualPath, string relative,
            double tolerance, List<string> errors)
        {
            List<List<string>> expected = ReadCsv(expectedPath);
            List<List<string>> actual = ReadCsv(actualPath);
            if (expected.Count == 0 || actual.Count == 0) { errors.Add(relative + ": missing header"); return 0; }
            Dictionary<string, int> eh = V10Csv.Header(expected[0]), ah = V10Csv.Header(actual[0]);
            foreach (string name in eh.Keys)
                if (!ah.ContainsKey(name)) errors.Add(relative + ": actual lacks expected column " + name);
            if (expected.Count != actual.Count)
            {
                errors.Add(relative + ": row count expected=" + (expected.Count - 1) + " actual=" + (actual.Count - 1));
                return Math.Min(expected.Count, actual.Count) - 1;
            }
            int rows = expected.Count - 1;
            for (int r = 1; r < expected.Count; r++)
            foreach (KeyValuePair<string, int> item in eh)
            {
                // The Stage1 ONNX contract intentionally exports only the two
                // calibrated scores. The raw semantic-head diagnostic is not an
                // input to deployed labels or Stage2 and is therefore non-contractual.
                if (item.Key.Equals("v4_semantic_head", StringComparison.OrdinalIgnoreCase)) continue;
                if (!ah.TryGetValue(item.Key, out int ai)) continue;
                string left = item.Value < expected[r].Count ? expected[r][item.Value] : "";
                string right = ai < actual[r].Count ? actual[r][ai] : "";
                if (Equivalent(left, right, tolerance)) continue;
                errors.Add(relative + ": row=" + r + " column=" + item.Key + " expected=" + left + " actual=" + right);
                if (errors.Count >= 500) return rows;
            }
            return rows;
        }

        private static List<List<string>> ReadCsv(string path)
        {
            var rows = new List<List<string>>();
            using var reader = new StreamReader(path, Encoding.UTF8, true);
            string line;
            while ((line = reader.ReadLine()) != null) rows.Add(V10Csv.ParseLine(line));
            return rows;
        }

        private static bool Equivalent(string left, string right, double tolerance)
        {
            if (StringComparer.Ordinal.Equals(left, right)) return true;
            bool ln = double.TryParse(left, NumberStyles.Float, CultureInfo.InvariantCulture, out double a);
            bool rn = double.TryParse(right, NumberStyles.Float, CultureInfo.InvariantCulture, out double b);
            if (!ln || !rn) return false;
            if (double.IsNaN(a) && double.IsNaN(b)) return true;
            return Math.Abs(a - b) <= tolerance * Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(b)));
        }

        private static void RequireNumber(string text, string path, string key, double expected, List<string> errors)
        {
            Match match = Regex.Match(text, "\\\"" + Regex.Escape(key) + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
            if (!match.Success || !double.TryParse(match.Groups[1].Value, NumberStyles.Float,
                    CultureInfo.InvariantCulture, out double actual) || Math.Abs(actual - expected) > 1e-12)
                errors.Add(Path.GetFileName(path) + ": invariant " + key + " must equal " + expected);
        }

        private static void RequireBool(string text, string path, string key, bool expected, List<string> errors)
        {
            Match match = Regex.Match(text, "\\\"" + Regex.Escape(key) + "\\\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
            if (!match.Success || bool.Parse(match.Groups[1].Value) != expected)
                errors.Add(Path.GetFileName(path) + ": invariant " + key + " must equal " + expected);
        }

        private static void ValidateAttachmentUniqueness(string root, List<string> errors)
        {
            if (!Directory.Exists(root)) return;
            foreach (string path in Directory.GetFiles(root, "*_pole_attachments.csv", SearchOption.AllDirectories))
            {
                List<List<string>> rows = ReadCsv(path);
                if (rows.Count == 0) continue;
                Dictionary<string, int> h = V10Csv.Header(rows[0]);
                string[] needed = { "component_id", "pole_component_id", "anchor_x", "anchor_y", "anchor_z" };
                if (needed.Any(k => !h.ContainsKey(k))) { errors.Add(path + ": attachment columns missing"); continue; }
                var seen = new Dictionary<string, string>(StringComparer.Ordinal);
                for (int r = 1; r < rows.Count; r++)
                {
                    string cid = rows[r][h["component_id"]];
                    string key = rows[r][h["pole_component_id"]] + "|" + Round(rows[r][h["anchor_x"]]) + "|" +
                        Round(rows[r][h["anchor_y"]]) + "|" + Round(rows[r][h["anchor_z"]]);
                    if (seen.TryGetValue(key, out string prior) && prior != cid)
                        errors.Add(path + ": different conductors share pole attachment " + key + " (" + prior + "," + cid + ")");
                    else seen[key] = cid;
                }
            }
        }

        private static string Round(string value)
        {
            return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed)
                ? Math.Round(parsed, 6, MidpointRounding.ToEven).ToString("R", CultureInfo.InvariantCulture) : value;
        }
    }
}

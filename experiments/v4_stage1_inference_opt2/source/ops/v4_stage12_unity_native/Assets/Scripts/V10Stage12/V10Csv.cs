using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace VegetationAssurance.V10
{
    internal static class V10Csv
    {
        public static List<string> ParseLine(string line)
        {
            var output = new List<string>();
            var field = new StringBuilder();
            bool quoted = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (quoted)
                {
                    if (c == '"' && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        field.Append('"');
                        i++;
                    }
                    else if (c == '"') quoted = false;
                    else field.Append(c);
                }
                else if (c == '"') quoted = true;
                else if (c == ',')
                {
                    output.Add(field.ToString());
                    field.Clear();
                }
                else field.Append(c);
            }
            if (quoted) throw new InvalidDataException("Unterminated quoted CSV field.");
            output.Add(field.ToString());
            return output;
        }

        public static string Escape(object value)
        {
            if (value == null) return "";
            string text;
            switch (value)
            {
                case double d:
                    text = double.IsNaN(d) ? "" : d.ToString("R", CultureInfo.InvariantCulture);
                    break;
                case float f:
                    text = float.IsNaN(f) ? "" : f.ToString("R", CultureInfo.InvariantCulture);
                    break;
                case bool b:
                    text = b ? "True" : "False";
                    break;
                case IFormattable formattable:
                    text = formattable.ToString(null, CultureInfo.InvariantCulture);
                    break;
                default:
                    text = value.ToString() ?? "";
                    break;
            }
            if (text.IndexOfAny(new[] { ',', '"', '\r', '\n' }) < 0) return text;
            return "\"" + text.Replace("\"", "\"\"") + "\"";
        }

        public static Dictionary<string, int> Header(IReadOnlyList<string> values)
        {
            var output = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < values.Count; i++) output[values[i].Trim()] = i;
            return output;
        }

        public static string At(IReadOnlyList<string> row, Dictionary<string, int> header, string name)
        {
            if (!header.TryGetValue(name, out int index) || index < 0 || index >= row.Count) return null;
            return row[index];
        }

        public static int RequiredInt(IReadOnlyList<string> row, Dictionary<string, int> header, string name)
        {
            string text = At(row, header, name);
            if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
                throw new InvalidDataException("Invalid integer column " + name + ": " + text);
            return value;
        }

        public static int RequiredCoordinate(IReadOnlyList<string> row, Dictionary<string, int> header, string name)
        {
            string text = At(row, header, name);
            if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ||
                double.IsNaN(value) || double.IsInfinity(value))
                throw new InvalidDataException("Invalid coordinate column " + name + ": " + text);
            double rounded = Math.Round(value, MidpointRounding.ToEven);
            if (rounded < int.MinValue || rounded > int.MaxValue)
                throw new InvalidDataException("Coordinate outside Int32 range in " + name + ": " + text);
            return (int)rounded;
        }

        public static bool TryCoordinate(IReadOnlyList<string> row, Dictionary<string, int> header, string name, out int result)
        {
            result = 0;
            string text = At(row, header, name);
            if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ||
                double.IsNaN(value) || double.IsInfinity(value)) return false;
            double rounded = Math.Round(value, MidpointRounding.ToEven);
            if (rounded < int.MinValue || rounded > int.MaxValue) return false;
            result = (int)rounded;
            return true;
        }

        public static float OptionalFloat(IReadOnlyList<string> row, Dictionary<string, int> header,
            string name, float fallback = 0f)
        {
            string text = At(row, header, name);
            if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)) return value;
            switch ((text ?? "").Trim().ToLowerInvariant())
            {
                case "nan": return float.NaN;
                case "inf": case "+inf": case "infinity": case "+infinity": return float.PositiveInfinity;
                case "-inf": case "-infinity": return float.NegativeInfinity;
                default: return fallback;
            }
        }

        public static long OptionalLong(IReadOnlyList<string> row, Dictionary<string, int> header,
            string name, long fallback)
        {
            string text = At(row, header, name);
            return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value)
                ? value : fallback;
        }
    }

    public static class V10SparseVoxelCsvReader
    {
        public static List<SparseVoxel> Read(string path)
        {
            using TextReader reader = OpenText(path);
            string first = reader.ReadLine();
            if (first == null) throw new InvalidDataException("Voxel CSV is empty: " + path);
            Dictionary<string, int> header = V10Csv.Header(V10Csv.ParseLine(first));
            foreach (string required in new[] { "x", "y", "z" })
                if (!header.ContainsKey(required)) throw new InvalidDataException("Voxel CSV lacks column " + required);
            string distanceColumn = header.ContainsKey("dist_center_ft") ? "dist_center_ft" :
                header.ContainsKey("dist_values") ? "dist_values" : null;
            var output = new List<SparseVoxel>();
            string line;
            long sourceRow = 0;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0) { sourceRow++; continue; }
                List<string> row = V10Csv.ParseLine(line);
                if (!V10Csv.TryCoordinate(row, header, "x", out int x) ||
                    !V10Csv.TryCoordinate(row, header, "y", out int y) ||
                    !V10Csv.TryCoordinate(row, header, "z", out int z))
                { sourceRow++; continue; }
                float distance = distanceColumn == null ? 0f : V10Csv.OptionalFloat(row, header, distanceColumn);
                long retainedRow = V10Csv.OptionalLong(row, header, "source_row", sourceRow);
                output.Add(new SparseVoxel(x, y, z, distance, retainedRow));
                sourceRow++;
            }
            return output;
        }

        public static Stage1SliceResult ReadStage1(string path)
        {
            using TextReader reader = OpenText(path);
            string first = reader.ReadLine();
            if (first == null) throw new InvalidDataException("Stage-1 CSV is empty: " + path);
            Dictionary<string, int> header = V10Csv.Header(V10Csv.ParseLine(first));
            foreach (string required in new[] { "x", "y", "z", "pole", "line", "deployed_label" })
                if (!header.ContainsKey(required)) throw new InvalidDataException("Stage-1 CSV lacks column " + required);
            var voxels = new List<SparseVoxel>();
            var distance = new List<float>();
            var pole = new List<float>();
            var lineScore = new List<float>();
            var labels = new List<byte>();
            string line;
            long sourceRow = 0;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0) { sourceRow++; continue; }
                List<string> row = V10Csv.ParseLine(line);
                int x = V10Csv.RequiredCoordinate(row, header, "x");
                int y = V10Csv.RequiredCoordinate(row, header, "y");
                int z = V10Csv.RequiredCoordinate(row, header, "z");
                float dist = V10Csv.OptionalFloat(row, header, "dist_values");
                long sr = V10Csv.OptionalLong(row, header, "source_row", sourceRow);
                int deployed = V10Csv.RequiredInt(row, header, "deployed_label");
                if (deployed < 0 || deployed > 2) throw new InvalidDataException("deployed_label must be 0, 1, or 2.");
                voxels.Add(new SparseVoxel(x, y, z, dist, sr));
                distance.Add(dist);
                pole.Add(V10Csv.OptionalFloat(row, header, "pole"));
                lineScore.Add(V10Csv.OptionalFloat(row, header, "line"));
                labels.Add((byte)deployed);
                sourceRow++;
            }
            return new Stage1SliceResult
            {
                success = true,
                voxels = voxels.ToArray(),
                normalizedDistance = distance.ToArray(),
                poleScore = pole.ToArray(),
                lineScore = lineScore.ToArray(),
                deployedLabel = labels.ToArray()
            };
        }

        private static TextReader OpenText(string path)
        {
            FileStream file = File.OpenRead(path);
            try
            {
                Stream stream = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase)
                    ? (Stream)new GZipStream(file, CompressionMode.Decompress, false) : file;
                return new StreamReader(stream, Encoding.UTF8, true);
            }
            catch
            {
                file.Dispose();
                throw;
            }
        }
    }
}

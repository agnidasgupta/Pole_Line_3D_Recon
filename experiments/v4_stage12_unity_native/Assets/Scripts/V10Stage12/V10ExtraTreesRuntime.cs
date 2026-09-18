using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace VegetationAssurance.V10
{
    public sealed class V10ExtraTreesBundle
    {
        private sealed class Tree
        {
            public int[] left;
            public int[] right;
            public int[] feature;
            public double[] threshold;
            public double[] positiveProbability;

            public double Predict(double[] values)
            {
                int node = 0;
                int guard = left.Length + 1;
                while (left[node] >= 0 && right[node] >= 0)
                {
                    if (--guard <= 0) throw new InvalidDataException("ExtraTrees cycle detected.");
                    int column = feature[node];
                    if (column < 0 || column >= values.Length)
                        throw new InvalidDataException("ExtraTrees feature index is outside the feature vector.");
                    node = values[column] <= threshold[node] ? left[node] : right[node];
                    if (node < 0 || node >= left.Length)
                        throw new InvalidDataException("ExtraTrees child index is invalid.");
                }
                return positiveProbability[node];
            }
        }

        private sealed class Forest
        {
            public Tree[] trees;
            public double Predict(double[] values)
            {
                if (trees == null || trees.Length == 0)
                    throw new InvalidDataException("ExtraTrees forest contains no trees.");
                double sum = 0.0;
                foreach (Tree tree in trees) sum += tree.Predict(values);
                return sum / trees.Length;
            }
        }

        private readonly string[] featureColumns;
        private readonly Dictionary<string, int> featureIndex;
        private readonly Forest pole;
        private readonly Forest line;

        public double PoleThreshold { get; }
        public double LineThreshold { get; }
        public IReadOnlyList<string> FeatureColumns => featureColumns;

        private V10ExtraTreesBundle(string[] columns, double poleThreshold, double lineThreshold,
            Forest pole, Forest line)
        {
            featureColumns = columns;
            featureIndex = new Dictionary<string, int>(columns.Length, StringComparer.Ordinal);
            for (int i = 0; i < columns.Length; i++)
            {
                if (featureIndex.ContainsKey(columns[i]))
                    throw new InvalidDataException("Duplicate feature column: " + columns[i]);
                featureIndex.Add(columns[i], i);
            }
            PoleThreshold = poleThreshold;
            LineThreshold = lineThreshold;
            this.pole = pole;
            this.line = line;
        }

        public static V10ExtraTreesBundle Load(byte[] bytes, string expectedSourceSha256)
        {
            if (bytes == null || bytes.Length == 0) throw new ArgumentException("Tree asset is empty.", nameof(bytes));
            using var stream = new MemoryStream(bytes, false);
            using var reader = new BinaryReader(stream, Encoding.UTF8, false);
            byte[] magic = reader.ReadBytes(8);
            if (Encoding.ASCII.GetString(magic) != "V10ETB1\0")
                throw new InvalidDataException("Invalid V10 ExtraTrees asset magic.");
            uint version = reader.ReadUInt32();
            if (version != 2) throw new InvalidDataException("Unsupported V10 ExtraTrees asset version=" + version);
            byte[] sourceDigest = reader.ReadBytes(32);
            if (sourceDigest.Length != 32) throw new EndOfStreamException();
            string actualSourceSha256 = BitConverter.ToString(sourceDigest).Replace("-", "").ToLowerInvariant();
            if (string.IsNullOrWhiteSpace(expectedSourceSha256) ||
                !StringComparer.OrdinalIgnoreCase.Equals(actualSourceSha256, expectedSourceSha256.Trim()))
                throw new InvalidDataException("Stage2 refiner tree SHA256 does not match the sidecar.");
            uint featureCount = ReadCount(reader, "feature count", 4096);
            var columns = new string[(int)featureCount];
            for (int i = 0; i < columns.Length; i++) columns[i] = ReadString(reader);
            double poleThreshold = reader.ReadDouble();
            double lineThreshold = reader.ReadDouble();
            Forest pole = ReadForest(reader);
            Forest line = ReadForest(reader);
            if (stream.Position != stream.Length)
                throw new InvalidDataException("Trailing data in V10 ExtraTrees asset.");
            return new V10ExtraTreesBundle(columns, poleThreshold, lineThreshold, pole, line);
        }

        public double[] BuildFeatureVector(IReadOnlyDictionary<string, double> features)
        {
            var output = new double[featureColumns.Length];
            for (int i = 0; i < output.Length; i++)
            {
                if (features.TryGetValue(featureColumns[i], out double value) && !double.IsNaN(value) && !double.IsInfinity(value))
                    output[i] = value;
            }
            return output;
        }

        public double PredictPole(IReadOnlyDictionary<string, double> features) => pole.Predict(BuildFeatureVector(features));
        public double PredictLine(IReadOnlyDictionary<string, double> features) => line.Predict(BuildFeatureVector(features));

        private static Forest ReadForest(BinaryReader reader)
        {
            uint treeCount = ReadCount(reader, "tree count", 100000);
            var trees = new Tree[(int)treeCount];
            for (int t = 0; t < trees.Length; t++)
            {
                uint nodeCount = ReadCount(reader, "node count", 100000000);
                var tree = new Tree
                {
                    left = new int[(int)nodeCount],
                    right = new int[(int)nodeCount],
                    feature = new int[(int)nodeCount],
                    threshold = new double[(int)nodeCount],
                    positiveProbability = new double[(int)nodeCount]
                };
                for (int i = 0; i < nodeCount; i++) tree.left[i] = reader.ReadInt32();
                for (int i = 0; i < nodeCount; i++) tree.right[i] = reader.ReadInt32();
                for (int i = 0; i < nodeCount; i++) tree.feature[i] = reader.ReadInt32();
                for (int i = 0; i < nodeCount; i++) tree.threshold[i] = reader.ReadDouble();
                for (int i = 0; i < nodeCount; i++) tree.positiveProbability[i] = reader.ReadDouble();
                trees[t] = tree;
            }
            return new Forest { trees = trees };
        }

        private static uint ReadCount(BinaryReader reader, string name, uint maximum)
        {
            uint value = reader.ReadUInt32();
            if (value > maximum) throw new InvalidDataException("Unreasonable " + name + ": " + value);
            return value;
        }

        private static string ReadString(BinaryReader reader)
        {
            uint length = ReadCount(reader, "string length", 1024 * 1024);
            byte[] data = reader.ReadBytes((int)length);
            if (data.Length != length) throw new EndOfStreamException();
            return Encoding.UTF8.GetString(data);
        }
    }
}

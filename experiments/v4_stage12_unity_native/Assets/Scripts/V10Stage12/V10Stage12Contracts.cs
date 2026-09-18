using System;
using System.Collections.Generic;

namespace VegetationAssurance.V10
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

    public sealed class Stage1SliceResult
    {
        public bool success;
        public string error;
        public SparseVoxel[] voxels = Array.Empty<SparseVoxel>();
        public float[] normalizedDistance = Array.Empty<float>();
        public float[] poleScore = Array.Empty<float>();
        public float[] lineScore = Array.Empty<float>();
        public byte[] deployedLabel = Array.Empty<byte>();
        public int activeCores;
        public double prepareSeconds;
        public double scheduleSeconds;
        public double readbackSeconds;
        public double totalSeconds;
    }

    public readonly struct Int3Key : IEquatable<Int3Key>, IComparable<Int3Key>
    {
        public readonly int x;
        public readonly int y;
        public readonly int z;

        public Int3Key(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public bool Equals(Int3Key other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is Int3Key other && Equals(other);
        public override int GetHashCode()
        {
            unchecked { return ((x * 397) ^ y) * 397 ^ z; }
        }

        public int CompareTo(Int3Key other)
        {
            int c = z.CompareTo(other.z);
            if (c != 0) return c;
            c = y.CompareTo(other.y);
            return c != 0 ? c : x.CompareTo(other.x);
        }
    }

    public readonly struct D3
    {
        public readonly double x;
        public readonly double y;
        public readonly double z;

        public D3(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static D3 operator +(D3 a, D3 b) => new D3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static D3 operator -(D3 a, D3 b) => new D3(a.x - b.x, a.y - b.y, a.z - b.z);
        public static D3 operator *(D3 a, double b) => new D3(a.x * b, a.y * b, a.z * b);
        public static D3 operator /(D3 a, double b) => new D3(a.x / b, a.y / b, a.z / b);
        public double Dot(D3 b) => x * b.x + y * b.y + z * b.z;
        public double Norm() => Math.Sqrt(Dot(this));
        public D3 Unit(D3 fallback)
        {
            double n = Norm();
            return n > 1e-12 ? this / n : fallback;
        }
    }

    [Serializable]
    public sealed class V10Stage2Profile
    {
        public int centerline_coverage_radius_vox = 1;
        public float max_local_turn_deg = 60f;
        public int turn_tangent_window_vox = 3;
        public float max_supported_chord_vox = 4f;
        public int max_supported_chord_lookahead = 24;
        public float support_sample_step_vox = 0.25f;
        public float pole_attach_max_angle_deg = 45f;
        public float pole_attach_min_height_fraction = 0.35f;
        public float pole_surface_standoff_min_ft = 0.5f;
        public float pole_contact_max_chebyshev_vox = 1f;
        public bool disconnected_fragment_bridge_allowed = false;
        public bool line_geometry_must_stay_inside_stage1_voxel_cells = true;
        public bool open_line_endpoints_preserved = true;
        public bool pole_attachment_requires_stage1_voxel_contact = true;
    }

    [Serializable]
    public sealed class V10SidecarInput
    {
        public int[] grid_size_xyz = { 400, 400, 200 };
    }

    [Serializable]
    public sealed class V10SidecarCalibration
    {
        public float pole_threshold = 0.2125f;
        public float line_threshold = 0.2125f;
    }

    [Serializable]
    public sealed class V10SidecarOnnx
    {
        public bool fp16_deployable;
    }

    [Serializable]
    public sealed class V10ArtifactMetadata
    {
        public long bytes;
        public string sha256;
    }

    [Serializable]
    public sealed class V10SidecarStage2
    {
        public string algorithm;
        public bool disconnected_fragment_bridges_allowed;
        public bool line_geometry_must_stay_inside_stage1_voxel_cells;
        public bool open_line_endpoints_preserved;
        public bool pole_attachment_requires_direct_stage1_contact;
        public V10ArtifactMetadata refiner_tree_metadata = new V10ArtifactMetadata();
        public V10Stage2Profile profile = new V10Stage2Profile();
    }

    [Serializable]
    public sealed class V10Sidecar
    {
        public int format_version;
        public V10SidecarOnnx onnx = new V10SidecarOnnx();
        public V10SidecarInput input = new V10SidecarInput();
        public V10SidecarCalibration calibration = new V10SidecarCalibration();
        public V10SidecarStage2 stage2 = new V10SidecarStage2();
    }

    public sealed class Stage2SliceResult
    {
        public bool success;
        public string error;
        public string fileId;
        public int sliceSeq;
        public readonly List<Dictionary<string, object>> poles = new List<Dictionary<string, object>>();
        public readonly List<Dictionary<string, object>> lines = new List<Dictionary<string, object>>();
        public readonly List<Dictionary<string, object>> vertices = new List<Dictionary<string, object>>();
        public readonly List<Dictionary<string, object>> components = new List<Dictionary<string, object>>();
        public readonly List<Dictionary<string, object>> tracks = new List<Dictionary<string, object>>();
        public readonly List<Dictionary<string, object>> attachments = new List<Dictionary<string, object>>();
        public readonly List<int> stage1LineRows = new List<int>();
        public readonly List<int> acceptedLineRows = new List<int>();
        public Stage2ElectricalAudit audit = new Stage2ElectricalAudit();
        public Stage2Timing timing = new Stage2Timing();
    }

    [Serializable]
    public sealed class Stage2ElectricalAudit
    {
        public string runtime_version = "stage1-electrical-tracks-v10-voxel-supported-opt1-fix2-unity-native-1";
        public string stage1_label_source = "unity_sentis.deployedLabel";
        public string line_geometry_source = "stage1_class2_26_neighbor_graph_supported_paths";
        public int stage1_inferred_line_voxels;
        public int accepted_stage1_line_voxels;
        public double stage1_to_stage2_voxel_preservation;
        public int raw_stage1_line_components;
        public int joined_stage2_tracks;
        public int line_path_fragments;
        public int point_line_fragments;
        public int pole_attachments;
        public int geometry_support_samples;
        public int geometry_supported_samples;
        public double geometry_stage1_voxel_support_fraction;
        public int geometry_outside_stage1_voxel_samples;
        public double max_output_turn_deg;
        public double max_track_radius_p95_ft;
        public bool production_poles_preserved = true;
        public bool production_line_outputs_replaced = true;
        public bool disconnected_fragment_bridges_allowed = false;
        public int selected_fragment_bridges;
        public int synthetic_line_voxels;
        public bool runtime_gt_usage;
        public bool pole_pair_inference;
        public bool line_refiner_used;
        public bool line_hysteresis_used;
        public bool bridge_requires_stage1_class2_both_sides = true;
        public bool line_to_line_bridge_near_pole_allowed;
        public bool parallel_lane_merge_allowed;
        public bool open_line_endpoints_preserved = true;
        public bool pole_attachment_requires_stage1_voxel_contact = true;
        public int attachment_geometry_vertices_added;
    }

    [Serializable]
    public sealed class Stage2Timing
    {
        public double production_component_ms;
        public double production_refiner_parametric_ms;
        public double line_connected_components_ms;
        public double line_trace_ms;
        public double line_assignment_audit_ms;
        public double line_geometry_ms;
        public double disconnected_bridge_pair_enumeration_ms;
        public double stage2_total_ms;
    }
}

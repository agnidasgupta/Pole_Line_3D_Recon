#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;
using Unity.InferenceEngine;

/// <summary>
/// Optional Sentis-native weight quantization. This is separate from TensorRT
/// and must pass the same real-slice quality gate before deployment.
/// </summary>
public static class V10SentisFp16Quantizer
{
    [MenuItem("Tools/V10/Quantize Selected Model To FP16 Sentis")]
    private static void QuantizeSelected()
    {
        var asset = Selection.activeObject as ModelAsset;
        if (asset == null)
        {
            Debug.LogError("Select the imported V10 FP32 ModelAsset first.");
            return;
        }
        string source = AssetDatabase.GetAssetPath(asset);
        string directory = Path.GetDirectoryName(source) ?? "Assets";
        string target = Path.Combine(directory, Path.GetFileNameWithoutExtension(source) + "_weights_fp16.sentis");
        Model model = ModelLoader.Load(asset);
        ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
        ModelWriter.Save(target, model);
        AssetDatabase.Refresh();
        Debug.Log("Wrote " + target + ". Validate decision parity before selecting it at runtime.");
    }

    [MenuItem("Tools/V10/Quantize Selected Model To FP16 Sentis", true)]
    private static bool ValidateQuantizeSelected() => Selection.activeObject is ModelAsset;
}
#endif

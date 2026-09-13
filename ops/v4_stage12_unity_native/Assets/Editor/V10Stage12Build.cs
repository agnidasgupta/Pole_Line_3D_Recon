using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.InferenceEngine;
using VegetationAssurance.V10;

public static class V10Stage12Build
{
    private const string Fp32Path = "Assets/V10Stage12/Models/v10_stage1_voxelnet3d_fp32.onnx";
    private const string Fp16Path = "Assets/V10Stage12/Models/v10_stage1_voxelnet3d_fp16.onnx";
    private const string SidecarPath = "Assets/V10Stage12/Models/v10_stage12_sidecar.json";
    private const string TreesPath = "Assets/V10Stage12/Models/v10_stage2_refiner_trees.bytes";
    private const string ScenePath = "Assets/V10Stage12/Scenes/V10Stage12Server.unity";

    public static void BuildLinuxServer()
    {
        ModelAsset fp32 = Require<ModelAsset>(Fp32Path);
        TextAsset sidecar = Require<TextAsset>(SidecarPath);
        TextAsset trees = Require<TextAsset>(TreesPath);
        ModelAsset fp16 = AssetDatabase.LoadAssetAtPath<ModelAsset>(Fp16Path);

        Directory.CreateDirectory(Path.GetDirectoryName(ScenePath) ?? "Assets/V10Stage12/Scenes");
        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var root = new GameObject("V10Stage12NativeServer");
        V10Stage12SentisInferenceManager manager = root.AddComponent<V10Stage12SentisInferenceManager>();
        V10NativeSessionRunner runner = root.AddComponent<V10NativeSessionRunner>();

        var managerObject = new SerializedObject(manager);
        managerObject.FindProperty("fp32ModelAsset").objectReferenceValue = fp32;
        managerObject.FindProperty("fp16ModelAsset").objectReferenceValue = fp16;
        managerObject.FindProperty("sidecarJson").objectReferenceValue = sidecar;
        managerObject.FindProperty("requestFp16").boolValue = false;
        managerObject.FindProperty("unityFp16ParityApproved").boolValue = false;
        managerObject.FindProperty("warmupOnStart").boolValue = true;
        string buildKind = (Environment.GetEnvironmentVariable("V10_UNITY_BUILD_KIND") ?? "server").ToLowerInvariant();
        bool dedicatedServer = buildKind != "player";
        managerObject.FindProperty("backend").intValue = dedicatedServer
            ? (int)BackendType.CPU : (int)BackendType.GPUCompute;
        managerObject.ApplyModifiedPropertiesWithoutUndo();

        var runnerObject = new SerializedObject(runner);
        runnerObject.FindProperty("inferenceManager").objectReferenceValue = manager;
        runnerObject.FindProperty("stage2TreeBinary").objectReferenceValue = trees;
        runnerObject.FindProperty("runWhenCommandLinePresent").boolValue = true;
        runnerObject.ApplyModifiedPropertiesWithoutUndo();
        EditorSceneManager.SaveScene(scene, ScenePath);

        string output = Environment.GetEnvironmentVariable("V10_UNITY_BUILD_OUTPUT");
        if (string.IsNullOrWhiteSpace(output)) output = dedicatedServer
            ? "Builds/Linux/V10Stage12Server" : "Builds/Linux/V10Stage12GpuPlayer";
        Directory.CreateDirectory(Path.GetDirectoryName(output) ?? "Builds/Linux");
        var options = new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = output,
            target = BuildTarget.StandaloneLinux64,
            subtarget = (int)(dedicatedServer ? StandaloneBuildSubtarget.Server : StandaloneBuildSubtarget.Player),
            options = BuildOptions.CleanBuildCache
        };
        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != BuildResult.Succeeded)
            throw new BuildFailedException("V10 Stage12 server build failed: " + report.summary.result);
        Debug.Log("V10_STAGE12_NATIVE_BUILD_OK output=" + Path.GetFullPath(output));
    }

    private static T Require<T>(string path) where T : UnityEngine.Object
    {
        T asset = AssetDatabase.LoadAssetAtPath<T>(path);
        if (asset == null) throw new FileNotFoundException("Required Unity asset missing", path);
        return asset;
    }
}

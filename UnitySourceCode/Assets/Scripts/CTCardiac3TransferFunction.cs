using UnityEngine;
using UnityVolumeRendering;

using TransferFunction = UnityVolumeRendering.TransferFunction;

/// <summary>
/// 3D Slicer CT-Cardiac3 transfer fonksiyonunu UnityVolumeRendering'e uygular.
/// Kaynak: github.com/Slicer/Slicer - presets.xml (CT-Cardiac3)
/// </summary>
public class CTCardiac3TransferFunction : MonoBehaviour
{
    [Header("Runtime")]
    public bool applyOnStart = false;

    [Tooltip("Debug loglarini yazdir")]
    public bool verbose = true;

    private static readonly float[][] opacityPoints = new float[][]
    {
        new float[] { -3024f,    0.0f      },
        new float[] { -86.9767f, 0.0f      },
        new float[] {  45.3791f, 0.169643f },
        new float[] { 139.919f,  0.589286f },
        new float[] { 347.907f,  0.607143f },
        new float[] { 1224.16f,  0.607143f },
        new float[] { 3071f,     0.616071f },
    };

    private static readonly float[][] colorPoints = new float[][]
    {
        new float[] { -3024f,    0.000000f, 0.000000f, 0.000000f },
        new float[] { -86.9767f, 0.000000f, 0.250980f, 1.000000f },
        new float[] {  45.3791f, 1.000000f, 0.000000f, 0.000000f },
        new float[] { 139.919f,  1.000000f, 0.894893f, 0.894893f },
        new float[] { 347.907f,  1.000000f, 1.000000f, 0.250980f },
        new float[] { 1224.16f,  1.000000f, 1.000000f, 1.000000f },
        new float[] { 3071f,     0.827451f, 0.658824f, 1.000000f },
    };

    private float NormalizeHU(float hu, float dataMin, float dataMax)
    {
        if (Mathf.Approximately(dataMax, dataMin))
            return 0f;
        return Mathf.Clamp01((hu - dataMin) / (dataMax - dataMin));
    }

    [ContextMenu("Apply CT-Cardiac3")]
    public void ApplyCTCardiac3()
    {
        ApplyCTCardiac3(verbose);
    }

    public void ApplyCTCardiac3(bool log)
    {
        TryApplyCTCardiac3(log);
    }

    public bool TryApplyCTCardiac3(bool log)
    {
        VolumeRenderedObject volObj = GetComponent<VolumeRenderedObject>();
        if (volObj == null)
        {
            if (log) Debug.LogError("[CT-Cardiac3] VolumeRenderedObject bulunamadi!");
            return false;
        }

        VolumeDataset dataset = volObj.dataset;
        if (dataset == null)
        {
            if (log) Debug.LogWarning("[CT-Cardiac3] Dataset henuz hazir degil (volObj.dataset == null).");
            return false;
        }

        float dataMin = dataset.GetMinDataValue();
        float dataMax = dataset.GetMaxDataValue();
        if (log) Debug.Log($"[CT-Cardiac3] Dataset araligi: [{dataMin}, {dataMax}]");
        if (dataMax < dataMin)
        {
            if (log) Debug.LogError("[CT-Cardiac3] Dataset min/max gecersiz (max < min).");
            return false;
        }

        TransferFunction tf = volObj.transferFunction;
        if (tf == null)
        {
            if (log) Debug.LogError("[CT-Cardiac3] Transfer fonksiyonu bulunamadi!");
            return false;
        }

        tf.colourControlPoints.Clear();
        tf.alphaControlPoints.Clear();

        for (int i = 0; i < colorPoints.Length; i++)
        {
            float hu = colorPoints[i][0];
            if (hu < dataMin && i < colorPoints.Length - 1 && colorPoints[i + 1][0] < dataMin)
                continue;
            if (hu > dataMax && i > 0 && colorPoints[i - 1][0] > dataMax)
                continue;

            TFColourControlPoint cp = new TFColourControlPoint();
            cp.dataValue = NormalizeHU(hu, dataMin, dataMax);
            cp.colourValue = new Color(colorPoints[i][1], colorPoints[i][2], colorPoints[i][3]);
            tf.AddControlPoint(cp);
        }

        for (int i = 0; i < opacityPoints.Length; i++)
        {
            float hu = opacityPoints[i][0];
            if (hu < dataMin && i < opacityPoints.Length - 1 && opacityPoints[i + 1][0] < dataMin)
                continue;
            if (hu > dataMax && i > 0 && opacityPoints[i - 1][0] > dataMax)
                continue;

            TFAlphaControlPoint ap = new TFAlphaControlPoint();
            ap.dataValue = NormalizeHU(hu, dataMin, dataMax);
            ap.alphaValue = opacityPoints[i][1];
            tf.AddControlPoint(ap);
        }

        tf.GenerateTexture();

        if (log)
            Debug.Log("[CT-Cardiac3] Basariyla uygulandi!");

        return true;
    }

    private void Start()
    {
        if (applyOnStart)
            ApplyCTCardiac3();
    }
}

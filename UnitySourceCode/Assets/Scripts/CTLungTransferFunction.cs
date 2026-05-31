using UnityEngine;
using UnityVolumeRendering;

using TransferFunction = UnityVolumeRendering.TransferFunction;

/// <summary>
/// 3D Slicer CT-Lung transfer fonksiyonunu UnityVolumeRendering'e uygular.
/// Kaynak: github.com/Slicer/Slicer - presets.xml (CT-Lung)
/// effectiveRange: -600 ~ -399
/// Renk paleti: Mavi - Yesil - Sari - Kirmizi (akciger yogunluguna gore)
/// </summary>
public class CTLungTransferFunction : MonoBehaviour
{
    [Header("Runtime")]
    public bool applyOnStart = false;

    [Tooltip("Debug loglarini yazdir")]
    public bool verbose = true;

    private static readonly float[][] opacityPoints = new float[][]
    {
        new float[] { -1000f, 0.0f  },
        new float[] { -600f,  0.0f  },
        new float[] { -599f,  0.15f },
        new float[] { -400f,  0.15f },
        new float[] { -399f,  0.0f  },
        new float[] {  2952f, 0.0f  },
    };

    private static readonly float[][] colorPoints = new float[][]
    {
        new float[] { -1000f, 0.300000f, 0.300000f, 1.000000f },
        new float[] { -600f,  0.000000f, 0.000000f, 1.000000f },
        new float[] { -530f,  0.134704f, 0.781726f, 0.072456f },
        new float[] { -460f,  0.929244f, 1.000000f, 0.109473f },
        new float[] { -400f,  0.888889f, 0.254949f, 0.024026f },
        new float[] {  2952f, 1.000000f, 0.300000f, 0.300000f },
    };

    private float NormalizeHU(float hu, float dataMin, float dataMax)
    {
        if (Mathf.Approximately(dataMax, dataMin))
            return 0f;
        return Mathf.Clamp01((hu - dataMin) / (dataMax - dataMin));
    }

    [ContextMenu("Apply CT-Lung")]
    public void ApplyCTLung()
    {
        ApplyCTLung(verbose);
    }

    public void ApplyCTLung(bool log)
    {
        TryApplyCTLung(log);
    }

    public bool TryApplyCTLung(bool log)
    {
        VolumeRenderedObject volObj = GetComponent<VolumeRenderedObject>();
        if (volObj == null)
        {
            if (log) Debug.LogError("[CT-Lung] VolumeRenderedObject bulunamadi!");
            return false;
        }

        VolumeDataset dataset = volObj.dataset;
        if (dataset == null)
        {
            if (log) Debug.LogWarning("[CT-Lung] Dataset henuz hazir degil (volObj.dataset == null).");
            return false;
        }
        float dataMin = dataset.GetMinDataValue();
        float dataMax = dataset.GetMaxDataValue();
        if (log) Debug.Log($"[CT-Lung] Dataset araligi: [{dataMin}, {dataMax}]");
        if (dataMax < dataMin)
        {
            if (log) Debug.LogError("[CT-Lung] Dataset min/max gecersiz (max < min).");
            return false;
        }

        TransferFunction tf = volObj.transferFunction;
        if (tf == null)
        {
            if (log) Debug.LogError("[CT-Lung] Transfer fonksiyonu bulunamadi!");
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
        {
            Debug.Log("[CT-Lung] Basariyla uygulandi!");
        }

        return true;
    }

    void Start()
    {
        if (applyOnStart)
            ApplyCTLung();
    }
}

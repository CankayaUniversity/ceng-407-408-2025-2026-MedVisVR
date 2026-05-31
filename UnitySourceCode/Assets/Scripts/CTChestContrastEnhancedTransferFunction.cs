using UnityEngine;
using UnityVolumeRendering;

using TransferFunction = UnityVolumeRendering.TransferFunction;

/// <summary>
/// 3D Slicer CT-Chest-Contrast-Enhanced transfer fonksiyonunu uygular.
/// Desktop'taki yedek projedeki preset scriptinden runtime toggle uyumlu hale getirildi.
/// </summary>
public class CTChestContrastEnhancedTransferFunction : MonoBehaviour
{
    [Header("Runtime")]
    public bool applyOnStart = false;

    [Tooltip("Debug loglarini yazdir")]
    public bool verbose = true;

    private static readonly float[][] opacityPoints = new float[][]
    {
        new float[] { -3024f,    0.0f      },
        new float[] {  67.0106f, 0.0f      },
        new float[] { 251.105f,  0.446429f },
        new float[] { 439.291f,  0.625f    },
        new float[] { 3071f,     0.616071f },
    };

    private static readonly float[][] colorPoints = new float[][]
    {
        new float[] { -3024f,    0.000000f, 0.000000f, 0.000000f },
        new float[] {  67.0106f, 0.549020f, 0.250980f, 0.149020f },
        new float[] { 251.105f,  0.882353f, 0.603922f, 0.290196f },
        new float[] { 439.291f,  1.000000f, 0.937033f, 0.954531f },
        new float[] { 3071f,     0.827451f, 0.658824f, 1.000000f },
    };

    private float NormalizeHU(float hu, float dataMin, float dataMax)
    {
        if (Mathf.Approximately(dataMax, dataMin))
            return 0f;
        return Mathf.Clamp01((hu - dataMin) / (dataMax - dataMin));
    }

    [ContextMenu("Apply CT-Chest-Contrast-Enhanced")]
    public void ApplyCTChestContrastEnhanced()
    {
        ApplyCTChestContrastEnhanced(verbose);
    }

    public void ApplyCTChestContrastEnhanced(bool log)
    {
        TryApplyCTChestContrastEnhanced(log);
    }

    public bool TryApplyCTChestContrastEnhanced(bool log)
    {
        VolumeRenderedObject volObj = GetComponent<VolumeRenderedObject>();
        if (volObj == null)
        {
            if (log) Debug.LogError("[CT-Chest-CE] VolumeRenderedObject bulunamadi!");
            return false;
        }

        VolumeDataset dataset = volObj.dataset;
        if (dataset == null)
        {
            if (log) Debug.LogWarning("[CT-Chest-CE] Dataset henuz hazir degil.");
            return false;
        }

        float dataMin = dataset.GetMinDataValue();
        float dataMax = dataset.GetMaxDataValue();
        if (log) Debug.Log($"[CT-Chest-CE] Dataset araligi: [{dataMin}, {dataMax}]");

        TransferFunction tf = volObj.transferFunction;
        if (tf == null)
        {
            if (log) Debug.LogError("[CT-Chest-CE] Transfer fonksiyonu bulunamadi!");
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
            Debug.Log("[CT-Chest-CE] Basariyla uygulandi!");

        return true;
    }

    private void Start()
    {
        if (applyOnStart)
            ApplyCTChestContrastEnhanced();
    }
}

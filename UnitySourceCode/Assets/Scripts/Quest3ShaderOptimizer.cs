using UnityEngine;
using UnityVolumeRendering;

/// <summary>
/// Quest 3 (Adreno GPU) icin volume rendering shader optimizasyonlari.
/// Android build'lerde otomatik olarak:
///   - Sampling rate'i dusurur (512 -> ~333 adim, opacity correction ile kalite korunur)
///   - Tricubic interpolation'i kapatir (8x texture okuma tasarrufu)
/// </summary>
public class Quest3ShaderOptimizer : MonoBehaviour
{
    [Header("Android/Quest 3 Ayarlari")]
    [Tooltip("Android'de kullanilacak sampling rate carpani (0.2-2.0). Dusuk = daha hizli, yuksek = daha net.")]
    [SerializeField, Range(0.2f, 2.0f)]
    private float mobileSamplingRate = 0.65f;

    [Tooltip("Android'de tricubic interpolation'i kapat (onemli performans kazanimi, gorsel kayip ihmal edilebilir)")]
    [SerializeField]
    private bool disableTricubicOnMobile = true;

    [Header("PC/Editor Ayarlari")]
    [SerializeField, Range(0.2f, 2.0f)]
    private float desktopSamplingRate = 1.0f;

    private VolumeRenderedObject volumeObject;
    private bool applied;

    private void Start()
    {
        TryApply();
    }

    private void Update()
    {
        if (!applied)
        {
            TryApply();
        }
    }

    private void TryApply()
    {
        if (volumeObject == null)
        {
            volumeObject = FindAnyObjectByType<VolumeRenderedObject>();
        }

        if (volumeObject == null || volumeObject.dataset == null)
            return;

#if UNITY_ANDROID && !UNITY_EDITOR
        volumeObject.SetSamplingRateMultiplier(mobileSamplingRate);
        if (disableTricubicOnMobile)
        {
            volumeObject.SetCubicInterpolationEnabled(false);
        }
        Debug.Log($"[Quest3ShaderOptimizer] Mobil optimizasyon uygulandi: samplingRate={mobileSamplingRate}, tricubic={!disableTricubicOnMobile}");
#else
        volumeObject.SetSamplingRateMultiplier(desktopSamplingRate);
        Debug.Log($"[Quest3ShaderOptimizer] Desktop ayarlari uygulandi: samplingRate={desktopSamplingRate}");
#endif

        applied = true;
        enabled = false;
    }
}

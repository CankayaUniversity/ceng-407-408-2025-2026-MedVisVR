using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using UnityVolumeRendering;

using TransferFunction = UnityVolumeRendering.TransferFunction;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Meta Quest controller B tusuna basarak runtime'da TF preset'lerini gecirir.
/// Baslangicta volume DEFAULT TF (volume olusturuldugunda paketin verdigi) ile baslar.
/// Cycle: Default -> CT-Lung -> CT-Cardiac3 -> CT-Chest-CE -> Default ...
///
/// New Input System kullanir (activeInputHandler: 1)
/// B tusu = Sag controller secondaryButton
/// </summary>
public class TransferFunctionToggle : MonoBehaviour
{
    [Header("Ayarlar")]
    [Tooltip("True ise CT-Lung ile baslar, false ise default ile baslar")]
    public bool startWithCTLung = false;

    [Tooltip("Input debug loglari (Quest'te logcat ile gorunur).")]
    public bool verboseInput = false;

    private enum PresetKind
    {
        Default = 0,
        CTLung = 1,
        CTCardiac3 = 2,
        CTChestContrastEnhanced = 3
    }

    // Volume olusunca first B basisi CT-Lung'a gecirir.
    private PresetKind nextPreset = PresetKind.CTLung;
    private PresetKind currentPreset = PresetKind.Default;

    private List<TFColourControlPoint> defaultColorPoints = new List<TFColourControlPoint>();
    private List<TFAlphaControlPoint> defaultAlphaPoints = new List<TFAlphaControlPoint>();
    private bool defaultSaved = false;

#if ENABLE_INPUT_SYSTEM
    // New Input System action
    private InputAction bButtonAction;
#endif

    // XR poll fallback (CommonUsages.secondaryButton)
    private bool lastSecondaryPressed = false;
    private int lastToggleFrame = -1;
    private float lastToggleTime = -999f;
    private float lastInputStatusLogTime = -999f;
    private readonly List<UnityEngine.XR.InputDevice> rightHandDevices = new List<UnityEngine.XR.InputDevice>();

    // Referanslar
    private VolumeRenderedObject volObj;
    private CTLungTransferFunction ctLungScript;
    private CTCardiac3TransferFunction ctCardiac3Script;
    private CTChestContrastEnhancedTransferFunction ctChestContrastEnhancedScript;

    private bool pendingApply;

    void OnEnable()
    {
#if ENABLE_INPUT_SYSTEM
        // B butonu icin InputAction olustur
        // Quest sag controller secondaryButton = B tusu
        // Klavye B tusu da eklendi (editor test)
        bButtonAction = new InputAction("BButton", InputActionType.Button);
        bButtonAction.AddBinding("<XRController>{RightHand}/secondaryButton");
        bButtonAction.AddBinding("<OculusTouchController>{RightHand}/secondaryButton");
        // Bazi cihaz/layout'larda handedness usage set olmayabiliyor.
        bButtonAction.AddBinding("<XRController>/secondaryButton");
        // Bazi layout'lar buttonEast (B) olarak raporlayabiliyor.
        bButtonAction.AddBinding("<XRController>{RightHand}/buttonEast");
        bButtonAction.AddBinding("<OculusTouchController>{RightHand}/buttonEast");
        bButtonAction.AddBinding("<XRController>/buttonEast");
        bButtonAction.AddBinding("<Keyboard>/b");

        bButtonAction.performed += OnBButtonPressed;
        bButtonAction.Enable();

        Debug.Log("[TF-Toggle] InputAction olusturuldu ve etkinlestirildi (New Input System)");
#else
        Debug.Log("[TF-Toggle] New Input System devre disi (ENABLE_INPUT_SYSTEM yok). XR poll fallback kullanilacak.");
#endif

        if (verboseInput)
            Debug.Log("[TF-Toggle] verboseInput=ON");
    }

    void OnDisable()
    {
#if ENABLE_INPUT_SYSTEM
        if (bButtonAction != null)
        {
            bButtonAction.performed -= OnBButtonPressed;
            bButtonAction.Disable();
            bButtonAction.Dispose();
            bButtonAction = null;
        }
#endif
    }

    void Start()
    {
        Debug.Log("[TF-Toggle] Script baslatildi (New Input System modu)");

        FindVolumeObject();

        if (startWithCTLung && volObj != null)
        {
            SaveDefaultTF();
            ApplyCTLung();
        }
    }

    void Update()
    {
        // VolumeRenderedObject henuz yoksa ara
        if (volObj == null)
        {
            FindVolumeObject();
            if (volObj == null) return;
        }

        if (!defaultSaved)
            SaveDefaultTF();

        PollRightHandSecondaryButton();

        if (pendingApply)
        {
            ApplyPendingPreset();
        }
    }

    /// <summary>
    /// B tusuna basildiginda cagirilir (New Input System callback)
    /// </summary>
#if ENABLE_INPUT_SYSTEM
    private void OnBButtonPressed(InputAction.CallbackContext context)
    {
        RequestToggle("InputSystem");
    }
#endif

    /// <summary>
    /// Default -> CT-Lung -> CT-Cardiac3 -> Default cycle.
    /// </summary>
    public void Toggle()
    {
        if (volObj == null)
        {
            Debug.LogWarning("[TF-Toggle] VolumeRenderedObject bulunamadi!");
            return;
        }

        if (ctLungScript == null || ctCardiac3Script == null || ctChestContrastEnhancedScript == null)
            FindVolumeObject();

        if (!defaultSaved)
            SaveDefaultTF();

        if (nextPreset == PresetKind.CTLung)
        {
            bool applied = ApplyCTLung();
            if (applied)
                Debug.Log("[TF-Toggle] >> CT-Lung TF aktif");
            else
                Debug.LogWarning("[TF-Toggle] >> CT-Lung beklemede (dataset/TF henuz hazir degil)");
        }
        else if (nextPreset == PresetKind.CTCardiac3)
        {
            bool applied = ApplyCTCardiac3();
            if (applied)
                Debug.Log("[TF-Toggle] >> CT-Cardiac3 TF aktif");
            else
                Debug.LogWarning("[TF-Toggle] >> CT-Cardiac3 beklemede (dataset/TF henuz hazir degil)");
        }
        else if (nextPreset == PresetKind.CTChestContrastEnhanced)
        {
            bool applied = ApplyCTChestContrastEnhanced();
            if (applied)
                Debug.Log("[TF-Toggle] >> CT-Chest-CE TF aktif");
            else
                Debug.LogWarning("[TF-Toggle] >> CT-Chest-CE beklemede (dataset/TF henuz hazir degil)");
        }
        else if (nextPreset == PresetKind.Default)
        {
            bool applied = ApplyDefault();
            if (applied)
                Debug.Log("[TF-Toggle] >> Default TF aktif");
            else
                Debug.LogWarning("[TF-Toggle] >> Default beklemede (dataset/TF henuz hazir degil)");
        }
    }

    // =========================================================================
    // PRIVATE METHODS
    // =========================================================================

    private void RequestToggle(string source)
    {
        if (Time.frameCount == lastToggleFrame)
            return;

        if (Time.unscaledTime - lastToggleTime < 0.2f)
            return;

        lastToggleFrame = Time.frameCount;
        lastToggleTime = Time.unscaledTime;
        Debug.Log($"[TF-Toggle] B tusu algilandi! (source={source})");
        Toggle();
    }

    private void PollRightHandSecondaryButton()
    {
        UnityEngine.XR.InputDevice dev = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand);
        if (!dev.isValid)
        {
            // XRNode bazen invalid donebiliyor; characteristics ile sag el controller ara.
            rightHandDevices.Clear();
            UnityEngine.XR.InputDevices.GetDevicesWithCharacteristics(
                InputDeviceCharacteristics.Controller | InputDeviceCharacteristics.Right,
                rightHandDevices
            );

            if (rightHandDevices.Count == 0)
            {
                if (verboseInput && Time.unscaledTime - lastInputStatusLogTime > 3f)
                {
                    lastInputStatusLogTime = Time.unscaledTime;
                    Debug.LogWarning("[TF-Toggle] RightHand XR device bulunamadi (XRNode invalid).");
                }
                return;
            }

            dev = rightHandDevices[0];
        }

        bool hasSecondary = dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.secondaryButton, out bool pressed);

        if (verboseInput && Time.unscaledTime - lastInputStatusLogTime > 3f)
        {
            lastInputStatusLogTime = Time.unscaledTime;
            Debug.Log($"[TF-Toggle] XR device='{dev.name}' isValid={dev.isValid} pressed={pressed} hasSecondary={hasSecondary}");
        }

        if (hasSecondary && pressed && !lastSecondaryPressed)
        {
            RequestToggle("XR.CommonUsages.secondaryButton");
        }
        lastSecondaryPressed = pressed;
    }

    private void FindVolumeObject()
    {
        volObj = GetComponent<VolumeRenderedObject>();
        if (volObj == null)
            volObj = FindAnyObjectByType<VolumeRenderedObject>();
        if (volObj != null)
        {
            Debug.Log($"[TF-Toggle] VolumeRenderedObject bulundu: {volObj.gameObject.name}");

            ctLungScript = volObj.GetComponent<CTLungTransferFunction>();
            if (ctLungScript == null)
            {
                ctLungScript = volObj.gameObject.AddComponent<CTLungTransferFunction>();
                ctLungScript.applyOnStart = false;
                ctLungScript.verbose = true;
                Debug.Log("[TF-Toggle] CTLungTransferFunction otomatik eklendi.");
            }
            else
            {
                // Toggle mantigi icin Start'ta otomatik uygulamayi kapat (default TF'nin bozulmasini onler).
                if (ctLungScript.applyOnStart)
                    Debug.LogWarning("[TF-Toggle] CTLungTransferFunction.applyOnStart acikmis; Toggle kontrolu icin kapatiliyor.");
                ctLungScript.applyOnStart = false;
                ctLungScript.verbose = true;
            }

            ctCardiac3Script = volObj.GetComponent<CTCardiac3TransferFunction>();
            if (ctCardiac3Script == null)
            {
                ctCardiac3Script = volObj.gameObject.AddComponent<CTCardiac3TransferFunction>();
                ctCardiac3Script.applyOnStart = false;
                ctCardiac3Script.verbose = true;
                Debug.Log("[TF-Toggle] CTCardiac3TransferFunction otomatik eklendi.");
            }
            else
            {
                ctCardiac3Script.applyOnStart = false;
                ctCardiac3Script.verbose = true;
            }

            ctChestContrastEnhancedScript = volObj.GetComponent<CTChestContrastEnhancedTransferFunction>();
            if (ctChestContrastEnhancedScript == null)
            {
                ctChestContrastEnhancedScript = volObj.gameObject.AddComponent<CTChestContrastEnhancedTransferFunction>();
                ctChestContrastEnhancedScript.applyOnStart = false;
                ctChestContrastEnhancedScript.verbose = true;
                Debug.Log("[TF-Toggle] CTChestContrastEnhancedTransferFunction otomatik eklendi.");
            }
            else
            {
                ctChestContrastEnhancedScript.applyOnStart = false;
                ctChestContrastEnhancedScript.verbose = true;
            }
        }
    }

    private bool ApplyCTLung()
    {
        if (ctLungScript == null)
        {
            Debug.LogError("[TF-Toggle] CTLungTransferFunction scripti yok!");
            return false;
        }

        bool applied = ctLungScript.TryApplyCTLung(true);
        if (!applied)
        {
            Debug.LogWarning("[TF-Toggle] CT-Lung uygulanamadi (dataset/TF henuz hazir olmayabilir).");
            nextPreset = PresetKind.CTLung;
            pendingApply = true;
            return false;
        }

        nextPreset = PresetKind.CTCardiac3;
        currentPreset = PresetKind.CTLung;
        pendingApply = false;
        return true;
    }

    private bool ApplyCTCardiac3()
    {
        if (ctCardiac3Script == null)
        {
            Debug.LogError("[TF-Toggle] CTCardiac3TransferFunction scripti yok!");
            return false;
        }

        bool applied = ctCardiac3Script.TryApplyCTCardiac3(true);
        if (!applied)
        {
            Debug.LogWarning("[TF-Toggle] CT-Cardiac3 uygulanamadi (dataset/TF henuz hazir olmayabilir).");
            nextPreset = PresetKind.CTCardiac3;
            pendingApply = true;
            return false;
        }

        nextPreset = PresetKind.CTChestContrastEnhanced;
        currentPreset = PresetKind.CTCardiac3;
        pendingApply = false;
        return true;
    }

    private bool ApplyCTChestContrastEnhanced()
    {
        if (ctChestContrastEnhancedScript == null)
        {
            Debug.LogError("[TF-Toggle] CTChestContrastEnhancedTransferFunction scripti yok!");
            return false;
        }

        bool applied = ctChestContrastEnhancedScript.TryApplyCTChestContrastEnhanced(true);
        if (!applied)
        {
            Debug.LogWarning("[TF-Toggle] CT-Chest-CE uygulanamadi (dataset/TF henuz hazir olmayabilir).");
            nextPreset = PresetKind.CTChestContrastEnhanced;
            pendingApply = true;
            return false;
        }

        nextPreset = PresetKind.Default;
        currentPreset = PresetKind.CTChestContrastEnhanced;
        pendingApply = false;
        return true;
    }

    private bool ApplyDefault()
    {
        if (volObj == null || !defaultSaved)
        {
            Debug.LogWarning("[TF-Toggle] Default uygulanamadi (volume/TF henuz hazir degil).");
            nextPreset = PresetKind.Default;
            pendingApply = true;
            return false;
        }

        TransferFunction tf = volObj.transferFunction;
        if (tf == null)
        {
            nextPreset = PresetKind.Default;
            pendingApply = true;
            return false;
        }

        tf.colourControlPoints.Clear();
        tf.alphaControlPoints.Clear();

        foreach (TFColourControlPoint cp in defaultColorPoints)
        {
            TFColourControlPoint copy = new TFColourControlPoint();
            copy.dataValue = cp.dataValue;
            copy.colourValue = cp.colourValue;
            tf.AddControlPoint(copy);
        }

        foreach (TFAlphaControlPoint ap in defaultAlphaPoints)
        {
            TFAlphaControlPoint copy = new TFAlphaControlPoint();
            copy.dataValue = ap.dataValue;
            copy.alphaValue = ap.alphaValue;
            tf.AddControlPoint(copy);
        }

        tf.GenerateTexture();
        currentPreset = PresetKind.Default;
        nextPreset = PresetKind.CTLung;
        pendingApply = false;
        return true;
    }

    private void SaveDefaultTF()
    {
        if (volObj == null)
            return;

        TransferFunction tf = volObj.transferFunction;
        if (tf == null)
            return;

        if (tf.colourControlPoints.Count == 0 && tf.alphaControlPoints.Count == 0)
            return;

        defaultColorPoints.Clear();
        defaultAlphaPoints.Clear();

        foreach (TFColourControlPoint cp in tf.colourControlPoints)
        {
            TFColourControlPoint copy = new TFColourControlPoint();
            copy.dataValue = cp.dataValue;
            copy.colourValue = cp.colourValue;
            defaultColorPoints.Add(copy);
        }

        foreach (TFAlphaControlPoint ap in tf.alphaControlPoints)
        {
            TFAlphaControlPoint copy = new TFAlphaControlPoint();
            copy.dataValue = ap.dataValue;
            copy.alphaValue = ap.alphaValue;
            defaultAlphaPoints.Add(copy);
        }

        defaultSaved = true;
    }

    private void ApplyPendingPreset()
    {
        if (nextPreset == PresetKind.CTLung)
            ApplyCTLung();
        else if (nextPreset == PresetKind.CTCardiac3)
            ApplyCTCardiac3();
        else if (nextPreset == PresetKind.CTChestContrastEnhanced)
            ApplyCTChestContrastEnhanced();
        else
            ApplyDefault();
    }
}

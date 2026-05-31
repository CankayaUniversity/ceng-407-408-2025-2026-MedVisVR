using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Runtime'da NIFTI yuklenirken kullaniciya durum gosteren basit world-space HUD.
/// Kameraya parent yapilir, HUD mesajlarini RuntimeVolumeBootstrap gonderir.
/// </summary>
public class RuntimeVolumeLoadingHUD : MonoBehaviour
{
    private const string HudRootName = "[HUD] Runtime Volume Loading";

    [Header("Placement")]
    [SerializeField] private Vector3 hudLocalPosition = new Vector3(0f, 0f, 1.2f);
    [SerializeField] private float hudWorldScale = 0.00075f;
    [SerializeField] private Vector2 hudSize = new Vector2(520f, 180f);
    [SerializeField] private bool invertFacing = true;

    [Header("Style")]
    [SerializeField] private Color backgroundColor = new Color(0.08f, 0.08f, 0.10f, 0.92f);
    [SerializeField] private Color textColor = Color.white;
    [SerializeField] private Color errorColor = new Color(1f, 0.45f, 0.45f, 1f);
    [SerializeField] private int fontSize = 22;
    [SerializeField] private float padding = 14f;

    private Canvas canvas;
    private Text statusText;
    private Image panelImage;
    private Camera hudCamera;

    private static RuntimeVolumeLoadingHUD _instance;
    public static RuntimeVolumeLoadingHUD Instance => _instance;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Bootstrap()
    {
        if (FindFirstObjectByType<RuntimeVolumeLoadingHUD>() != null)
            return;

        GameObject go = new GameObject(HudRootName);
        DontDestroyOnLoad(go);
        go.AddComponent<RuntimeVolumeLoadingHUD>();
    }

    private void Awake()
    {
        _instance = this;
        SetupHud();
        SetMessage("Baslatiliyor...");
    }

    private void Update()
    {
        EnsureCameraAttachment();
        UpdateRotation();
    }

    private void SetupHud()
    {
        GameObject canvasGo = new GameObject("Canvas");
        canvasGo.transform.SetParent(transform, false);

        canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.overrideSorting = true;
        canvas.sortingOrder = 5500;

        CanvasGroup canvasGroup = canvasGo.AddComponent<CanvasGroup>();
        canvasGroup.blocksRaycasts = false;
        canvasGroup.interactable = false;

        GraphicRaycaster raycaster = canvasGo.AddComponent<GraphicRaycaster>();
        raycaster.enabled = false;

        RectTransform canvasRt = canvas.GetComponent<RectTransform>();
        canvasRt.sizeDelta = hudSize;
        canvasRt.localScale = Vector3.one * hudWorldScale;

        GameObject panelGo = new GameObject("Panel");
        panelGo.transform.SetParent(canvasGo.transform, false);
        panelImage = panelGo.AddComponent<Image>();
        panelImage.color = backgroundColor;
        panelImage.raycastTarget = false;

        RectTransform panelRt = panelImage.GetComponent<RectTransform>();
        panelRt.anchorMin = Vector2.zero;
        panelRt.anchorMax = Vector2.one;
        panelRt.offsetMin = Vector2.zero;
        panelRt.offsetMax = Vector2.zero;

        GameObject textGo = new GameObject("Status");
        textGo.transform.SetParent(panelGo.transform, false);
        statusText = textGo.AddComponent<Text>();
        statusText.raycastTarget = false;
        statusText.color = textColor;
        statusText.fontSize = fontSize;
        statusText.fontStyle = FontStyle.Bold;
        statusText.alignment = TextAnchor.MiddleCenter;
        statusText.horizontalOverflow = HorizontalWrapMode.Wrap;
        statusText.verticalOverflow = VerticalWrapMode.Overflow;
        statusText.supportRichText = false;
        statusText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

        RectTransform textRt = statusText.GetComponent<RectTransform>();
        textRt.anchorMin = new Vector2(0f, 0f);
        textRt.anchorMax = new Vector2(1f, 1f);
        textRt.offsetMin = new Vector2(padding, padding);
        textRt.offsetMax = new Vector2(-padding, -padding);

        transform.localScale = Vector3.one;
    }

    private void EnsureCameraAttachment()
    {
        Camera cam = TryFindHudCamera();
        if (cam == null)
            return;

        hudCamera = cam;
        transform.SetParent(hudCamera.transform, false);
        transform.localPosition = hudLocalPosition;

        if (canvas != null)
            canvas.worldCamera = hudCamera;
    }

    private void UpdateRotation()
    {
        if (hudCamera == null)
            return;

        Vector3 toCam = hudCamera.transform.position - transform.position;
        if (toCam.sqrMagnitude > 0.0001f)
        {
            Vector3 lookDir = invertFacing ? -toCam : toCam;
            transform.rotation = Quaternion.LookRotation(lookDir, hudCamera.transform.up);
        }
    }

    private static Camera TryFindHudCamera()
    {
        Camera main = Camera.main;
        if (main != null && main.isActiveAndEnabled)
            return main;

        Camera[] cams = FindObjectsByType<Camera>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        if (cams == null || cams.Length == 0)
            return null;

        for (int i = 0; i < cams.Length; i++)
        {
            Camera c = cams[i];
            if (c != null && c.isActiveAndEnabled && (c.stereoEnabled || c.stereoTargetEye != StereoTargetEyeMask.None))
                return c;
        }

        for (int i = 0; i < cams.Length; i++)
        {
            Camera c = cams[i];
            if (c != null && c.isActiveAndEnabled)
                return c;
        }
        return cams[0];
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    public void SetMessage(string message)
    {
        if (statusText == null)
            return;
        statusText.color = textColor;
        statusText.text = message;
        if (panelImage != null)
            panelImage.gameObject.SetActive(true);
    }

    public void SetError(string message)
    {
        if (statusText == null)
            return;
        statusText.color = errorColor;
        statusText.text = message;
        if (panelImage != null)
            panelImage.gameObject.SetActive(true);
    }

    public void Hide()
    {
        if (canvas != null)
            canvas.gameObject.SetActive(false);
    }

    public void Show()
    {
        if (canvas != null)
            canvas.gameObject.SetActive(true);
    }
}

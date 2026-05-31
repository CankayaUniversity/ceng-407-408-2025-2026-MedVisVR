using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

public class SlicingPlaneAngleHUD : MonoBehaviour
{
    private const string HudRootName = "[HUD] SlicingPlane Angles";

    [Header("Debug")]
    [SerializeField] private bool verbose = false;
    [Tooltip("HUD layer'ini kameranin layer'i ile ayni yapar (genelde gerekmez).")]
    [SerializeField] private bool matchCameraLayer = true;

    [Header("Target")]
    [Tooltip("Birincil hedef: kullanicinin kavradigi Sphere. RuntimePlaneRig sphere.rotation'u " +
             "SlicingPlane'e aktariyor, bu yuzden sphere.up = slicingPlane.up. Sphere bulunmazsa " +
             "slicingPlane fallback olarak kullanilir.")]
    [SerializeField] private Transform sphere;

    [Tooltip("Sphere otomatik bulunurken aranacak GO ismi (RuntimeVolumeBootstrap 'Sphere_Runtime' uretir).")]
    [SerializeField] private string sphereSearchName = "Sphere_Runtime";

    [Tooltip("Sphere bulunmazsa fallback olarak kullanilan slicing plane referansi.")]
    [SerializeField] private Transform slicingPlane;

    [Tooltip("Slicing plane'i otomatik bulmak icin isim parcasi.")]
    [SerializeField] private string slicingPlaneNameContains = "SlicingPlane";

    [Header("Screen Anchor")]
    [Tooltip("HUD'u kameranin goruntusune gore sabitler.")]
    [SerializeField] private bool anchorToViewport = true;

    [Tooltip("Viewport koordinati (0..1). Ornek: (0.9, 0.1) sag-alt. (0.5, 0.5) merkez.")]
    [SerializeField] private Vector2 viewportAnchor = new Vector2(0.58f, 0.27f);

    [Tooltip("Kameradan uzaklik (metre).")]
    [SerializeField] private float viewportDistance = 1.0f;

    [Header("HUD Placement (camera local)")]
    [SerializeField] private Vector3 hudLocalPosition = new Vector3(0.12f, -0.16f, 0.60f);
    [SerializeField] private Vector3 hudLocalEuler = Vector3.zero;
    [SerializeField] private bool faceCamera = true;
    [Tooltip("Bazi XR kurulumlarinda UI'nin on/arka yuzu ters olabiliyor. Yazi ters gorunuyorsa acik birakin.")]
    [SerializeField] private bool invertFacing = true;
    [SerializeField] private float hudWorldScale = 0.00085f;
    [SerializeField] private Vector2 hudSize = new Vector2(280f, 92f);

    [Header("HUD Style")]
    [SerializeField] private Color backgroundColor = new Color(0.25f, 0.25f, 0.25f, 0.85f);
    [SerializeField] private Color textColor = Color.white;
    [SerializeField] private int fontSize = 24;
    [SerializeField] private float iconSize = 42f;
    [SerializeField] private float padding = 10f;

    [Header("Update")]
    [SerializeField] private float refreshRateHz = 15f;

    private Text angleText;
    private float nextRefreshTime;
    private string lastRendered;
    private Canvas canvas;
    private Camera hudCamera;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Bootstrap()
    {
        if (FindFirstObjectByType<SlicingPlaneAngleHUD>() != null)
            return;

        GameObject go = new GameObject(HudRootName);
        DontDestroyOnLoad(go);
        go.AddComponent<SlicingPlaneAngleHUD>();
    }

    private void Awake()
    {
        SetupHud();
        if (verbose) Debug.Log("[SlicingPlaneAngleHUD] HUD olusturuldu.");
    }

    private void OnEnable()
    {
        nextRefreshTime = 0f;
        lastRendered = null;
    }

    private void Update()
    {
        EnsureCameraAttachment();
        UpdateAnchorPosition();

        if (Time.unscaledTime < nextRefreshTime)
            return;

        nextRefreshTime = Time.unscaledTime + (refreshRateHz <= 0f ? 0.1f : (1f / refreshRateHz));

        Transform freshSphere = TryFindSphere();
        if (freshSphere != null)
        {
            if (verbose && sphere != freshSphere)
                Debug.Log($"[SlicingPlaneAngleHUD] Sphere -> '{freshSphere.name}' (rot={freshSphere.rotation.eulerAngles})");
            sphere = freshSphere;
        }
        else
        {
            sphere = null;
        }

        if (sphere == null)
            slicingPlane = TryFindSlicingPlane();

        if (angleText == null)
            return;

        Transform source = sphere != null ? sphere : slicingPlane;
        if (source == null)
        {
            SetTextIfChanged("Sphere: (bulunamadi)");
            return;
        }

        Vector3 normal = source.up;
        float xzAngle = Vector3.Angle(normal, Vector3.up);
        if (xzAngle > 90f) xzAngle = 180f - xzAngle;

        const string deg = "°";
        string s = "XZ acisi: " + xzAngle.ToString("0.0", CultureInfo.InvariantCulture) + deg;
        SetTextIfChanged(s);
    }

    private void UpdateAnchorPosition()
    {
        if (!anchorToViewport || hudCamera == null)
            return;

        float d = Mathf.Max(0.2f, viewportDistance);
        Vector3 worldPoint = hudCamera.ViewportToWorldPoint(new Vector3(viewportAnchor.x, viewportAnchor.y, d));
        Vector3 localPoint = hudCamera.transform.InverseTransformPoint(worldPoint);
        transform.localPosition = localPoint;
        UpdateRotation();
    }

    private void UpdateRotation()
    {
        if (hudCamera == null)
            return;

        if (faceCamera)
        {
            Vector3 toCam = hudCamera.transform.position - transform.position;
            if (toCam.sqrMagnitude > 0.0001f)
            {
                Vector3 lookDir = invertFacing ? -toCam : toCam;
                transform.rotation = Quaternion.LookRotation(lookDir, hudCamera.transform.up);
            }
        }
        else
        {
            transform.localRotation = Quaternion.Euler(hudLocalEuler);
        }
    }

    private void SetupHud()
    {
        GameObject canvasGo = new GameObject("Canvas");
        canvasGo.transform.SetParent(transform, false);

        canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.overrideSorting = true;
        canvas.sortingOrder = 5000;

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
        Image panel = panelGo.AddComponent<Image>();
        panel.color = backgroundColor;
        panel.raycastTarget = false;

        RectTransform panelRt = panel.GetComponent<RectTransform>();
        panelRt.anchorMin = Vector2.zero;
        panelRt.anchorMax = Vector2.one;
        panelRt.offsetMin = Vector2.zero;
        panelRt.offsetMax = Vector2.zero;

        GameObject iconGo = new GameObject("Icon");
        iconGo.transform.SetParent(panelGo.transform, false);
        Image icon = iconGo.AddComponent<Image>();
        icon.sprite = CreateSlicingPlaneIconSprite();
        icon.preserveAspect = true;
        icon.raycastTarget = false;
        icon.color = Color.white;

        RectTransform iconRt = icon.GetComponent<RectTransform>();
        iconRt.anchorMin = new Vector2(0f, 0.5f);
        iconRt.anchorMax = new Vector2(0f, 0.5f);
        iconRt.pivot = new Vector2(0f, 0.5f);
        iconRt.anchoredPosition = new Vector2(padding, 0f);
        iconRt.sizeDelta = new Vector2(iconSize, iconSize);

        GameObject textGo = new GameObject("Angles");
        textGo.transform.SetParent(panelGo.transform, false);
        angleText = textGo.AddComponent<Text>();
        angleText.raycastTarget = false;
        angleText.color = textColor;
        angleText.fontSize = fontSize;
        angleText.fontStyle = FontStyle.Bold;
        angleText.alignment = TextAnchor.MiddleLeft;
        angleText.horizontalOverflow = HorizontalWrapMode.Overflow;
        angleText.verticalOverflow = VerticalWrapMode.Overflow;
        angleText.supportRichText = false;
        angleText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

        RectTransform textRt = angleText.GetComponent<RectTransform>();
        textRt.anchorMin = new Vector2(0f, 0f);
        textRt.anchorMax = new Vector2(1f, 1f);
        textRt.offsetMin = new Vector2(padding + iconSize + 10f, 0f);
        textRt.offsetMax = new Vector2(-padding, 0f);

        transform.localScale = Vector3.one;
    }

    private void EnsureCameraAttachment()
    {
        Camera cam = TryFindHudCamera();
        if (cam == null)
            return;

        bool cameraChanged = hudCamera != cam;

        hudCamera = cam;
        transform.SetParent(hudCamera.transform, false);
        transform.localPosition = hudLocalPosition;
        UpdateRotation();

        if (canvas != null)
            canvas.worldCamera = hudCamera;

        if (matchCameraLayer)
            SetLayerRecursively(gameObject, hudCamera.gameObject.layer);

        if (verbose && cameraChanged) Debug.Log($"[SlicingPlaneAngleHUD] Kamera baglandi: {hudCamera.name}");
    }

    private static void SetLayerRecursively(GameObject go, int layer)
    {
        if (go == null) return;
        go.layer = layer;
        Transform t = go.transform;
        for (int i = 0; i < t.childCount; i++)
            SetLayerRecursively(t.GetChild(i).gameObject, layer);
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

    private Transform TryFindSphere()
    {
        if (!string.IsNullOrWhiteSpace(sphereSearchName))
        {
            GameObject byName = GameObject.Find(sphereSearchName);
            if (byName != null)
                return byName.transform;
        }

        GameObject runtime = GameObject.Find("Sphere_Runtime");
        if (runtime != null)
            return runtime.transform;

        GameObject template = GameObject.Find("Sphere");
        if (template != null && template.activeInHierarchy)
            return template.transform;

        Transform[] transforms = FindObjectsByType<Transform>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        for (int i = 0; i < transforms.Length; i++)
        {
            Transform t = transforms[i];
            if (t == null || t.name == null) continue;
            if (t.name.IndexOf("sphere", System.StringComparison.OrdinalIgnoreCase) >= 0)
                return t;
        }

        return null;
    }

    private Transform TryFindSlicingPlane()
    {
        GameObject exact = GameObject.Find("SlicingPlane(Clone)");
        if (exact != null)
            return exact.transform;

        exact = GameObject.Find("SlicingPlane");
        if (exact != null)
            return exact.transform;

        if (string.IsNullOrWhiteSpace(slicingPlaneNameContains))
            return null;

        Transform[] transforms = FindObjectsByType<Transform>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        for (int i = 0; i < transforms.Length; i++)
        {
            Transform t = transforms[i];
            if (t != null && t.name != null && t.name.Contains(slicingPlaneNameContains))
                return t;
        }

        return null;
    }

    private void SetTextIfChanged(string s)
    {
        if (lastRendered == s)
            return;
        lastRendered = s;
        angleText.text = s;
    }

    private static Sprite CreateSlicingPlaneIconSprite()
    {
        const int size = 64;
        Texture2D tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;

        Color clear = new Color(0f, 0f, 0f, 0f);
        Color border = new Color(1f, 1f, 1f, 1f);
        Color fill = new Color(1f, 1f, 1f, 0.10f);
        Color cut = new Color(0.25f, 0.85f, 1f, 1f);

        Color[] pixels = new Color[size * size];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = clear;

        int min = 10;
        int max = size - 11;
        for (int y = min; y <= max; y++)
        {
            for (int x = min; x <= max; x++)
            {
                bool isBorder = (x == min || x == max || y == min || y == max);
                pixels[y * size + x] = isBorder ? border : fill;
            }
        }

        for (int i = 0; i < 44; i++)
        {
            int x = min + 3 + i;
            int y = max - 3 - i;
            if (x < min || x > max || y < min || y > max)
                continue;

            for (int t = -1; t <= 1; t++)
            {
                int yy = Mathf.Clamp(y + t, min, max);
                pixels[yy * size + x] = cut;
            }
        }

        tex.SetPixels(pixels);
        tex.Apply(false, true);

        return Sprite.Create(tex, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), 100f);
    }
}

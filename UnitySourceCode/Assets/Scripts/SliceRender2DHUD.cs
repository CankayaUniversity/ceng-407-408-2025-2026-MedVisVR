using UnityEngine;
using UnityVolumeRendering;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Runtime (VR) 2D Slice Render HUD.
/// Sahnedeki SlicingPlane(Clone)'un paylastigi SliceRenderingShader materyalini
/// bir Quad uzerine basarak, acilar HUD'unun ustunde 2D kesit goruntusunu gosterir.
///
/// SlicingPlane.cs her frame materyalin _planeMat ve _parentInverseMat uniformlarini
/// guncelledigi icin paylasilan materyalle ayni kesit otomatik olarak bu quad'da da gorunur.
/// </summary>
public class SliceRender2DHUD : MonoBehaviour
{
    private const string HudRootName = "[HUD] Slice Render 2D";

    [Header("Debug")]
    [SerializeField] private bool verbose = false;
    [SerializeField] private bool matchCameraLayer = true;

    [Header("Screen Anchor")]
    [Tooltip("HUD'u kameranin goruntusune gore sabitler.")]
    [SerializeField] private bool anchorToViewport = true;

    [Tooltip("Viewport koordinati (0..1). Acilar HUD'u ile birlikte gozluk goruntusunun icinde kalacak sekilde ortalanir.")]
    [SerializeField] private Vector2 viewportAnchor = new Vector2(0.58f, 0.46f);

    [Tooltip("Kameradan uzaklik (metre). Acilar HUD ile ayni olmali ki ayni duzlemde gorunsunler.")]
    [SerializeField] private float viewportDistance = 1.0f;

    [Header("Quad Placement (camera local)")]
    [SerializeField] private Vector3 hudLocalPosition = new Vector3(0.12f, 0.09f, 0.60f);
    [SerializeField] private bool faceCamera = true;
    [Tooltip("XR kurulumlarinda on/arka yuz ters olabilir. Goruntu ters gorunuyorsa degistirin.")]
    [SerializeField] private bool invertFacing = true;

    [Tooltip("Quad'in dunya uzerindeki kenar uzunlugu (metre).")]
    [SerializeField] private float quadWorldSize = 0.36f;

    [Header("Monitor Frame")]
    [SerializeField] private bool createMonitorFrame = false;
    [SerializeField] private float frameDepth = 0.006f;
    [SerializeField] private float bezelThickness = 0.014f;
    [SerializeField] private float screenInset = 0.002f;
    [SerializeField] private Color frameColor = new Color(0.08f, 0.09f, 0.10f, 1f);
    [SerializeField] private Color bezelEdgeColor = new Color(0.24f, 0.25f, 0.27f, 1f);

    [Header("Target")]
    [SerializeField] private string slicingPlaneNameContains = "SlicingPlane";

    [Header("Update")]
    [SerializeField] private float searchIntervalSec = 0.5f;

    private Camera hudCamera;
    private MeshRenderer quadRenderer;
    private Transform quadTransform;
    private Transform frameRoot;
    private SlicingPlane targetPlane;
    private Material hudMaterial;
    private Material frameMaterial;
    private Material edgeMaterial;
    private float nextSearchTime;

    private static readonly int IdDataTex = Shader.PropertyToID("_DataTex");
    private static readonly int IdTFTex = Shader.PropertyToID("_TFTex");
    private static readonly int IdParentInverseMat = Shader.PropertyToID("_parentInverseMat");
    private static readonly int IdPlaneMat = Shader.PropertyToID("_planeMat");

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Bootstrap()
    {
        if (FindFirstObjectByType<SliceRender2DHUD>() != null)
            return;

        GameObject go = new GameObject(HudRootName);
        DontDestroyOnLoad(go);
        go.AddComponent<SliceRender2DHUD>();
    }

    private void Awake()
    {
        SetupQuad();
        if (verbose) Debug.Log("[SliceRender2DHUD] HUD olusturuldu.");
    }

    private void Update()
    {
        EnsureCameraAttachment();
        UpdateAnchorPosition();
        EnsureMaterialBinding();
    }

    private void SetupQuad()
    {
        if (createMonitorFrame)
            CreateMonitorFrame();

        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "SliceQuad";

        Collider col = quad.GetComponent<Collider>();
        if (col != null) Destroy(col);

        quad.transform.SetParent(frameRoot != null ? frameRoot : transform, false);
        quad.transform.localPosition = Vector3.zero;
        quad.transform.localRotation = Quaternion.identity;
        quad.transform.localScale = new Vector3(
            Mathf.Max(0.01f, quadWorldSize - bezelThickness * 2f - screenInset * 2f),
            Mathf.Max(0.01f, quadWorldSize - bezelThickness * 2f - screenInset * 2f),
            1f);

        quadTransform = quad.transform;
        quadRenderer = quad.GetComponent<MeshRenderer>();
        quadRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        quadRenderer.receiveShadows = false;
        quadRenderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
        quadRenderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
        quadRenderer.allowOcclusionWhenDynamic = false;
        quadRenderer.sortingOrder = 32000;

        // HUD-specific shader (ZTest Always, Queue=Overlay) olustur.
        Shader hudShader = Shader.Find("VolumeRendering/HUD/SliceRender2DHUD");
        if (hudShader == null)
        {
            Debug.LogError("[SliceRender2DHUD] HUD shader bulunamadi: VolumeRendering/HUD/SliceRender2DHUD");
            quadRenderer.enabled = false;
            return;
        }
        hudMaterial = new Material(hudShader);
        hudMaterial.renderQueue = 4000; // Overlay uzeri
        quadRenderer.sharedMaterial = hudMaterial;
        quadRenderer.enabled = false;
    }

    private void CreateMonitorFrame()
    {
        frameMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        frameMaterial.color = frameColor;
        edgeMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        edgeMaterial.color = bezelEdgeColor;

        frameRoot = new GameObject("MonitorFrame").transform;
        frameRoot.SetParent(transform, false);
        frameRoot.localPosition = Vector3.zero;
        frameRoot.localRotation = Quaternion.identity;

        CreateFramePart("BackPlate", frameRoot, Vector3.zero, new Vector3(quadWorldSize, quadWorldSize, frameDepth), frameMaterial);
        CreateFramePart("TopBezel", frameRoot, new Vector3(0f, (quadWorldSize - bezelThickness) * 0.5f, -frameDepth * 0.25f), new Vector3(quadWorldSize, bezelThickness, frameDepth * 0.6f), edgeMaterial);
        CreateFramePart("BottomBezel", frameRoot, new Vector3(0f, -(quadWorldSize - bezelThickness) * 0.5f, -frameDepth * 0.25f), new Vector3(quadWorldSize, bezelThickness, frameDepth * 0.6f), edgeMaterial);
        CreateFramePart("LeftBezel", frameRoot, new Vector3(-(quadWorldSize - bezelThickness) * 0.5f, 0f, -frameDepth * 0.25f), new Vector3(bezelThickness, quadWorldSize - bezelThickness * 2f, frameDepth * 0.6f), edgeMaterial);
        CreateFramePart("RightBezel", frameRoot, new Vector3((quadWorldSize - bezelThickness) * 0.5f, 0f, -frameDepth * 0.25f), new Vector3(bezelThickness, quadWorldSize - bezelThickness * 2f, frameDepth * 0.6f), edgeMaterial);
    }

    private static void CreateFramePart(string name, Transform parent, Vector3 localPosition, Vector3 localScale, Material material)
    {
        GameObject part = GameObject.CreatePrimitive(PrimitiveType.Cube);
        part.name = name;
        Collider col = part.GetComponent<Collider>();
        if (col != null) Object.Destroy(col);

        part.transform.SetParent(parent, false);
        part.transform.localPosition = localPosition;
        part.transform.localRotation = Quaternion.identity;
        part.transform.localScale = localScale;

        MeshRenderer renderer = part.GetComponent<MeshRenderer>();
        renderer.sharedMaterial = material;
        renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
        renderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
    }

    private void EnsureMaterialBinding()
    {
        if (quadRenderer == null || hudMaterial == null)
            return;

        if (targetPlane == null)
        {
            if (Time.unscaledTime < nextSearchTime)
                return;
            nextSearchTime = Time.unscaledTime + searchIntervalSec;

            targetPlane = TryFindSlicingPlane();
            if (targetPlane == null)
            {
                quadRenderer.enabled = false;
                return;
            }
        }

        MeshRenderer planeRenderer = targetPlane.GetComponent<MeshRenderer>();
        if (planeRenderer == null || planeRenderer.sharedMaterial == null)
        {
            quadRenderer.enabled = false;
            return;
        }

        Material src = planeRenderer.sharedMaterial;

        // Volume verisini ve slicing plane matrislerini her frame kopyala.
        Texture dataTex = src.HasProperty(IdDataTex) ? src.GetTexture(IdDataTex) : null;
        Texture tfTex = src.HasProperty(IdTFTex) ? src.GetTexture(IdTFTex) : null;
        if (dataTex == null || tfTex == null)
        {
            quadRenderer.enabled = false;
            return;
        }

        hudMaterial.SetTexture(IdDataTex, dataTex);
        hudMaterial.SetTexture(IdTFTex, tfTex);
        hudMaterial.SetMatrix(IdParentInverseMat, src.GetMatrix(IdParentInverseMat));
        hudMaterial.SetMatrix(IdPlaneMat, src.GetMatrix(IdPlaneMat));

        quadRenderer.enabled = true;
    }

    private void OnDestroy()
    {
        if (hudMaterial != null)
        {
            Destroy(hudMaterial);
            hudMaterial = null;
        }
        if (frameMaterial != null)
        {
            Destroy(frameMaterial);
            frameMaterial = null;
        }
        if (edgeMaterial != null)
        {
            Destroy(edgeMaterial);
            edgeMaterial = null;
        }
    }

    private SlicingPlane TryFindSlicingPlane()
    {
        SlicingPlane[] planes = FindObjectsByType<SlicingPlane>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        if (planes == null || planes.Length == 0)
            return null;

        if (!string.IsNullOrWhiteSpace(slicingPlaneNameContains))
        {
            for (int i = 0; i < planes.Length; i++)
            {
                if (planes[i] != null && planes[i].name != null && planes[i].name.Contains(slicingPlaneNameContains))
                    return planes[i];
            }
        }
        return planes[0];
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

        if (matchCameraLayer)
            SetLayerRecursively(gameObject, hudCamera.gameObject.layer);

        if (verbose && cameraChanged) Debug.Log($"[SliceRender2DHUD] Kamera baglandi: {hudCamera.name}");
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
}

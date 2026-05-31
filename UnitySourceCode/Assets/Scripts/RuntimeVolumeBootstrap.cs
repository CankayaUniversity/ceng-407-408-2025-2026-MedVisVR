using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityVolumeRendering;

/// <summary>
/// Build alindiktan sonra APK'ya dokunmadan runtime'da NIFTI yukleyip
/// tum volume rendering setup'ini koddan kuran bootstrap.
///
/// Akis:
///   1. Application.persistentDataPath altinda "volume.nii" / "volume.nii.gz" ara.
///   2. NiftiImporter.ImportAsync ile datasetle.
///   3. VolumeObjectFactory.CreateObjectAsync ile VolumeRenderedObject olustur.
///   4. Yeni volume icin yeni bir Sphere GameObject insa et. Eski sahnedeki sphere
///      SABLON olarak kullanilir; gorunus/fizik ayarlari deterministik bicimde kopyalanir.
///   5. CreateSlicingPlane + CrossSectionPlane + SlicingPlaneFollowSphere
///      attach (yeni sphere'e baglanir).
///   6. CTLungTransferFunction + Quest3ShaderOptimizer + TransferFunctionToggle AddComponent.
///   7. Volume transform'unu (pozisyon/olcek) kullanici ayarlarina uygula.
///   8. Loading HUD'u gizle.
///
/// SliceRender2DHUD ve SlicingPlaneAngleHUD kendi [RuntimeInitializeOnLoadMethod]
/// bootstrap'leriyle zaten sahneye eklenir; slicing plane olustugunda otomatik bulurlar.
/// </summary>
public class RuntimeVolumeBootstrap : MonoBehaviour
{
    private static readonly Quaternion SlicingPlaneXzWorldRotation = Quaternion.identity;
    private static readonly Quaternion CrossSectionPlaneXzWorldRotation = Quaternion.Euler(270f, 0f, 0f);

    private enum StartupTransferFunction
    {
        Default = 0,
        CTLung = 1,
        CTCardiac3 = 2,
        CTChestContrastEnhanced = 3
    }

    [Header("NIFTI Dosya Arama")]
    [Tooltip("Mutlak dosya yolu. Doluysa ve dosya mevcutsa DIGER TUM arama gecersiz olur. " +
             "Editor'da veya Windows standalone'da C:\\... yolu verebilirsin. " +
             "Android/Quest'te cogunlukla bos birakilir (persistentDataPath kullanilir).")]
    [SerializeField] private string absolutePathOverride = "";

    [Tooltip("persistentDataPath / streamingAssetsPath altinda denenecek dosya isimleri (siralama onemli).")]
    [SerializeField]
    private string[] candidateFileNames = new string[]
    {
        "volume.nii",
        "volume.nii.gz"
    };

    [Tooltip("persistentDataPath'a ek olarak StreamingAssets'i de ara (APK icine gomulu dosya icin).")]
    [SerializeField] private bool alsoSearchStreamingAssets = true;

    [Header("Sphere Referansi")]
    [Tooltip("Sablon (template) Sphere GameObject. Runtime'da bu SABLONA bakilarak " +
             "yeni bir Sphere sifirdan insa edilir, orijinali kapatilir. Bos birakilirsa " +
             "sphereSearchName ile sahnede aranir.")]
    [SerializeField] private Transform sphereReference;

    [Tooltip("sphereReference null ise bu isimle sahnede aranir.")]
    [SerializeField] private string sphereSearchName = "Sphere";

    [Tooltip("Yeni volume icin yeni bir Sphere GameObject insa et. " +
             "Orijinal sphere SABLON olarak kullanilir; gorunur mesh/materyal ve temel fizik " +
             "ayarlari kopyalanir. Sonuc: eski volume'a bagli referans tasimayan bagimsiz nesne.")]
    [SerializeField] private bool buildFreshSphere = true;

    [Tooltip("Yeni sphere insa edildikten sonra orijinal Sphere GO'su kapatilsin mi. " +
             "Orijinal eski volume setup'ina bagli oldugu icin genellikle kapatilir.")]
    [SerializeField] private bool disableOriginalSphere = true;

    [Tooltip("Yeni sphere'in volume'a gore world konum offseti. Volume konumuna eklenir. " +
             "targetLargestEdgeMeters=0.35 icin sphere volume kenarinin biraz disinda olmalidir.")]
    [SerializeField] private Vector3 runtimeSphereOffsetFromVolume = new Vector3(0.18f, 0f, -0.05f);

    [Tooltip("Sphere kavrama yaricapi (metre). Volume'dan kucuk olmali ki el sphere'e " +
             "yaklastiginda volume yerine sphere kavransin. Tipik degerler: 0.08-0.15.")]
    [SerializeField, Range(0.03f, 0.5f)] private float sphereGrabRadius = 0.12f;

    [Tooltip("Runtime'da volume'a clipping yapan CrossSectionPlane olustur. " +
             "AÇIK: plane'in lokal +Z tarafindaki voxeller silinir (yarim volume goruntusu). " +
             "KAPALI (DEFAULT): clipping yok, volume tamami goruntulenir - volume goruntulenememe " +
             "sorunlarinin onune gecer. Slicing plane (rendering) ZATEN sphere'i takip eder; " +
             "clipping plane sadece volume'u kesmek/yarim acmak icindir.")]
    [SerializeField] private bool createRuntimeCrossSectionPlane = false;

    [Tooltip("CrossSectionPlane'in sphere'e gore lokal offseti. Varsayilan sifir: plane merkezi sphere merkeziyle cakisir.")]
    [SerializeField] private Vector3 crossSectionPlaneLocalOffsetFromSphere = Vector3.zero;

    [Tooltip("Ana sahnedeki Sphere altindaki mevcut CrossSectionPlane'in lokal pozisyonunu da birebir kullan.")]
    [SerializeField] private bool useTemplateCrossSectionPlaneOffset = false;

    [Tooltip("Runtime'da olusan CrossSectionPlane ilk karede volume merkezine yerlestirilsin. " +
             "Bu sayede volume tamamen clip edilip gorunmez hale gelmez.")]
    [SerializeField] private bool centerCrossSectionPlaneOnVolumeAtStart = true;

    [Header("Volume Yerlesimi")]
    [Tooltip("VolumeRenderedObject world konumu. Kullanicinin onunde, kavrama mesafesinde olmali.")]
    [SerializeField] private Vector3 volumeWorldPosition = new Vector3(0f, 1.3f, 0.5f);

    [Tooltip("VolumeRenderedObject world rotasyonu (euler).")]
    [SerializeField] private Vector3 volumeWorldEuler = new Vector3(0f, 0f, 0f);

    [Tooltip("Acikken: volume bounding box'inin en uzun kenari otomatik olarak " +
             "targetLargestEdgeMeters'e getirilir (NormaliseScale + bounds-based fit). " +
             "Bu, NIFTI voxel boyutu ne olursa olsun deterministik el-boyutu uretir. " +
             "Kapatilirsa volumeUniformScale dogrudan uygulanir (eski davranis).")]
    [SerializeField] private bool autoFitToHandSize = true;

    [Tooltip("Auto-fit aciksa: volume'in en uzun kenari bu metre degerine getirilir. " +
             "0.25-0.45 araligi VR'da elle kavramak icin idealdir.")]
    [SerializeField, Range(0.1f, 1.5f)] private float targetLargestEdgeMeters = 0.35f;

    [Tooltip("Auto-fit KAPALIYKEN kullanilan dogrudan scale carpani. " +
             "Tipik NIFTI CT icin ~0.001 elle kullanilabilir boyut verir.")]
    [SerializeField] private float volumeUniformScale = 0.001f;

    [Tooltip("Volume'un el/controller ile tutulma yaricapi. Sphere'den buyuk olabilir.")]
    [SerializeField, Range(0.05f, 1.5f)] private float volumeGrabRadius = 0.28f;

    [Header("Runtime Gorunurluk")]
    [Tooltip("Volume olustuktan hemen sonra sahnede gorunur bir transfer function uygula. " +
             "2D HUD renk gosterse bile alpha sifirsa 3D volume tamamen saydam kalabilir.")]
    [SerializeField] private bool applyStartupTransferFunction = true;

    [Tooltip("Baslangicta uygulanacak TF. Bu projedeki lung_only NIFTI icin CT-Lung daha guvenli bir ilk gorunum verir.")]
    [SerializeField] private StartupTransferFunction startupTransferFunction = StartupTransferFunction.CTLung;

    [Tooltip("Acikken baslangicta dataset araligindan bagimsiz, kesin gorunur bir TF kullanir. " +
             "B tusu ile preset'lere gecis yine calisir.")]
    [SerializeField] private bool useVisibleStartupTransferFunction = true;

    [Tooltip("Secilen TF dataset araligina uymayip alpha tamamen sifir kalirsa, gorunur bir fallback TF uret.")]
    [SerializeField] private bool fallbackToVisibleTransferFunction = true;

    [Tooltip("World-space SlicingPlane shader'indaki _SceneAlpha degeri. 0 ise plane datayi uretir ama sahnede gorunmez.")]
    [SerializeField, Range(0.05f, 1f)] private float slicingPlaneSceneAlpha = 0.85f;

    [Tooltip("Runtime'da uretilen volume/slicing renderer'larini aktif ve gorunur durumda tut.")]
    [SerializeField] private bool forceRuntimeRenderersVisible = true;

    [Tooltip("Sphere handle template'ten saydam/gizli material kopyalasa bile runtime'da gorunur yap.")]
    [SerializeField] private bool forceRuntimeSphereVisible = true;

    [SerializeField] private Color runtimeSphereColor = new Color(0.1f, 0.85f, 1f, 1f);

    [Header("Shader Ayarlari")]
    [Tooltip("Android'de kullanilacak sampling rate carpani (Quest3ShaderOptimizer ile tutarli).")]
    [SerializeField, Range(0.2f, 2.0f)] private float mobileSamplingRate = 0.65f;

    [SerializeField] private bool disableTricubicOnMobile = true;

    [Header("Davranis")]
    [Tooltip("Start'ta otomatik yukleme baslasin mi.")]
    [SerializeField] private bool autoLoadOnStart = true;

    [Tooltip("Yeni runtime VolumeRenderedObject basariyla olustuktan sonra sahneye elle eklenmis " +
             "eski VolumeRenderedObject'leri devre disi birakir. NIFTI bulunamazsa/okunamazsa " +
             "eski sahne volume'u gorunur kalir.")]
    [SerializeField] private bool disableExistingVolumesOnStart = true;

    [Tooltip("Ayrintili log.")]
    [SerializeField] private bool verbose = true;

    private bool loadingInProgress;
    private VolumeRenderedObject createdVolume;
    private Transform runtimeSphere; // sifirdan insa edilmis yeni sphere (veya orijinal)
    private Transform runtimeCrossSectionPlane;

    private void Start()
    {
        Log($"v3 visibility bootstrap: visibleStartupTF={useVisibleStartupTransferFunction}, forceSphere={forceRuntimeSphereVisible}");
        if (autoLoadOnStart)
            _ = LoadAsyncSafe();
    }

    // Fire-and-forget wrapper: exception'lari yakalayip HUD'a yansitir ve
    // HUD'in her halde kapatildigindan emin olur.
    private async Task LoadAsyncSafe()
    {
        bool volumeCreated = false;
        try
        {
            volumeCreated = await LoadAsync();
        }
        catch (System.Exception e)
        {
            Debug.LogException(e);
            ReportError("Beklenmeyen hata:\n" + e.Message);
        }
        finally
        {
            // Volume basariyla olustuysa HUD mutlaka gizlensin
            // (sonraki post-processing adimlarinda hata olsa bile).
            if (volumeCreated && RuntimeVolumeLoadingHUD.Instance != null)
            {
                await Task.Delay(300);
                RuntimeVolumeLoadingHUD.Instance.Hide();
            }
        }
    }

    public async Task<bool> LoadAsync()
    {
        if (loadingInProgress)
        {
            LogWarn("Yukleme zaten devam ediyor.");
            return false;
        }
        loadingInProgress = true;

        try
        {
            ReportStatus("Baslatiliyor...");

            string filePath = ResolveNiftiPath();
            if (filePath == null)
            {
                string searched = string.Join(", ", candidateFileNames);
                string msg = $"NIFTI dosyasi bulunamadi.\n\nKlasor:\n{Application.persistentDataPath}\n\nAranan: {searched}\n\nMevcut sahne volume'u korunuyor.\n\nornek:\nadb push volume.nii \"{Application.persistentDataPath}/\"";
                ReportError(msg);
                Debug.LogError("[RuntimeVolumeBootstrap] " + msg);
                return false;
            }

            // Dosya zaman damgasini logla - push sonrasi dogru dosyanin yuklendigini gormek icin kritik
            try
            {
                System.IO.FileInfo fi = new System.IO.FileInfo(filePath);
                Log($"NIFTI dosyasi bulundu: {filePath}");
                Log($"  Boyut : {fi.Length / 1024 / 1024} MB");
                Log($"  Tarih : {fi.LastWriteTime:yyyy-MM-dd HH:mm:ss}");
            }
            catch { }

            ReportStatus("NIFTI okunuyor...\n" + Path.GetFileName(filePath));

            // --- Adim 1: Dataset import ---
            IImageFileImporter importer = ImporterFactory.CreateImageFileImporter(ImageFileFormat.NIFTI);
            if (importer == null)
            {
                ReportError("NIFTI importer olusturulamadi.");
                return false;
            }

            VolumeDataset dataset = null;
            try
            {
                dataset = await importer.ImportAsync(filePath);
            }
            catch (System.Exception e)
            {
                ReportError("NIFTI okuma hatasi:\n" + e.Message);
                Debug.LogException(e);
                return false;
            }

            if (dataset == null)
            {
                ReportError("Dataset yuklenemedi (null dondu).");
                return false;
            }
            Log($"Dataset yuklendi: {dataset.datasetName} ({dataset.dimX}x{dataset.dimY}x{dataset.dimZ})");

            // --- Adim 2: VolumeRenderedObject olustur ---
            ReportStatus("Volume olusturuluyor...\n" + dataset.datasetName);

            BootstrapProgressHandler progress = new BootstrapProgressHandler(this, dataset.datasetName);
            VolumeRenderedObject volObj = null;
            try
            {
                volObj = await VolumeObjectFactory.CreateObjectAsync(dataset, progress);
            }
            catch (System.Exception e)
            {
                ReportError("VolumeRenderedObject olusturulamadi:\n" + e.Message);
                Debug.LogException(e);
                return false;
            }

            if (volObj == null)
            {
                ReportError("VolumeRenderedObject null dondu.");
                return false;
            }

            createdVolume = volObj;
            Log("VolumeRenderedObject olusturuldu.");

            // Yeni runtime volume basariyla olustuktan sonra eski sahne volume'larini
            // devre disi birak. Dosya yoksa/import veya factory basarisizsa eski setup
            // gorunur kalir ve uygulama bos ekrana dusmez.
            if (disableExistingVolumesOnStart)
                DisablePreExistingVolumes();

            // ----- VOLUME OLUSTU: bundan sonraki adimlar hata verse bile
            // HUD gizlensin ve volume kullanilabilir kalsin. Her birini
            // ayri try/catch ile saralim ki tek bir hata zinciri kirmasin.

            // --- Adim 3: Volume transform ---
            try
            {
                volObj.transform.position = volumeWorldPosition;
                volObj.transform.rotation = Quaternion.Euler(volumeWorldEuler);

                if (autoFitToHandSize)
                {
                    // 1) Voxel anizotropisini dogru oranla (dataset.scale'e gore normalize).
                    volObj.NormaliseScale();

                    // 2) World bounds'a bakip en uzun kenari hedef metreye fit et.
                    //    Bu, NIFTI dataset'i ne olursa olsun deterministik bir VR boyutu verir.
                    Physics.SyncTransforms();
                    if (TryComputeFitScale(volObj, targetLargestEdgeMeters, out float fitScale))
                    {
                        volObj.transform.localScale *= fitScale;
                        Log($"Auto-fit: targetEdge={targetLargestEdgeMeters}m, fitScale={fitScale:F4}, " +
                            $"finalScale={volObj.transform.localScale}");
                    }
                    else
                    {
                        // Bounds hazir degilse volumeUniformScale ile 1mm'e ezmek volume'u gorunmez yapardi.
                        // NormaliseScale sonucunu koru (en azindan voxel oranlari dogru, kullanici grab ile buyutebilir).
                        LogWarn($"Volume bounds hesaplanamadi; NormaliseScale sonucu korundu: scale={volObj.transform.localScale}");
                    }
                }
                else
                {
                    volObj.transform.localScale = Vector3.one * volumeUniformScale;
                }

                // Final scale'i NaN/sifir/devasa olmasi durumuna karsi koru.
                Vector3 fs = volObj.transform.localScale;
                bool bad = float.IsNaN(fs.x) || float.IsNaN(fs.y) || float.IsNaN(fs.z) ||
                           float.IsInfinity(fs.x) || float.IsInfinity(fs.y) || float.IsInfinity(fs.z) ||
                           Mathf.Abs(fs.x) < 1e-6f || Mathf.Abs(fs.y) < 1e-6f || Mathf.Abs(fs.z) < 1e-6f ||
                           Mathf.Abs(fs.x) > 100f || Mathf.Abs(fs.y) > 100f || Mathf.Abs(fs.z) > 100f;
                if (bad)
                {
                    LogWarn($"Volume scale gecersiz (NaN/0/devasa): {fs}. 1.0'a sifirlaniyor.");
                    volObj.transform.localScale = Vector3.one;
                }

                // Tani: gercek world bounds + pozisyon, logcat'te volume nerede onu gormek icin.
                Renderer[] diagRenderers = volObj.GetComponentsInChildren<Renderer>(includeInactive: false);
                if (diagRenderers != null && diagRenderers.Length > 0)
                {
                    Bounds wb = diagRenderers[0].bounds;
                    for (int i = 1; i < diagRenderers.Length; i++)
                        if (diagRenderers[i] != null) wb.Encapsulate(diagRenderers[i].bounds);
                    Log($"DIAG: volume worldPos={volObj.transform.position}, scale={volObj.transform.localScale}, bounds.center={wb.center}, bounds.size={wb.size}");
                }
                else
                {
                    LogWarn("DIAG: volume hicbir Renderer barindirmiyor - gorunmemesi normal.");
                }
            }
            catch (System.Exception e) { Debug.LogException(e); }

            // --- Adim 4: Transfer function + renderer gorunurlugu ---
            try
            {
                ReportStatus("Transfer function ve gorunurluk ayarlaniyor...");
                AttachVolumeScripts(volObj);
                ApplyStartupTransferFunction(volObj);
                EnsureVisibleTransferFunctionAlpha(volObj, useVisibleStartupTransferFunction);
                ConfigureVolumeRendererVisibility(volObj);
            }
            catch (System.Exception e)
            {
                Debug.LogException(e);
                LogWarn("Transfer function/gorunurluk ayarlanirken hata (devam ediliyor): " + e.Message);
            }

            // --- Adim 5: SlicingPlane olustur + sphere follow ---
            try
            {
                ReportStatus("Slicing plane kuruluyor...");
                SlicingPlane slicingPlane = volObj.CreateSlicingPlane();
                if (slicingPlane == null)
                {
                    LogWarn("CreateSlicingPlane null dondu.");
                }
                else
                {
                    // Plane scale paket varsayilani 0.1 olarak kalir - shader'in
                    // uvMod*10 sample alani volume bbox'ina tam denk gelir.
                    ConfigureSlicingPlaneVisibility(slicingPlane);
                    Transform crossSectionPlane = CreateRuntimeCrossSectionPlane(volObj, slicingPlane.transform);
                    BuildRuntimePlaneRig(slicingPlane.transform, crossSectionPlane);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogException(e);
                LogWarn("SlicingPlane kurulumunda hata (devam ediliyor): " + e.Message);
            }

            // --- Adim 6: Shader sampling rate ---
            try
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                volObj.SetSamplingRateMultiplier(mobileSamplingRate);
                if (disableTricubicOnMobile)
                    volObj.SetCubicInterpolationEnabled(false);
#endif
            }
            catch (System.Exception e)
            {
                Debug.LogException(e);
                LogWarn("Sampling rate ayarlanirken hata (devam ediliyor): " + e.Message);
            }

            ReportStatus("Hazir.");
            Log("Pipeline tamamlandi.");
            return true;
        }
        finally
        {
            loadingInProgress = false;
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    /// <summary>
    /// Volume'un dunya uzayindaki bounding box'una bakip, en uzun kenarini
    /// targetEdge metreye getirecek olcek carpanini hesaplar. Boylece dataset
    /// boyutu ne olursa olsun deterministik bir VR el-boyutu uretilir.
    /// </summary>
    private bool TryComputeFitScale(VolumeRenderedObject volObj, float targetEdge, out float fitScale)
    {
        fitScale = 1f;
        if (volObj == null || targetEdge <= 0.0001f)
            return false;

        // Once volObj.meshRenderer field'ini dene (factory tarafindan atanmis olur).
        Bounds bounds;
        bool haveBounds = false;

        if (volObj.meshRenderer != null && volObj.meshRenderer.enabled)
        {
            bounds = volObj.meshRenderer.bounds;
            haveBounds = true;
        }
        else
        {
            // Fallback: tum child renderer'lari topla.
            Renderer[] rends = volObj.GetComponentsInChildren<Renderer>(includeInactive: false);
            if (rends == null || rends.Length == 0)
                return false;

            bounds = rends[0].bounds;
            for (int i = 1; i < rends.Length; i++)
            {
                if (rends[i] != null)
                    bounds.Encapsulate(rends[i].bounds);
            }
            haveBounds = true;
        }

        if (!haveBounds)
            return false;

        float largest = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
        if (largest < 0.0001f)
            return false;

        fitScale = targetEdge / largest;
        return true;
    }

    private void ApplyStartupTransferFunction(VolumeRenderedObject volObj)
    {
        if (!applyStartupTransferFunction || volObj == null)
            return;

        bool applied = false;
        switch (startupTransferFunction)
        {
            case StartupTransferFunction.CTLung:
                applied = volObj.GetComponent<CTLungTransferFunction>()?.TryApplyCTLung(verbose) == true;
                break;
            case StartupTransferFunction.CTCardiac3:
                applied = volObj.GetComponent<CTCardiac3TransferFunction>()?.TryApplyCTCardiac3(verbose) == true;
                break;
            case StartupTransferFunction.CTChestContrastEnhanced:
                applied = volObj.GetComponent<CTChestContrastEnhancedTransferFunction>()?.TryApplyCTChestContrastEnhanced(verbose) == true;
                break;
            case StartupTransferFunction.Default:
            default:
                if (volObj.transferFunction != null)
                {
                    volObj.transferFunction.GenerateTexture();
                    applied = true;
                }
                break;
        }

        if (applied)
        {
            volObj.UpdateMaterialProperties();
            Log($"Baslangic transfer function uygulandi: {startupTransferFunction}");
        }
        else
        {
            LogWarn($"Baslangic transfer function uygulanamadi: {startupTransferFunction}");
        }
    }

    private void EnsureVisibleTransferFunctionAlpha(VolumeRenderedObject volObj, bool forceVisible)
    {
        if ((!fallbackToVisibleTransferFunction && !forceVisible) || volObj == null || volObj.transferFunction == null)
            return;

        if (!forceVisible)
        {
            float maxAlpha = 0f;
            for (int i = 0; i < volObj.transferFunction.alphaControlPoints.Count; i++)
                maxAlpha = Mathf.Max(maxAlpha, volObj.transferFunction.alphaControlPoints[i].alphaValue);

            if (maxAlpha > 0.001f)
                return;
        }

        UnityVolumeRendering.TransferFunction tf = volObj.transferFunction;
        tf.colourControlPoints.Clear();
        tf.alphaControlPoints.Clear();

        tf.AddControlPoint(new TFColourControlPoint(0.00f, new Color(0.04f, 0.08f, 0.14f, 1f)));
        tf.AddControlPoint(new TFColourControlPoint(0.20f, new Color(0.10f, 0.55f, 0.95f, 1f)));
        tf.AddControlPoint(new TFColourControlPoint(0.62f, new Color(1.00f, 0.78f, 0.28f, 1f)));
        tf.AddControlPoint(new TFColourControlPoint(1.00f, Color.white));

        tf.AddControlPoint(new TFAlphaControlPoint(0.00f, 0.00f));
        tf.AddControlPoint(new TFAlphaControlPoint(0.02f, 0.08f));
        tf.AddControlPoint(new TFAlphaControlPoint(0.25f, 0.22f));
        tf.AddControlPoint(new TFAlphaControlPoint(0.70f, 0.45f));
        tf.AddControlPoint(new TFAlphaControlPoint(1.00f, 0.55f));
        tf.GenerateTexture();
        volObj.UpdateMaterialProperties();

        VolumeDataset dataset = volObj.dataset;
        string range = dataset != null ? $" range=[{dataset.GetMinDataValue()}, {dataset.GetMaxDataValue()}]" : "";
        LogWarn((forceVisible
            ? "Visible startup TF uygulandi."
            : "Secilen TF alpha uretmedi; runtime visible fallback TF uygulandi.") + range);
    }

    private void ConfigureVolumeRendererVisibility(VolumeRenderedObject volObj)
    {
        if (!forceRuntimeRenderersVisible || volObj == null)
            return;

        volObj.gameObject.SetActive(true);
        if (volObj.volumeContainerObject != null)
            volObj.volumeContainerObject.SetActive(true);

        MeshRenderer renderer = volObj.meshRenderer != null
            ? volObj.meshRenderer
            : volObj.GetComponentInChildren<MeshRenderer>(includeInactive: true);

        if (renderer == null)
        {
            LogWarn("Volume renderer bulunamadi; 3D volume sahnede cizilemez.");
            return;
        }

        renderer.enabled = true;
        renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.allowOcclusionWhenDynamic = false;

        Material mat = renderer.sharedMaterial;
        if (mat != null)
        {
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            if (!createRuntimeCrossSectionPlane)
                mat.DisableKeyword("CROSS_SECTION_ON");
            Log($"Volume renderer gorunur: shader='{mat.shader.name}', queue={mat.renderQueue}, layer={renderer.gameObject.layer}");
        }
    }

    private void ConfigureSlicingPlaneVisibility(SlicingPlane slicingPlane)
    {
        if (!forceRuntimeRenderersVisible || slicingPlane == null)
            return;

        MeshRenderer renderer = slicingPlane.GetComponent<MeshRenderer>();
        if (renderer == null)
        {
            LogWarn("SlicingPlane MeshRenderer bulunamadi.");
            return;
        }

        renderer.enabled = true;
        renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.allowOcclusionWhenDynamic = false;

        Material mat = renderer.sharedMaterial;
        if (mat != null)
        {
            if (mat.HasProperty("_SceneAlpha"))
                mat.SetFloat("_SceneAlpha", slicingPlaneSceneAlpha);
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent + 10;
            Log($"SlicingPlane gorunur: SceneAlpha={slicingPlaneSceneAlpha:F2}, shader='{mat.shader.name}', queue={mat.renderQueue}");
        }
    }

    private void DisablePreExistingVolumes()
    {
        // Sahnede onceden var olan tum VolumeRenderedObject root GO'lari bul
        // ve kapat. Inactive olanlari da yakalamak icin FindObjectsInactive.Include.
        VolumeRenderedObject[] existing = FindObjectsByType<VolumeRenderedObject>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None);

        int disabledCount = 0;
        for (int i = 0; i < existing.Length; i++)
        {
            VolumeRenderedObject v = existing[i];
            if (v == null)
                continue;
            // Yeni runtime volume bu noktada zaten mevcut olabilir; onu kapatma.
            if (createdVolume != null && v == createdVolume)
                continue;

            // Sadece aktif olanlari kapat.
            if (v.gameObject.activeSelf)
            {
                v.gameObject.SetActive(false);
                disabledCount++;
                Log($"Eski sahne volume'u devre disi birakildi: {v.gameObject.name}");
            }
        }

        if (disabledCount > 0)
            Log($"Toplam {disabledCount} eski volume devre disi birakildi. Sahne dosyasi degismedi.");
    }

    private string ResolveNiftiPath()
    {
        // 1. Mutlak yol override (en yuksek oncelik)
        if (!string.IsNullOrWhiteSpace(absolutePathOverride))
        {
            string abs = absolutePathOverride.Trim();
            if (File.Exists(abs))
            {
                Log($"Mutlak yol kullaniliyor: {abs}");
                return abs;
            }
            LogWarn($"absolutePathOverride dolu ama dosya yok: {abs}");
        }

        // 2. persistentDataPath (Quest 3'te adb push hedefi)
        string persistentRoot = Application.persistentDataPath;
        for (int i = 0; i < candidateFileNames.Length; i++)
        {
            string candidate = Path.Combine(persistentRoot, candidateFileNames[i]);
            if (File.Exists(candidate))
                return candidate;
        }

        // 3. StreamingAssets (APK icine gomulu, read-only)
        if (alsoSearchStreamingAssets)
        {
            string saRoot = Application.streamingAssetsPath;
            for (int i = 0; i < candidateFileNames.Length; i++)
            {
                string candidate = Path.Combine(saRoot, candidateFileNames[i]);
                // Android'de StreamingAssets jar:// URL'si olabilir. File.Exists
                // desktop/Editor icin yeterli; Android StreamingAssets icin
                // UnityWebRequest gerekir - bu pipeline'da siradan File.Exists
                // yeterli (persistentDataPath zaten kapsiyor).
                if (File.Exists(candidate))
                    return candidate;
            }
        }

        return null;
    }

    private void BuildRuntimePlaneRig(Transform slicingPlaneTransform, Transform crossSectionPlaneTransform)
    {
        Transform sphere = ResolveSphere();
        if (sphere == null)
        {
            LogWarn($"Sphere bulunamadi (isim='{sphereSearchName}'). Runtime plane rig kurulamadı.");
            return;
        }

        if (slicingPlaneTransform != null)
            slicingPlaneTransform.SetParent(null, true);
        if (crossSectionPlaneTransform != null)
            crossSectionPlaneTransform.SetParent(null, true);

        RuntimePlaneRigController rig = sphere.GetComponent<RuntimePlaneRigController>();
        if (rig == null)
            rig = sphere.gameObject.AddComponent<RuntimePlaneRigController>();
        rig.InitializeRuntime(sphere, slicingPlaneTransform, crossSectionPlaneTransform);
        Log($"RuntimePlaneRigController baglandi -> sphere='{sphere.name}'");

        AttachSphereGrab(sphere);
    }

    private Transform CreateRuntimeCrossSectionPlane(VolumeRenderedObject volObj, Transform slicingPlaneTransform)
    {
        if (volObj == null)
            return null;

        if (runtimeCrossSectionPlane != null)
            return runtimeCrossSectionPlane;

        // Clipping plane sadece opt-in'de olusturulur. Kapali oldugunda
        // SlicingPlaneFollowSphere zaten sphere'in kendisine fallback eder
        // (GetTarget: crossSectionPlane != null ? crossSectionPlane : sphere).
        if (!createRuntimeCrossSectionPlane)
        {
            Log("createRuntimeCrossSectionPlane=false; clipping plane olusturulmadi. Volume tam goruntulenir.");
            return null;
        }

        Transform sphere = ResolveSphere();
        if (sphere == null)
            return null;

        GameObject planeObject = GameObject.Instantiate(Resources.Load<GameObject>("CrossSectionPlane"));
        planeObject.name = "CrossSectionPlane_Runtime";
        planeObject.transform.SetParent(null, true);
        ApplyCrossSectionPlaneTemplateTransform(planeObject.transform);

        MeshRenderer planeRenderer = planeObject.GetComponent<MeshRenderer>();
        if (planeRenderer != null)
            planeRenderer.material = new Material(ShaderFactory.GetCrossSectionPlaneShader());

        CrossSectionPlane crossSectionPlane = planeObject.GetComponent<CrossSectionPlane>();
        crossSectionPlane.SetTargetObject(volObj);

        if (centerCrossSectionPlaneOnVolumeAtStart && volObj.volumeContainerObject != null)
        {
            planeObject.transform.position = volObj.volumeContainerObject.transform.position;
            Log($"CrossSectionPlane volume merkezine tasindi: worldPos={planeObject.transform.position}");
        }

        if (slicingPlaneTransform != null)
        {
            slicingPlaneTransform.rotation = SlicingPlaneXzWorldRotation;
            planeObject.transform.position = slicingPlaneTransform.position;
            planeObject.transform.rotation = CrossSectionPlaneXzWorldRotation;
            planeObject.transform.localScale = ToCrossSectionQuadScale(slicingPlaneTransform.lossyScale);
            Log($"CrossSectionPlane SlicingPlane ile hizalandi: worldPos={planeObject.transform.position}, worldScale={planeObject.transform.localScale}");
        }

        runtimeCrossSectionPlane = planeObject.transform;
        Log("Runtime clipping icin Sphere'e bagli CrossSectionPlane olusturuldu.");
        return runtimeCrossSectionPlane;
    }

    private void ApplyCrossSectionPlaneTemplateTransform(Transform targetPlane)
    {
        if (targetPlane == null)
            return;

        Transform templateSphere = FindSourceSphere();
        Transform templatePlane = FindTemplateCrossSectionPlane(templateSphere);
        if (useTemplateCrossSectionPlaneOffset && templatePlane != null)
        {
            targetPlane.localPosition = templatePlane.localPosition;
            Log($"Template CrossSectionPlane lokal pozisyonu kullanildi: localPos={targetPlane.localPosition}");
        }
        else
        {
            targetPlane.localPosition = crossSectionPlaneLocalOffsetFromSphere;
        }

        targetPlane.rotation = CrossSectionPlaneXzWorldRotation;
        targetPlane.localScale = Vector3.one * 0.1f;
    }

    private static Vector3 ToCrossSectionQuadScale(Vector3 slicingPlaneScale)
    {
        return new Vector3(slicingPlaneScale.x * 10f, slicingPlaneScale.z * 10f, slicingPlaneScale.y * 10f);
    }

    private Transform FindTemplateCrossSectionPlane(Transform templateSphere)
    {
        if (templateSphere == null)
            return null;

        for (int i = 0; i < templateSphere.childCount; i++)
        {
            Transform child = templateSphere.GetChild(i);
            if (child == null || string.IsNullOrEmpty(child.name))
                continue;

            if (child.name.Contains("CrossSectionPlane"))
                return child;
        }

        return null;
    }

    private void AttachSphereGrab(Transform sphere)
    {
        if (sphere == null) return;

        VolumeGrab grab = sphere.GetComponent<VolumeGrab>();
        if (grab == null)
            grab = sphere.gameObject.AddComponent<VolumeGrab>();

        // Sphere kavrama yaricapi volume'unkinden kucuk olmali; aksi halde
        // el volume'a yaklastiginda iki grab da claim alabilir.
        grab.ConfigureGrabRadius(sphereGrabRadius);

        Log($"Sphere'e VolumeGrab eklendi (radius={sphereGrabRadius}m).");
    }

    private Transform ResolveSphere()
    {
        // Daha onceki cagriden cache varsa direkt don
        if (runtimeSphere != null)
            return runtimeSphere;

        // Sablon (kaynak) sphere'i bul - bu sadece referans; kopya degil
        Transform template = FindSourceSphere();
        if (template == null)
            return null;

        if (!buildFreshSphere)
        {
            // Sifirdan insa yapma - orijinali dogrudan kullan
            runtimeSphere = template;
            return runtimeSphere;
        }

        // SIFIRDAN yeni GameObject insa et, eskisini sablon olarak kullan
        runtimeSphere = BuildRuntimeSphereFromTemplate(template);

        if (disableOriginalSphere && template.gameObject.activeSelf)
        {
            template.gameObject.SetActive(false);
            Log($"Orijinal sphere kapatildi: '{template.name}'");
        }

        return runtimeSphere;
    }

    private Transform BuildRuntimeSphereFromTemplate(Transform template)
    {
        GameObject newGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        newGo.name = template.name + "_Runtime";
        newGo.transform.SetParent(template.parent, false);
        newGo.transform.position = volumeWorldPosition + runtimeSphereOffsetFromVolume;
        newGo.transform.rotation = template.rotation;
        newGo.transform.localScale = template.localScale;
        newGo.layer = template.gameObject.layer;
        try { newGo.tag = template.gameObject.tag; } catch { /* tag yoksa gormezden gel */ }
        CopySphereVisuals(template.gameObject, newGo);
        CopySpherePhysics(template.gameObject, newGo);
        ConfigureRuntimeSphereVisibility(newGo);

        Log($"Runtime sphere insa edildi: '{newGo.name}' @ {newGo.transform.position}");
        return newGo.transform;
    }

    private void ConfigureRuntimeSphereVisibility(GameObject sphereGo)
    {
        if (!forceRuntimeSphereVisible || sphereGo == null)
            return;

        MeshRenderer renderer = sphereGo.GetComponent<MeshRenderer>();
        if (renderer == null)
            renderer = sphereGo.AddComponent<MeshRenderer>();

        MeshFilter filter = sphereGo.GetComponent<MeshFilter>();
        if (filter == null)
            filter = sphereGo.AddComponent<MeshFilter>();

        renderer.enabled = true;
        renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
        renderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
        renderer.allowOcclusionWhenDynamic = false;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Standard");

        Material mat = shader != null
            ? new Material(shader)
            : (renderer.sharedMaterial != null ? new Material(renderer.sharedMaterial) : null);
        if (mat == null)
        {
            LogWarn("Runtime sphere material olusturulamadi; mevcut material ile devam ediliyor.");
            return;
        }
        mat.name = "RuntimeSphereVisible_Mat";
        if (mat.HasProperty("_BaseColor"))
            mat.SetColor("_BaseColor", runtimeSphereColor);
        if (mat.HasProperty("_Color"))
            mat.SetColor("_Color", runtimeSphereColor);
        mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Overlay;
        renderer.sharedMaterial = mat;

        Log($"Runtime sphere gorunur yapildi: '{sphereGo.name}', shader='{mat.shader.name}', color={runtimeSphereColor}");
    }

    private void CopySphereVisuals(GameObject template, GameObject destination)
    {
        MeshFilter srcFilter = template.GetComponent<MeshFilter>();
        MeshRenderer srcRenderer = template.GetComponent<MeshRenderer>();
        MeshFilter dstFilter = destination.GetComponent<MeshFilter>();
        MeshRenderer dstRenderer = destination.GetComponent<MeshRenderer>();

        if (srcFilter != null && dstFilter != null && srcFilter.sharedMesh != null)
            dstFilter.sharedMesh = srcFilter.sharedMesh;

        if (srcRenderer != null && dstRenderer != null)
        {
            dstRenderer.sharedMaterials = srcRenderer.sharedMaterials;
            dstRenderer.shadowCastingMode = srcRenderer.shadowCastingMode;
            dstRenderer.receiveShadows = srcRenderer.receiveShadows;
            dstRenderer.lightProbeUsage = srcRenderer.lightProbeUsage;
            dstRenderer.reflectionProbeUsage = srcRenderer.reflectionProbeUsage;
            dstRenderer.renderingLayerMask = srcRenderer.renderingLayerMask;
            dstRenderer.enabled = srcRenderer.enabled;
        }
    }

    private void CopySpherePhysics(GameObject template, GameObject destination)
    {
        SphereCollider srcSphereCollider = template.GetComponent<SphereCollider>();
        SphereCollider dstSphereCollider = destination.GetComponent<SphereCollider>();
        if (dstSphereCollider == null)
            dstSphereCollider = destination.AddComponent<SphereCollider>();

        if (srcSphereCollider != null)
        {
            dstSphereCollider.center = srcSphereCollider.center;
            dstSphereCollider.radius = srcSphereCollider.radius;
            dstSphereCollider.isTrigger = srcSphereCollider.isTrigger;
            dstSphereCollider.enabled = srcSphereCollider.enabled;
        }

        Rigidbody srcBody = template.GetComponent<Rigidbody>();
        if (srcBody != null)
        {
            Rigidbody dstBody = destination.GetComponent<Rigidbody>();
            if (dstBody == null)
                dstBody = destination.AddComponent<Rigidbody>();

            dstBody.mass = srcBody.mass;
            dstBody.linearDamping = srcBody.linearDamping;
            dstBody.angularDamping = srcBody.angularDamping;
            dstBody.useGravity = srcBody.useGravity;
            dstBody.isKinematic = srcBody.isKinematic;
            dstBody.interpolation = srcBody.interpolation;
            dstBody.collisionDetectionMode = srcBody.collisionDetectionMode;
            dstBody.constraints = srcBody.constraints;
        }
    }

    private Transform FindSourceSphere()
    {
        // Oncelik 1: inspector'dan baglanmis referans (aktif olmasa da kullanilabilir)
        if (sphereReference != null)
            return sphereReference;

        // Oncelik 2: isimle ara. GameObject.Find inactive GO'lari bulmaz, bu yuzden
        // FindObjectsByType + Include inactive kullanalim ki eski kapatilmis sphere'i da bulabilelim.
        if (string.IsNullOrWhiteSpace(sphereSearchName))
            return null;

        // Once aktif GO'larda dogrudan Find (hizli)
        GameObject go = GameObject.Find(sphereSearchName);
        if (go != null)
            return go.transform;

        // Aktif bulunamazsa inactive dahil tum transform'lari tara
        Transform[] allTransforms = FindObjectsByType<Transform>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None);
        for (int i = 0; i < allTransforms.Length; i++)
        {
            if (allTransforms[i] != null && allTransforms[i].name == sphereSearchName)
                return allTransforms[i];
        }

        return null;
    }

    private void AttachVolumeScripts(VolumeRenderedObject volObj)
    {
        GameObject host = volObj.gameObject;

        if (host.GetComponent<CTLungTransferFunction>() == null)
        {
            CTLungTransferFunction tf = host.AddComponent<CTLungTransferFunction>();
            tf.applyOnStart = false;
            tf.verbose = verbose;
        }

        if (host.GetComponent<CTCardiac3TransferFunction>() == null)
        {
            CTCardiac3TransferFunction tf = host.AddComponent<CTCardiac3TransferFunction>();
            tf.applyOnStart = false;
            tf.verbose = verbose;
        }

        if (host.GetComponent<CTChestContrastEnhancedTransferFunction>() == null)
        {
            CTChestContrastEnhancedTransferFunction tf = host.AddComponent<CTChestContrastEnhancedTransferFunction>();
            tf.applyOnStart = false;
            tf.verbose = verbose;
        }

        if (host.GetComponent<Quest3ShaderOptimizer>() == null)
        {
            host.AddComponent<Quest3ShaderOptimizer>();
        }

        if (host.GetComponent<TransferFunctionToggle>() == null)
        {
            TransferFunctionToggle toggle = host.AddComponent<TransferFunctionToggle>();
            toggle.startWithCTLung = false;
            toggle.verboseInput = false;
        }

        VolumeGrab hostGrab = host.GetComponent<VolumeGrab>();
        if (hostGrab == null)
            hostGrab = host.AddComponent<VolumeGrab>();

        hostGrab.ConfigureGrabRadius(volumeGrabRadius);
        hostGrab.enabled = true;

        Log($"Volume GO'ya CT-Lung / CT-Cardiac3 / CT-Chest-CE / Quest3Optimizer / TF-Toggle eklendi. VolumeGrab aktif (radius={volumeGrabRadius}m).");
    }

    private void ReportStatus(string message)
    {
        if (RuntimeVolumeLoadingHUD.Instance != null)
            RuntimeVolumeLoadingHUD.Instance.SetMessage(message);
    }

    private void ReportError(string message)
    {
        if (RuntimeVolumeLoadingHUD.Instance != null)
            RuntimeVolumeLoadingHUD.Instance.SetError(message);
    }

    private void Log(string msg)
    {
        if (verbose)
            Debug.Log("[RuntimeVolumeBootstrap] " + msg);
    }

    private void LogWarn(string msg)
    {
        Debug.LogWarning("[RuntimeVolumeBootstrap] " + msg);
    }

    // =========================================================================
    // Progress Handler (HUD'a progress yansitir)
    // =========================================================================

    private class BootstrapProgressHandler : IProgressHandler
    {
        private readonly RuntimeVolumeBootstrap owner;
        private readonly string datasetName;

        public BootstrapProgressHandler(RuntimeVolumeBootstrap owner, string datasetName)
        {
            this.owner = owner;
            this.datasetName = datasetName;
        }

        public void StartStage(float weight, string description = "") { }
        public void EndStage() { }

        public void ReportProgress(float progress, string description = "")
        {
            int pct = Mathf.RoundToInt(Mathf.Clamp01(progress) * 100f);
            string label = string.IsNullOrEmpty(description) ? "Volume olusturuluyor" : description;
            owner.ReportStatus($"{label}... %{pct}\n{datasetName}");
        }

        public void ReportProgress(int currentStep, int totalSteps, string description = "")
        {
            float p = totalSteps > 0 ? (float)currentStep / totalSteps : 0f;
            ReportProgress(p, description);
        }

        public void Fail()
        {
            owner.ReportError("Ilerleme basarisiz.");
        }
    }
}

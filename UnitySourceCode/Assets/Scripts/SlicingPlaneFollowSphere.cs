using UnityEngine;

public class SlicingPlaneFollowSphere : MonoBehaviour
{
    private static readonly Quaternion CrossSectionPlaneRotationOffset = Quaternion.Euler(270f, 0f, 0f);

    [Header("Refs")]
    [Tooltip("Varsa asıl surucu transform. SlicingPlane bunun pozisyon/rotasyonunu izler.")]
    [SerializeField] private Transform crossSectionPlane;

    [Tooltip("Hareketini takip edeceginiz obje (Sphere).")]
    [SerializeField] private Transform sphere;

    [Tooltip("SlicingPlane'in parent'i (Volume Container). Bos birakilirsa transform.parent kullanilir.")]
    [SerializeField] private Transform volumeContainer;

    [Header("Follow")]
    [Tooltip("Sphere'in world pozisyonunu, Volume Container lokal uzayina cevirip SlicingPlane.localPosition olarak yazar.")]
    [SerializeField] private bool followPosition = true;

    [Tooltip("Sphere'in rotasyonunu da takip etsin mi? (Opsiyonel)")]
    [SerializeField] private bool followRotation = true;

    [Tooltip("Acikken SlicingPlane, sphere rotasyonunu takip eder ve XZ tabanli yuzeyini korur.")]
    [SerializeField] private bool rotateWithSphere = true;

    [Tooltip("CrossSectionPlane yokken sphere plane merkezinde degil, plane kosesinde dursun.")]
    [SerializeField] private bool placeSphereAtPlaneCorner = true;

    [Header("Initial Rotation")]
    [Tooltip("Baslangicta SlicingPlane.localEulerAngles degerini bu degere ayarlar.")]
    [SerializeField] private bool setInitialLocalEuler = true;

    [SerializeField] private Vector3 initialLocalEuler = new Vector3(90f, 0f, 0f);

    [Tooltip("Baslangictaki SlicingPlane-Sphere farkini offset olarak korur.")]
    [SerializeField] private bool keepInitialOffset = true;

    [Tooltip("Runtime initialize sirasinda plane'i dogrudan sphere'in merkezine oturtur.")]
    [SerializeField] private bool snapToSphereOnInitialize = true;

    private Vector3 localPosOffset;
    private Quaternion localRotOffset = Quaternion.identity;

    /// <summary>
    /// Runtime bootstrap'ten referanslari koddan set etmek icin (inspector yerine).
    /// Hem sphere hem volumeContainer'i set eder. Awake/Start cagrilmadan once
    /// veya hemen sonra cagirilmasi guvenli.
    /// </summary>
    public void InitializeRuntime(Transform sphereTarget, Transform volumeContainerTarget, Transform crossSectionPlaneTarget = null)
    {
        this.sphere = sphereTarget;
        this.volumeContainer = volumeContainerTarget;
        this.crossSectionPlane = crossSectionPlaneTarget;

        if (crossSectionPlane != null)
        {
            // Gercek clipping plane varsa gorsel slicing plane onunla birebir cakissin.
            setInitialLocalEuler = false;
            keepInitialOffset = false;
            snapToSphereOnInitialize = false;

            transform.position = crossSectionPlane.position;
            ApplyRotationFromTarget();
            MatchScaleToCrossSectionPlane();
            CacheOffsets();
            return;
        }

        // Offset'leri yeni referanslara gore yeniden hesapla.
        ApplyInitialRotationIfNeeded();
        SnapToSphereIfNeeded();
        CacheOffsets();
    }

    private void Awake()
    {
        if (volumeContainer == null)
            volumeContainer = transform.parent;
    }

    private void Start()
    {
        ApplyInitialRotationIfNeeded();
        CacheOffsets();
    }

    private void OnValidate()
    {
        if (volumeContainer == null)
            volumeContainer = transform.parent;
    }

    private void CacheOffsets()
    {
        if (rotateWithSphere || placeSphereAtPlaneCorner || !keepInitialOffset)
        {
            localPosOffset = Vector3.zero;
            localRotOffset = Quaternion.identity;
            return;
        }

        Transform target = GetTarget();
        if (target == null || volumeContainer == null)
            return;

        Vector3 targetLocalPos = volumeContainer.InverseTransformPoint(target.position);
        localPosOffset = transform.localPosition - targetLocalPos;

        Quaternion targetLocalRot = Quaternion.Inverse(volumeContainer.rotation) * target.rotation;
        localRotOffset = Quaternion.Inverse(targetLocalRot) * transform.localRotation;
    }

    private void LateUpdate()
    {
        Transform target = GetTarget();
        if (target == null || volumeContainer == null)
            return;

        // SlicingPlane'in Volume Container alt-objesi olma ozelligini bozmadan,
        // target'in world transformunu container local uzayina map'leyip uygula.
        if (followPosition)
        {
            if (crossSectionPlane != null)
            {
                Vector3 targetLocalPos = volumeContainer.InverseTransformPoint(target.position);
                transform.localPosition = targetLocalPos + localPosOffset;
            }
            else if (placeSphereAtPlaneCorner)
            {
                Vector3 planeCenterWorld = target.position + (target.rotation * GetSlicingPlaneCornerToCenterOffset());
                transform.localPosition = volumeContainer.InverseTransformPoint(planeCenterWorld);
            }
            else
            {
                Vector3 targetLocalPos = volumeContainer.InverseTransformPoint(target.position);
                transform.localPosition = targetLocalPos + localPosOffset;
            }
        }

        if (followRotation)
        {
            if (rotateWithSphere)
            {
                ApplyRotationFromTarget();
            }
            else
            {
                Quaternion targetLocalRot = Quaternion.Inverse(volumeContainer.rotation) * target.rotation;
                transform.localRotation = targetLocalRot * localRotOffset;
            }
        }

        if (crossSectionPlane != null)
            MatchScaleToCrossSectionPlane();
    }

    private void ApplyInitialRotationIfNeeded()
    {
        if (rotateWithSphere)
        {
            ApplyRotationFromTarget();
            return;
        }

        if (!setInitialLocalEuler)
            return;

        transform.localEulerAngles = initialLocalEuler;
    }

    private void SnapToSphereIfNeeded()
    {
        Transform target = GetTarget();
        if (!snapToSphereOnInitialize || target == null)
            return;

        if (placeSphereAtPlaneCorner && crossSectionPlane == null)
            transform.position = target.position + (target.rotation * GetSlicingPlaneCornerToCenterOffset());
        else
            transform.position = target.position;
    }

    private Transform GetTarget()
    {
        if (crossSectionPlane != null)
            return crossSectionPlane;
        return sphere;
    }

    private void MatchScaleToCrossSectionPlane()
    {
        if (crossSectionPlane == null || volumeContainer == null)
            return;

        Vector3 parentScale = volumeContainer.lossyScale;
        Vector3 targetScale = ToSlicingPlaneScale(crossSectionPlane.lossyScale);

        transform.localScale = new Vector3(
            SafeDivide(targetScale.x, parentScale.x),
            SafeDivide(targetScale.y, parentScale.y),
            SafeDivide(targetScale.z, parentScale.z));
    }

    private static float SafeDivide(float value, float divisor)
    {
        return Mathf.Abs(divisor) < 0.00001f ? value : value / divisor;
    }

    private void ApplyRotationFromTarget()
    {
        Transform target = GetTarget();
        Quaternion worldRotation = Quaternion.identity;

        if (target == crossSectionPlane && crossSectionPlane != null)
            worldRotation = crossSectionPlane.rotation * Quaternion.Inverse(CrossSectionPlaneRotationOffset);
        else if (target != null)
            worldRotation = target.rotation;

        if (volumeContainer != null)
            transform.localRotation = Quaternion.Inverse(volumeContainer.rotation) * worldRotation;
        else
            transform.rotation = worldRotation;
    }

    private Vector3 GetSlicingPlaneCornerToCenterOffset()
    {
        Vector3 scale = transform.lossyScale;
        return new Vector3(scale.x * 5f, 0f, scale.z * 5f);
    }

    private static Vector3 ToSlicingPlaneScale(Vector3 crossSectionScale)
    {
        return new Vector3(crossSectionScale.x / 10f, crossSectionScale.z / 10f, crossSectionScale.y / 10f);
    }

}

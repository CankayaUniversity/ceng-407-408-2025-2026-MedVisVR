using UnityEngine;

/// <summary>
/// Sphere'i tek handle olarak kullanip SlicingPlane ve CrossSectionPlane'i
/// ayni transform rig'inde world-space olarak surer.
/// </summary>
public class RuntimePlaneRigController : MonoBehaviour
{
    private static readonly Quaternion CrossSectionPlaneRotationOffset = Quaternion.Euler(270f, 0f, 0f);

    [SerializeField] private Transform sphere;
    [SerializeField] private Transform slicingPlane;
    [SerializeField] private Transform crossSectionPlane;
    [SerializeField] private Vector3 sphereToPlaneCenterOffsetLocal = Vector3.zero;
    [SerializeField] private Vector3 planeWorldScale = new Vector3(0.1f, 0.1f, 0.1f);
    [SerializeField] private Vector3 crossSectionWorldScale = Vector3.one;

    public void InitializeRuntime(Transform sphereTarget, Transform slicingPlaneTarget, Transform crossSectionPlaneTarget)
    {
        sphere = sphereTarget;
        slicingPlane = slicingPlaneTarget;
        crossSectionPlane = crossSectionPlaneTarget;

        if (slicingPlane != null)
        {
            planeWorldScale = slicingPlane.lossyScale;
            crossSectionWorldScale = ToCrossSectionQuadScale(planeWorldScale);
            sphereToPlaneCenterOffsetLocal = GetSlicingPlaneCornerToCenterOffset(planeWorldScale);
            sphere.rotation = slicingPlane.rotation;
            sphere.position = slicingPlane.position - (sphere.rotation * sphereToPlaneCenterOffsetLocal);
        }
        SnapNow();
    }

    private void LateUpdate()
    {
        SnapNow();
    }

    private void SnapNow()
    {
        if (sphere == null)
            return;

        Vector3 planeCenterWorld = sphere.position + (sphere.rotation * sphereToPlaneCenterOffsetLocal);

        if (slicingPlane != null)
        {
            slicingPlane.position = planeCenterWorld;
            slicingPlane.rotation = sphere.rotation;
            slicingPlane.localScale = planeWorldScale;
        }

        if (crossSectionPlane != null)
        {
            crossSectionPlane.position = planeCenterWorld;
            crossSectionPlane.rotation = sphere.rotation * CrossSectionPlaneRotationOffset;
            crossSectionPlane.localScale = crossSectionWorldScale;
        }
    }

    private static Vector3 GetSlicingPlaneCornerToCenterOffset(Vector3 slicingPlaneScale)
    {
        // Unity's built-in Plane mesh is 10x10 on local X/Z, so scale * 5 is half-size.
        return new Vector3(slicingPlaneScale.x * 5f, 0f, slicingPlaneScale.z * 5f);
    }

    private static Vector3 ToCrossSectionQuadScale(Vector3 slicingPlaneScale)
    {
        // CrossSectionPlane uses a 1x1 Quad on local X/Y, while SlicingPlane uses a 10x10 Plane on X/Z.
        return new Vector3(slicingPlaneScale.x * 10f, slicingPlaneScale.z * 10f, slicingPlaneScale.y * 10f);
    }
}

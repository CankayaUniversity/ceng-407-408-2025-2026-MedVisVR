using UnityEngine;

/// <summary>
/// Gercek clipping plane'i sphere'in child'i yapmadan takip ettirir.
/// Boylece sphere'in olcegi plane matrisini bozmaz.
/// </summary>
public class CrossSectionPlaneFollowSphere : MonoBehaviour
{
    private static readonly Quaternion CrossSectionPlaneRotationOffset = Quaternion.Euler(270f, 0f, 0f);

    [SerializeField] private Transform sphere;
    [SerializeField] private Vector3 localPositionOffsetFromSphere = Vector3.zero;
    [SerializeField] private Vector3 localEulerOffsetFromSphere = new Vector3(270f, 0f, 0f);
    [SerializeField] private bool followRotation = true;
    [SerializeField] private bool rotateWithSphere = true;
    [SerializeField] private bool placeSphereAtPlaneCorner = true;

    public void InitializeRuntime(Transform sphereTarget, Vector3 localPositionOffset, Quaternion localRotationOffset)
    {
        sphere = sphereTarget;
        localPositionOffsetFromSphere = localPositionOffset;
        localEulerOffsetFromSphere = localRotationOffset.eulerAngles;
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

        Vector3 offset = placeSphereAtPlaneCorner ? GetCornerToCenterOffset() : localPositionOffsetFromSphere;
        transform.position = sphere.position + (sphere.rotation * offset);
        if (followRotation)
        {
            transform.rotation = rotateWithSphere
                ? sphere.rotation * CrossSectionPlaneRotationOffset
                : sphere.rotation * Quaternion.Euler(localEulerOffsetFromSphere);
        }
    }

    private Vector3 GetCornerToCenterOffset()
    {
        Vector3 scale = transform.lossyScale;
        return new Vector3(scale.x * 0.5f, 0f, scale.y * 0.5f);
    }
}

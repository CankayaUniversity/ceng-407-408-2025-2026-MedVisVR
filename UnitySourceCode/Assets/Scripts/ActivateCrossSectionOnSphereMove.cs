using UnityEngine;
using UnityVolumeRendering;

/// <summary>
/// CrossSectionPlane'i baslangicta pasif tutar.
/// Sphere ilk kez anlamli sekilde hareket ettiginde clipping aktif olur.
/// Boylece acilista volume tamamen clip edilip gorunmez hale gelmez.
/// </summary>
public class ActivateCrossSectionOnSphereMove : MonoBehaviour
{
    [SerializeField] private Transform sphere;
    [SerializeField] private float activationDistance = 0.01f;

    private CrossSectionPlane crossSectionPlane;
    private Vector3 initialSpherePosition;
    private bool activated;

    public void InitializeRuntime(Transform sphereTarget, float movementThreshold = 0.01f)
    {
        sphere = sphereTarget;
        activationDistance = movementThreshold;
        crossSectionPlane = GetComponent<CrossSectionPlane>();
        initialSpherePosition = sphere != null ? sphere.position : Vector3.zero;

        if (crossSectionPlane != null)
            crossSectionPlane.enabled = false;
    }

    private void Awake()
    {
        if (crossSectionPlane == null)
            crossSectionPlane = GetComponent<CrossSectionPlane>();
    }

    private void Update()
    {
        if (activated || sphere == null || crossSectionPlane == null)
            return;

        if (Vector3.Distance(initialSpherePosition, sphere.position) < activationDistance)
            return;

        crossSectionPlane.enabled = true;
        activated = true;
    }
}

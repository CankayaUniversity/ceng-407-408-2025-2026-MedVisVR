using UnityEngine;

namespace UnityVolumeRendering
{
    [ExecuteInEditMode]
    public class SlicingPlane : MonoBehaviour
    {
        public VolumeRenderedObject targetObject;
        private MeshRenderer meshRenderer;

        private void Start()
        {
            meshRenderer = GetComponent<MeshRenderer>();
        }

        private void Update()
        {
            Matrix4x4 parentInverseMat = transform.parent != null
                ? transform.parent.worldToLocalMatrix
                : Matrix4x4.identity;

            if (targetObject != null && targetObject.volumeContainerObject != null)
                parentInverseMat = targetObject.volumeContainerObject.transform.worldToLocalMatrix;

            meshRenderer.sharedMaterial.SetMatrix("_parentInverseMat", parentInverseMat);
            meshRenderer.sharedMaterial.SetMatrix("_planeMat", transform.localToWorldMatrix); // TODO: allow changing scale
        }
    }
}

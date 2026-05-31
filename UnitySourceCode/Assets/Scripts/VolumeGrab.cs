using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.XR;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Meta XR icin tek elli VR grab component'i. Hem controller (Touch) hem
/// hand tracking (pinch) kavramayi destekler. Her frame iki taraf icin
/// (sol/sag) once OVRHand IsTracked ise hand pose + IsPinching(Index)
/// kullanilir; yoksa LeftHandAnchor/RightHandAnchor + grip/trigger fallback'i.
///
/// Kavrama akisi:
///   1. Bir tarafta press-DOWN (yeni basildi) ve volume yeterince yakinsa kavrama baslar.
///   2. Kavrama suresince pose-local offset korunur, her frame world transform yeniden uygulanir.
///   3. Press birakildiginda volume world space'te kalir.
///
/// SlicingPlane VolumeContainer altinda parent-child oldugu icin volume ile beraber tasinir.
/// </summary>
public class VolumeGrab : MonoBehaviour
{
    public enum GrabButton { Grip, Trigger }

    [Header("Controller Anchor Names")]
    [Tooltip("Sahnede aranacak Meta XR sol controller anchor GO'su.")]
    [SerializeField] private string leftHandAnchorName = "LeftHandAnchor";

    [Tooltip("Sahnede aranacak Meta XR sag controller anchor GO'su.")]
    [SerializeField] private string rightHandAnchorName = "RightHandAnchor";

    [Header("Grab")]
    [Tooltip("Kavrayan elin volume merkezine bu mesafe icinde olmasi gerekir (metre).")]
    [SerializeField, Range(0.05f, 2f)] private float maxGrabDistance = 0.45f;

    [Tooltip("0 = anlik takip. 0.1-0.3 = yumusak takip (jitter azaltir, gecikme ekler).")]
    [SerializeField, Range(0f, 0.5f)] private float followSmoothing = 0f;

    [Tooltip("Controller modunda hangi buton kavramayi tetikler. Hand tracking modunda " +
             "her zaman index parmagi pinch kullanilir.")]
    [SerializeField] private GrabButton grabButton = GrabButton.Grip;

    [Header("Hand Tracking")]
    [Tooltip("OVRHand bulundugunda controller anchor'i yerine hand pose'unu kullan.")]
    [SerializeField] private bool preferHandTrackingWhenAvailable = true;

    [Tooltip("Pinch icin minimum strength esigi (0-1). 0.7 deger UX standardi.")]
    [SerializeField, Range(0.3f, 1f)] private float pinchThreshold = 0.7f;

    [Header("Debug")]
    [SerializeField] private bool verbose = false;

    // Sahnedeki tum aktif VolumeGrab'larin kaydi - press-DOWN aninda en yakin
    // adayin claim almasi icin (ornegin volume + sphere ayni anda yakinsa,
    // sphere daha yakinsa sphere kavranir, volume kavranmaz).
    private static readonly List<VolumeGrab> instances = new List<VolumeGrab>();

    // Anchor referanslari (controller fallback)
    private Transform leftAnchor;
    private Transform rightAnchor;

    // OVRHand referanslari (hand tracking)
    private OVRHand leftHand;
    private OVRHand rightHand;
    private bool ovrHandsResolved;

    // Aktif kavrama durumu
    private Transform activePose;          // Su an kavrayan elin pose transform'u
    private Vector3 grabLocalPos;
    private Quaternion grabLocalRot;

    // Edge-detection
    private bool leftPressedPrev;
    private bool rightPressedPrev;

#if ENABLE_INPUT_SYSTEM
    private InputAction leftGripAction;
    private InputAction rightGripAction;
    private InputAction leftTriggerAction;
    private InputAction rightTriggerAction;
#endif

    /// <summary>
    /// Runtime'da olcek/parametre ayarlamak icin public setter (Bootstrap kullanir).
    /// </summary>
    public void ConfigureGrabRadius(float meters)
    {
        maxGrabDistance = Mathf.Clamp(meters, 0.02f, 5f);
    }

    private void OnEnable()
    {
        if (!instances.Contains(this)) instances.Add(this);

#if ENABLE_INPUT_SYSTEM
        leftGripAction = MakeAction("VG_LeftGrip", "<XRController>{LeftHand}/gripButton",
                                    "<OculusTouchController>{LeftHand}/gripButton",
                                    "<XRController>{LeftHand}/grip");
        rightGripAction = MakeAction("VG_RightGrip", "<XRController>{RightHand}/gripButton",
                                     "<OculusTouchController>{RightHand}/gripButton",
                                     "<XRController>{RightHand}/grip");
        leftTriggerAction = MakeAction("VG_LeftTrigger", "<XRController>{LeftHand}/triggerButton",
                                       "<OculusTouchController>{LeftHand}/triggerButton",
                                       "<XRController>{LeftHand}/trigger");
        rightTriggerAction = MakeAction("VG_RightTrigger", "<XRController>{RightHand}/triggerButton",
                                        "<OculusTouchController>{RightHand}/triggerButton",
                                        "<XRController>{RightHand}/trigger");
#endif
        if (verbose) Debug.Log("[VolumeGrab] Enabled.");
    }

    private void OnDisable()
    {
        instances.Remove(this);

#if ENABLE_INPUT_SYSTEM
        DisposeAction(ref leftGripAction);
        DisposeAction(ref rightGripAction);
        DisposeAction(ref leftTriggerAction);
        DisposeAction(ref rightTriggerAction);
#endif
        activePose = null;
    }

#if ENABLE_INPUT_SYSTEM
    private static InputAction MakeAction(string name, params string[] bindings)
    {
        InputAction a = new InputAction(name, InputActionType.Button);
        for (int i = 0; i < bindings.Length; i++) a.AddBinding(bindings[i]);
        a.Enable();
        return a;
    }

    private static void DisposeAction(ref InputAction a)
    {
        if (a == null) return;
        a.Disable();
        a.Dispose();
        a = null;
    }
#endif

    private void Update()
    {
        EnsureAnchors();
        EnsureOVRHands();

        // Her taraf icin pose ve press durumunu coz
        ResolveSide(isLeft: true, out Transform leftPose, out bool leftPressed);
        ResolveSide(isLeft: false, out Transform rightPose, out bool rightPressed);

        UpdateGrabState(leftPose, leftPressed, rightPose, rightPressed);

        leftPressedPrev = leftPressed;
        rightPressedPrev = rightPressed;
    }

    // =========================================================================
    // RESOLUTION (her frame: hangi pose, basili mi)
    // =========================================================================

    private void ResolveSide(bool isLeft, out Transform pose, out bool pressed)
    {
        pose = null;
        pressed = false;

        // 1) Hand tracking aktif mi?
        if (preferHandTrackingWhenAvailable)
        {
            OVRHand hand = isLeft ? leftHand : rightHand;
            if (hand != null && hand.IsTracked)
            {
                pose = hand.transform;
                // Pinch strength + threshold (IsPinching cogu zaman tutarli ama
                // bazi versiyonlarda strength daha kararli sinyaldir).
                float strength = hand.GetFingerPinchStrength(OVRHand.HandFinger.Index);
                bool isPinching = hand.GetFingerIsPinching(OVRHand.HandFinger.Index);
                pressed = isPinching && strength >= pinchThreshold;
                return;
            }
        }

        // 2) Controller anchor + button
        pose = isLeft ? leftAnchor : rightAnchor;
        if (pose == null)
            return;

        pressed = ReadControllerButton(isLeft);
    }

    private bool ReadControllerButton(bool isLeft)
    {
#if ENABLE_INPUT_SYSTEM
        InputAction action = grabButton == GrabButton.Grip
            ? (isLeft ? leftGripAction : rightGripAction)
            : (isLeft ? leftTriggerAction : rightTriggerAction);
        if (action != null && action.IsPressed())
            return true;
#endif

        // XR poll fallback
        UnityEngine.XR.InputDevice dev = InputDevices.GetDeviceAtXRNode(isLeft ? XRNode.LeftHand : XRNode.RightHand);
        if (!dev.isValid)
            return false;

        if (grabButton == GrabButton.Grip)
        {
            if (dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.gripButton, out bool pressed)) return pressed;
            if (dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.grip, out float amt)) return amt > 0.5f;
        }
        else
        {
            if (dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.triggerButton, out bool pressed)) return pressed;
            if (dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.trigger, out float amt)) return amt > 0.5f;
        }
        return false;
    }

    // =========================================================================
    // GRAB STATE MACHINE
    // =========================================================================

    private void UpdateGrabState(Transform leftPose, bool leftPressed, Transform rightPose, bool rightPressed)
    {
        // Aktif kavrama yok: yeni press-DOWN bekle
        if (activePose == null)
        {
            bool rightDown = rightPressed && !rightPressedPrev;
            bool leftDown = leftPressed && !leftPressedPrev;

            // En-yakin-aday koordinasyonu: ayni press-DOWN'da iki VolumeGrab
            // (volume + sphere) ayni anda yakinsa, sadece elle daha yakin olan
            // claim alir. Diger instance bu frame'de pas gecer.
            if (rightDown && CanGrab(rightPose) && IsClosestCandidate(rightPose))
                BeginGrab(rightPose, "right");
            else if (leftDown && CanGrab(leftPose) && IsClosestCandidate(leftPose))
                BeginGrab(leftPose, "left");
            return;
        }

        // Aktif pose hala basili mi? Aktif tarafi pose referansi ile esle.
        bool activeStillPressed =
            (activePose == rightPose && rightPressed) ||
            (activePose == leftPose && leftPressed);

        if (!activeStillPressed)
        {
            EndGrab();
            return;
        }

        Vector3 worldPos = activePose.TransformPoint(grabLocalPos);
        Quaternion worldRot = activePose.rotation * grabLocalRot;

        if (followSmoothing > 0f)
        {
            float t = 1f - Mathf.Clamp01(followSmoothing);
            transform.position = Vector3.Lerp(transform.position, worldPos, t);
            transform.rotation = Quaternion.Slerp(transform.rotation, worldRot, t);
        }
        else
        {
            transform.position = worldPos;
            transform.rotation = worldRot;
        }
    }

    private bool CanGrab(Transform pose)
    {
        if (pose == null) return false;
        return Vector3.Distance(pose.position, transform.position) <= maxGrabDistance;
    }

    /// <summary>
    /// Sahnedeki diger VolumeGrab instance'lari arasinda bu transform'un
    /// pose'a en yakin (kavrayabilir) aday olup olmadigini dondurur.
    /// Ayni press-DOWN frame'inde birden fazla instance yakinsa, sadece
    /// en yakini kavrayar - boylece volume ve sphere ayni anda kavranmaz.
    /// </summary>
    private bool IsClosestCandidate(Transform pose)
    {
        if (pose == null) return false;
        float myDist = Vector3.Distance(pose.position, transform.position);

        for (int i = 0; i < instances.Count; i++)
        {
            VolumeGrab other = instances[i];
            if (other == null || other == this) continue;
            if (!other.isActiveAndEnabled) continue;
            // Zaten baska bir pose'a kavramis instance bu yarismaya katilmaz
            if (other.activePose != null) continue;
            // Diger instance kendi maxGrabDistance'i icinde mi?
            float otherDist = Vector3.Distance(pose.position, other.transform.position);
            if (otherDist > other.maxGrabDistance) continue;
            // Daha yakinsa biz aday degiliz
            if (otherDist < myDist) return false;
        }
        return true;
    }

    private void BeginGrab(Transform pose, string label)
    {
        activePose = pose;
        grabLocalPos = pose.InverseTransformPoint(transform.position);
        grabLocalRot = Quaternion.Inverse(pose.rotation) * transform.rotation;
        if (verbose) Debug.Log($"[VolumeGrab] Kavrandi: {label} ({pose.name})");
    }

    private void EndGrab()
    {
        if (verbose) Debug.Log($"[VolumeGrab] Birakildi.");
        activePose = null;
    }

    // =========================================================================
    // ANCHOR / OVRHAND DISCOVERY
    // =========================================================================

    private void EnsureAnchors()
    {
        if (leftAnchor == null) leftAnchor = FindByName(leftHandAnchorName);
        if (rightAnchor == null) rightAnchor = FindByName(rightHandAnchorName);
    }

    private void EnsureOVRHands()
    {
        // Bir kere bulduktan sonra cache'le; null kalirsa yeniden tara
        // (hand tracking sahneye gec ekleniyor olabilir).
        if (ovrHandsResolved && leftHand != null && rightHand != null)
            return;

        OVRHand[] hands = FindObjectsByType<OVRHand>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        if (hands == null || hands.Length == 0)
            return;

        // OVRHand.HandType internal; reflection ile oku.
        FieldInfo handTypeField = typeof(OVRHand).GetField("HandType",
            BindingFlags.NonPublic | BindingFlags.Instance);

        for (int i = 0; i < hands.Length; i++)
        {
            OVRHand h = hands[i];
            if (h == null) continue;

            string side = ResolveHandSide(h, handTypeField);
            if (side == "Left" && leftHand == null)
                leftHand = h;
            else if (side == "Right" && rightHand == null)
                rightHand = h;
        }

        ovrHandsResolved = true;

        if (verbose)
            Debug.Log($"[VolumeGrab] OVRHand resolve: left={(leftHand != null ? leftHand.name : "null")} " +
                      $"right={(rightHand != null ? rightHand.name : "null")}");
    }

    private static string ResolveHandSide(OVRHand h, FieldInfo handTypeField)
    {
        // 1) Reflection ile internal HandType field'i (en kesin)
        if (handTypeField != null)
        {
            try
            {
                object val = handTypeField.GetValue(h);
                if (val != null)
                {
                    string s = val.ToString();
                    if (s == "HandLeft") return "Left";
                    if (s == "HandRight") return "Right";
                }
            }
            catch { /* sessizce isim fallback'ine dus */ }
        }

        // 2) Isim tabanli fallback
        string n = h.gameObject.name.ToLowerInvariant();
        if (n.Contains("left")) return "Left";
        if (n.Contains("right")) return "Right";

        // 3) Parent isim fallback
        Transform p = h.transform.parent;
        while (p != null)
        {
            string pn = p.name.ToLowerInvariant();
            if (pn.Contains("left")) return "Left";
            if (pn.Contains("right")) return "Right";
            p = p.parent;
        }

        return null;
    }

    private static Transform FindByName(string name)
    {
        if (string.IsNullOrEmpty(name)) return null;
        GameObject go = GameObject.Find(name);
        if (go != null) return go.transform;

        Transform[] all = FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        for (int i = 0; i < all.Length; i++)
        {
            if (all[i] != null && all[i].name == name) return all[i];
        }
        return null;
    }
}

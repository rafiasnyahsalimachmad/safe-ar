using System.Collections;
using UnityEngine;

// Umpan balik saat Supply Box masuk Receiving Zone.
// Komponen pilihan: pseudo-haptic (bunyi, perubahan warna, denyut zona, hambatan semu)
// dan fisika yang berperan (kejadian ditentukan oleh trigger collider, bukan hiasan).
[RequireComponent(typeof(AudioSource))]
public class SupplyReceivedFeedback : MonoBehaviour
{
    [Header("Referensi")]
    [SerializeField] private ReceivingZone zone;
    [SerializeField] private Renderer zoneRenderer;

    [Header("Tampilan")]
    [SerializeField] private float showSeconds = 2f;
    [SerializeField] private Color receivedColor = new Color(0.153f, 0.682f, 0.376f); // #27AE60
    [SerializeField] private float pulseScale = 1.15f;
    [SerializeField] private float pulseSeconds = 0.3f;

    [Header("Bunyi")]
    [SerializeField, Range(0f, 1f)] private float volume = 0.6f;

    [Header("Pseudo-haptic: hambatan semu")]
    [Tooltip("0 = mati. Nilai tinggi membuat kotak terasa tertahan saat masuk zona.")]
    [SerializeField, Range(0f, 1f)] private float resistanceStrength = 0.5f;
    [SerializeField] private float maxExtraDrag = 8f;

    private AudioSource audioSource;
    private AudioClip dingClip;
    private Color originalColor;
    private Vector3 originalScale;
    private bool showing;

    private void Awake()
    {
        audioSource = GetComponent<AudioSource>();
        audioSource.playOnAwake = false;
        audioSource.spatialBlend = 0f;
        dingClip = MakeDing();

        if (zoneRenderer) originalColor = zoneRenderer.material.color;
        if (zone) originalScale = zone.transform.localScale;
    }

    private void OnEnable()  { if (zone) zone.OnSupplyReceived += Handle; }
    private void OnDisable() { if (zone) zone.OnSupplyReceived -= Handle; }

    private void Handle(GameObject box)
    {
        ApplyResistance(box);
        audioSource.PlayOneShot(dingClip, volume);
        StopAllCoroutines();
        StartCoroutine(Show());
    }

    // Hambatan semu: kotak diperlambat sesaat setelah menyentuh zona,
    // seperti mendarat di permukaan empuk. Fisikanya tetap dihitung mesin fisika.
    private void ApplyResistance(GameObject box)
    {
        if (resistanceStrength <= 0f) return;
        var body = box.GetComponent<Rigidbody>();
        if (body == null) return;
        body.linearDamping += maxExtraDrag * resistanceStrength;
        body.angularDamping += maxExtraDrag * resistanceStrength;
    }

    private IEnumerator Show()
    {
        showing = true;
        if (zoneRenderer) zoneRenderer.material.color = receivedColor;

        // Denyut zona: membesar lalu kembali.
        if (zone)
        {
            float t = 0f;
            while (t < pulseSeconds)
            {
                t += Time.deltaTime;
                float k = Mathf.Sin(Mathf.Clamp01(t / pulseSeconds) * Mathf.PI);
                zone.transform.localScale = originalScale * Mathf.Lerp(1f, pulseScale, k);
                yield return null;
            }
            zone.transform.localScale = originalScale;
        }

        yield return new WaitForSeconds(showSeconds);

        if (zoneRenderer) zoneRenderer.material.color = originalColor;
        showing = false;
    }

    // Bunyi dua nada dibuat lewat kode, jadi tidak perlu berkas audio.
    private AudioClip MakeDing()
    {
        const int rate = 44100;
        const float duration = 0.35f;
        int samples = (int)(rate * duration);
        var data = new float[samples];
        float phase = 0f;

        for (int i = 0; i < samples; i++)
        {
            float t = i / (float)rate;
            float freq = t < 0.15f ? 660f : 880f;
            phase += 2f * Mathf.PI * freq / rate;
            float envelope = Mathf.Exp(-6f * t);
            data[i] = Mathf.Sin(phase) * envelope * 0.5f;
        }

        var clip = AudioClip.Create("SupplyDing", samples, 1, rate, false);
        clip.SetData(data, 0);
        return clip;
    }

    private void OnGUI()
    {
        if (!showing) return;
        var style = new GUIStyle(GUI.skin.label) { fontSize = 36, alignment = TextAnchor.MiddleCenter };
        GUI.Label(new Rect(0, 40, Screen.width, 60), "SUPPLY RECEIVED", style);
    }
}

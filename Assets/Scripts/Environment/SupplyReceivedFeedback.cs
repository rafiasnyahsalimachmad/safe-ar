using System.Collections;
using UnityEngine;

public class SupplyReceivedFeedback : MonoBehaviour
{
    [SerializeField] private ReceivingZone zone;
    [SerializeField] private Renderer zoneRenderer;
    [SerializeField] private float showSeconds = 2f;
    private bool showing;

    private void OnEnable()  { if (zone) zone.OnSupplyReceived += Handle; }
    private void OnDisable() { if (zone) zone.OnSupplyReceived -= Handle; }

    private void Handle(GameObject box)
    {
        StopAllCoroutines();
        StartCoroutine(Show());
    }

    private IEnumerator Show()
    {
        showing = true;
        if (zoneRenderer) zoneRenderer.material.color = Color.yellow;
        yield return new WaitForSeconds(showSeconds);
        showing = false;
    }

    private void OnGUI()
    {
        if (!showing) return;
        var style = new GUIStyle(GUI.skin.label) { fontSize = 36, alignment = TextAnchor.MiddleCenter };
        GUI.Label(new Rect(0, 40, Screen.width, 60), "SUPPLY RECEIVED", style);
    }
}
using System;
using UnityEngine;

[RequireComponent(typeof(Collider))]
public class ReceivingZone : MonoBehaviour
{
    [SerializeField] private string supplyTag = "SupplyBox";
    public event Action<GameObject> OnSupplyReceived;
    private bool received;

    private void Reset() { GetComponent<Collider>().isTrigger = true; }

    private void OnTriggerEnter(Collider other)
    {
        if (received || !other.CompareTag(supplyTag)) return;
        received = true;
        Debug.Log("SUPPLY RECEIVED");
        OnSupplyReceived?.Invoke(other.gameObject);
    }

    public void ResetZone() { received = false; }
}
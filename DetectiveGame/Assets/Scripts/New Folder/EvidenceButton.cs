using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class EvidenceButton : MonoBehaviour
{
    [Header("UI Components")]
    public Button button;
    public TMP_Text descriptionText;
    public Image backgroundImage;
    public GameObject revealedIcon;

    [Header("Evidence Data")]
    public int evidenceID;
    public string description;
    private bool isRevealed = false;

    [Header("Colors")]
    public Color hiddenColor = new Color(0.2f, 0.2f, 0.2f, 1f);
    public Color revealedColor = new Color(0.8f, 0.2f, 0.2f, 1f);

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        button.onClick.AddListener(OnClicked);
        UpdateVisuals();
    }

    public void Initialize(int id, string desc)
    {
        evidenceID = id;
        description = desc;

        descriptionText.text = isRevealed ? description : "Evidence ???";
        UpdateVisuals();
    }

    void OnClicked()
    {
        if (!isRevealed)
        {
            RevealEvidence();
        }
    }

    void RevealEvidence()
    {
        isRevealed = true;
        descriptionText.text = description;
        backgroundImage.color = revealedColor;
        if (revealedIcon) revealedIcon.SetActive(true);

        // Notify game manager
        GameManager.Instance.OnEvidenceRevealed(evidenceID, description);

        // Notify server
        FindObjectOfType<ServerManager>().RevealEvidence(evidenceID);

        Debug.Log($"📋 Evidence {evidenceID} revealed: {description}");
    }

    void UpdateVisuals()
    {
        backgroundImage.color = isRevealed ? revealedColor : hiddenColor;
        if (revealedIcon) revealedIcon.SetActive(isRevealed);
    }

    // Public method to reveal programmatically
    public void ForceReveal()
    {
        if (!isRevealed)
        {
            RevealEvidence();
        }
    }
}

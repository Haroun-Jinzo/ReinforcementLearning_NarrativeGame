using UnityEngine;
using TMPro;
using NUnit.Framework;
using System.Collections.Generic;

public class GameManager : MonoBehaviour
{
    public static GameManager Instance;

    [Header("UI References")]
    public TMP_InputField questionInput;
    public TMP_Text conversationText;
    public TMP_Text suspicionText;
    public TMP_Text questionsLeftText;
    public GameObject askButton;
    public GameObject accuseButton;
    public GameObject gameOverPanel;
    public TMP_Text gameOverText;

    [Header("Evidence System")]
    public GameObject evidencePanel;
    public GameObject evidenceButtonPrefab;
    public Transform evidenceContainer;
    public TMP_Text evidenceCountText;

    public ServerManager serverManager;
    private int questionsAsked = 0;
    private bool gameOver = false;
    private List<EvidenceButton> evidenceButtons = new List<EvidenceButton>();
    private int revealedEvidenceCount = 0;

    void Awake()
    {
        Instance = this;
    }

    public void OnGameStarted(StartGameResponse response)
    {
        AddToConversation($"<b>Case: {response.case_title}</b>\n");
        AddToConversation($"{response.intro}\n");
        UpdateUI(0.3f, 10);
    }

    public void InitializeEvidence(EvidenceItem[] evidenceList)
    {
        foreach (var btn in evidenceButtons)
        {
            Destroy(btn.gameObject);
        }
        evidenceButtons.Clear();

        for (int i = 0; i < evidenceList.Length; i++)
        {
            GameObject btnObj = Instantiate(evidenceButtonPrefab, evidenceContainer);
            EvidenceButton evidenceBtn = btnObj.GetComponent<EvidenceButton>();

            evidenceBtn.Initialize(evidenceList[i].id, evidenceList[i].description);
            evidenceButtons.Add(evidenceBtn);
        }

        UpdateEvidenceCount();
        Debug.Log($"📋 Initialized {evidenceList.Length} evidence items");
    }

    public void OnEvidenceRevealed(int evidenceId, string description)
    {
        revealedEvidenceCount++;

        AddToConversation($"\n<color=purple><b>Evidence Revealed:</b> {description}</color>\n");

        UpdateEvidenceCount();
    }

    void UpdateEvidenceCount()
    {
        int total = evidenceButtons.Count;
        evidenceCountText.text = $"Evidence: {revealedEvidenceCount}/{total}";
    }

    public void OnAskButtonClicked()
    {
        if (gameOver) return;

        string question = questionInput.text.Trim();
        if (string.IsNullOrEmpty(question))
        {
            Debug.LogWarning("Empty question!");
            return;
        }

        // Disable input while waiting
        askButton.SetActive(false);

        AddToConversation($"\n<color=blue><b>You:</b> {question}</color>\n");

        Debug.Log($"🔍 Asking question: {question}");

        StartCoroutine(serverManager.AskQuestion(question, OnQuestionResponse));

        questionInput.text = "";
    }

    void OnQuestionResponse(QuestionResponse response)
    {
        if (response.status == "success")
        {
            AddToConversation($"<color=red><b>Suspect:</b> {response.response}</color>\n");

            questionsAsked++;
            UpdateUI(response.suspicion_level, response.questions_remaining);

            if (response.contradiction_detected)
            {
                AddToConversation("<color=yellow>⚠️ Contradiction detected!</color>\n");
            }

            // Check if questions exhausted
            if (response.questions_remaining <= 0)
            {
                AddToConversation("\n<b>No more questions left. Time to decide!</b>\n");
                askButton.SetActive(false);
            }
            else
            {
                askButton.SetActive(true);
            }
        }
    }

    public void OnAccuseButtonClicked()
    {
        if (gameOver) return;

        askButton.SetActive(false);
        accuseButton.SetActive(false);

        StartCoroutine(serverManager.Accuse(OnAccusationResponse));
    }

    void OnAccusationResponse(AccusationResponse response)
    {
        gameOver = true;

        string result = response.outcome == "caught"
            ? "🎉 <color=green><b>GUILTY!</b></color> You caught the suspect!"
            : "❌ <color=red><b>NOT GUILTY.</b></color> The suspect escaped...";

        gameOverText.text = $"{result}\n\n" +
            $"Final Suspicion: {response.suspicion_level:P0}\n" +
            $"Contradictions Found: {response.total_contradictions}\n" +
            $"Questions Asked: {questionsAsked}";

        gameOverPanel.SetActive(true);
    }

    void AddToConversation(string text)
    {
        conversationText.text += text;
    }

    void UpdateUI(float suspicion, int questionsLeft)
    {
        suspicionText.text = $"Suspicion: {suspicion:P0}";
        questionsLeftText.text = $"Questions Left: {questionsLeft}";

        // Color code suspicion
        if (suspicion < 0.4f)
            suspicionText.color = Color.green;
        else if (suspicion < 0.65f)
            suspicionText.color = Color.yellow;
        else
            suspicionText.color = Color.red;
    }

    public void RestartGame()
    {
        UnityEngine.SceneManagement.SceneManager.LoadScene(0);
    }
}
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.Collections.Generic;

public class ServerManager : MonoBehaviour
{
    private const string SERVER_URL = "http://localhost:5000";
    private string sessionId;
    private List<int> revealedEvidence = new List<int>();

    void Start()
    {
        sessionId = "unity_" + System.Guid.NewGuid().ToString();
        Debug.Log($"📱 Session ID: {sessionId}");
        StartCoroutine(StartGame());
    }

    // Start a new game - FIXED
    public IEnumerator StartGame()
    {
        // ✅ Use the class, not anonymous object
        var data = new StartGameRequest
        {
            session_id = sessionId,
            case_id = "case_001"
        };

        string json = JsonUtility.ToJson(data);
        Debug.Log($"📤 Starting game: {json}");

        using (UnityWebRequest www = UnityWebRequest.Post(SERVER_URL + "/start_game", json, "application/json"))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("✅ Game started!");
                Debug.Log($"📥 Response: {www.downloadHandler.text}");

                var response = JsonUtility.FromJson<StartGameResponse>(www.downloadHandler.text);
                GameManager.Instance.OnGameStarted(response);

                // Load evidence after game starts
                StartCoroutine(LoadEvidence());
            }
            else
            {
                Debug.LogError("❌ Failed to start game: " + www.error);
            }
        }
    }

    // Ask a question - FIXED
    public IEnumerator AskQuestion(string playerQuestion, Action<QuestionResponse> callback)
    {
        // ✅ Use the class, not anonymous object
        var data = new QuestionRequest
        {
            session_id = sessionId,
            question = playerQuestion,
            evidence_shown = revealedEvidence.ToArray()
        };

        string json = JsonUtility.ToJson(data);
        Debug.Log($"📤 Asking question: {json}");

        using (UnityWebRequest www = UnityWebRequest.Post(SERVER_URL + "/ask_question", json, "application/json"))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"📥 Response: {www.downloadHandler.text}");
                var response = JsonUtility.FromJson<QuestionResponse>(www.downloadHandler.text);
                callback?.Invoke(response);
            }
            else
            {
                Debug.LogError("❌ Question failed: " + www.error);
                Debug.LogError($"Response Code: {www.responseCode}");
            }
        }
    }

    // Accuse the suspect - FIXED
    public IEnumerator Accuse(Action<AccusationResponse> callback)
    {
        // ✅ Use the class, not anonymous object
        var data = new SessionRequest { session_id = sessionId };
        string json = JsonUtility.ToJson(data);
        Debug.Log($"📤 Making accusation: {json}");

        using (UnityWebRequest www = UnityWebRequest.Post(SERVER_URL + "/accuse", json, "application/json"))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<AccusationResponse>(www.downloadHandler.text);
                callback?.Invoke(response);
            }
            else
            {
                Debug.LogError("❌ Accusation failed: " + www.error);
            }
        }
    }

    public void RevealEvidence(int evidenceId)
    {
        if (!revealedEvidence.Contains(evidenceId))
        {
            revealedEvidence.Add(evidenceId);
            Debug.Log($"📋 Revealed evidence #{evidenceId}");
        }
    }

    // Load evidence from server - FIXED
    public IEnumerator LoadEvidence()
    {
        // ✅ Use the class, not anonymous object
        var data = new SessionRequest { session_id = sessionId };
        string json = JsonUtility.ToJson(data);
        Debug.Log($"📤 Loading evidence: {json}");

        using (UnityWebRequest www = UnityWebRequest.Post(SERVER_URL + "/get_evidence", json, "application/json"))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"📥 Evidence response: {www.downloadHandler.text}");
                var response = JsonUtility.FromJson<EvidenceResponse>(www.downloadHandler.text);
                GameManager.Instance.InitializeEvidence(response.evidence);
            }
            else
            {
                Debug.LogError("❌ Failed to load evidence: " + www.error);
            }
        }
    }
}

// ==================== REQUEST CLASSES ====================

[Serializable]
public class SessionRequest
{
    public string session_id;
}

[Serializable]
public class StartGameRequest
{
    public string session_id;
    public string case_id;
}

[Serializable]
public class QuestionRequest
{
    public string session_id;
    public string question;
    public int[] evidence_shown;
}

// ==================== RESPONSE CLASSES ====================

[Serializable]
public class StartGameResponse
{
    public string status;
    public string case_title;
    public string intro;
    public int evidence_count;
}

[Serializable]
public class QuestionResponse
{
    public string status;
    public string response;
    public float suspicion_level;
    public int questions_remaining;
    public bool contradiction_detected;
    public int evidence_revealed;
}

[Serializable]
public class AccusationResponse
{
    public string status;
    public string outcome;
    public float suspicion_level;
    public int total_contradictions;
}

[Serializable]
public class EvidenceResponse
{
    public string status;
    public EvidenceItem[] evidence;
}

[Serializable]
public class EvidenceItem
{
    public int id;
    public string description;
    public bool revealed;
}
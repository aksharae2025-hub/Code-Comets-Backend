# app.py
"""
HealthHelper backend (Flask). Features:
- /diagnose: AI-based symptom classification via Hugging Face Inference API
- Context-aware: fetches last few messages from Supabase to add context
- Stores chat history to Supabase (chat_history)
- Stores hospital recommendations to Supabase (hospital_history)
- Emergency keyword detection
- Summary endpoint (simple aggregator, optional HF summarization if configured)
"""

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ---------- Config (from .env or Render environment variables) ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")          # Use service_role key on server for full DB access if needed
HF_API_TOKEN = os.getenv("HF_API_TOKEN")          # Hugging Face inference API token
HF_MODEL = os.getenv("HF_MODEL", "Lech-Iyoko/bert-symptom-checker")
SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "") # optional: e.g. "facebook/bart-large-cnn"
USE_HF_SUMMARY = bool(SUMMARY_MODEL and HF_API_TOKEN)

# Safety: ensure critical keys are present (we will still run but log warnings)
if not HF_API_TOKEN:
    print("WARNING: HF_API_TOKEN not set. Hugging Face calls will fail without a token.")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE variables missing. Supabase operations will be disabled.")

# Create Supabase client (or None)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# HF inference endpoint base
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# ---------- Flask init ----------
app = Flask(__name__)
CORS(app)


# ---------- Helpers ----------
EMERGENCY_KEYWORDS = [
    "chest pain", "shortness of breath", "cannot breathe", "fainting", "severe bleeding",
    "unconscious", "severe pain", "loss of consciousness"
]

ADVICE_BANK = {
    "Flu": "You might be experiencing a mild influenza infection. Rest, hydrate, and monitor your temperature. If fever persists >48 hours or you have difficulty breathing, seek medical care.",
    "Common Cold": "This looks like a common cold. Rest, fluids, and OTC medications may help. If symptoms worsen, see a doctor.",
    "Allergy": "This seems like an allergic reaction. Avoid suspected triggers and consider an antihistamine. If breathing is affected, seek emergency care.",
    "Migraine": "Symptoms suggest a migraine. Try resting in a dark room, hydration, and OTC pain relief if appropriate. If severe or unusual, see a clinician.",
    "COVID": "Some symptoms match COVID-19. Please test if possible, isolate, and seek medical advice for breathing issues.",
    "default": "These symptoms could have several causes. Keep monitoring and consult a healthcare professional when concerned."
}

def detect_emergency(text):
    t = text.lower()
    for key in EMERGENCY_KEYWORDS:
        if key in t:
            return True, key
    return False, None

def call_hf_inference(prompt_text):
    """
    Sends prompt_text to the configured Hugging Face model via the inference API.
    Returns the raw JSON response or raises an exception.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not configured")
    resp = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt_text}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def save_chat_to_supabase(user_id, user_msg, bot_response, diagnosis_label=None, confidence=None):
    if not supabase:
        return False
    try:
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "user_message": user_msg,
            "bot_response": bot_response,
            "diagnosis": diagnosis_label,
            "confidence": confidence
        }).execute()
        return True
    except Exception as e:
        app.logger.error("Supabase insert chat failed: %s", e)
        return False

def save_hospitals_to_supabase(hospitals, location):
    if not supabase:
        return False
    try:
        # insert many
        for h in hospitals:
            supabase.table("hospital_history").insert({
                "hospital_name": h.get("hospital_name"),
                "location": location,
                "contact_info": h.get("contact_info")
            }).execute()
        return True
    except Exception as e:
        app.logger.error("Supabase insert hospitals failed: %s", e)
        return False

def fetch_recent_chats(user_id="guest", limit=3):
    if not supabase:
        return []
    try:
        res = supabase.table("chat_history").select("user_message, bot_response, diagnosis, created_at").eq("user_id", user_id).order("id", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as e:
        app.logger.error("Supabase fetch recent chats failed: %s", e)
        return []


def make_friendly_reply(label, confidence, advice_text, emergency=False, emergency_note=None):
    """Return a human-friendly multi-line text reply (string)"""
    lines = []
    lines.append(f"🩺 Possible condition: **{label}**")
    lines.append(f"💡 Confidence: {round(confidence * 100, 2)}%")
    lines.append("")  # blank
    lines.append(f"💬 Advice: {advice_text}")
    if emergency:
        lines.append("")
        lines.append("⚠️ Emergency flag: This message contains signs of a potentially serious condition. Please seek immediate medical attention or call emergency services.")
        if emergency_note:
            lines.append(f"Detected phrase: \"{emergency_note}\"")
    return "\n".join(lines)


# ---------- Routes ----------

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "HealthHelper backend is alive", "timestamp": datetime.utcnow().isoformat()}), 200


@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    Body: { "symptoms": "...", "user_id": "optional" }
    Returns JSON:
    { diagnosis: "Label", confidence: 0.82, response_text: "...", emergency: bool }
    """
    try:
        data = request.get_json(force=True)
        symptoms = (data.get("symptoms") or "").strip()
        user_id = data.get("user_id") or "guest"

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # 1) Check for emergency keywords first (fast path)
        emergency, emergency_note = detect_emergency(symptoms)
        if emergency:
            # Fast emergency reply and still log it
            emergency_reply = ("⚠️ Emergency detected: please seek immediate medical help. "
                               f"Detected phrase: '{emergency_note}'")
            save_chat_to_supabase(user_id, symptoms, emergency_reply, diagnosis_label="EMERGENCY", confidence=1.0)
            return jsonify({"diagnosis": "EMERGENCY", "confidence": 1.0, "response": emergency_reply, "emergency": True}), 200

        # 2) Build context: fetch last few chats and append the new message
        recent = fetch_recent_chats(user_id=user_id, limit=3)
        context_text = ""
        if recent:
            # Build a simple context string: "User: ... Bot: ..."
            for c in reversed(recent):  # oldest -> newest
                u = c.get("user_message", "")
                b = c.get("bot_response", "")
                context_text += f"User: {u}\nAssistant: {b}\n"
        context_text += f"User: {symptoms}\n"

        # 3) Send to Hugging Face inference API
        hf_json = call_hf_inference(context_text)

        # hf_json expected format: list of {"label": "...", "score": 0.xx}
        if isinstance(hf_json, list) and len(hf_json) > 0:
            top = hf_json[0]
            label = top.get("label", "Unknown")
            confidence = float(top.get("score", 0.0))
        else:
            # fallback
            label = "Unknown"
            confidence = 0.0

        # 4) Pick advice text
        advice_text = ADVICE_BANK.get(label, ADVICE_BANK["default"])

        # 5) Construct friendly reply
        reply_text = make_friendly_reply(label, confidence, advice_text, emergency=False)

        # 6) Save the chat in Supabase
        save_chat_to_supabase(user_id, symptoms, reply_text, diagnosis_label=label, confidence=confidence)

        # 7) Return structured JSON
        return jsonify({
            "diagnosis": label,
            "confidence": round(confidence, 4),
            "response": reply_text,
            "emergency": False
        }), 200

    except requests.exceptions.RequestException as e:
        app.logger.error("HF API request failed: %s", e)
        return jsonify({"error": "External inference API failed", "details": str(e)}), 502
    except Exception as e:
        app.logger.error("Internal server error: %s", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/find_hospitals", methods=["POST"])
def find_hospitals():
    """
    Body: { "location": "city or coordinates", "user_id": "optional" }
    Returns a small list of recommended hospitals and logs them to Supabase.
    NOTE: This demo uses a static/hardcoded list. Replace with real API (Google Places) if needed.
    """
    try:
        data = request.get_json(force=True)
        location = data.get("location", "unknown location")
        user_id = data.get("user_id", "guest")

        # Dummy/hardcoded recommendations (for hackathon demo). Replace with real API integration later.
        hospitals = [
            {"hospital_name": "Apollo Hospitals", "location": location, "contact_info": "https://apollohospitals.com"},
            {"hospital_name": "Fortis Healthcare", "location": location, "contact_info": "https://www.fortishealthcare.com"},
            {"hospital_name": "CityCare Clinic", "location": location, "contact_info": "tel:+911234567890"}
        ]

        # Save to Supabase
        save_hospitals_to_supabase(hospitals, location)

        return jsonify({"results": hospitals, "message": f"Top hospitals in {location}"}), 200

    except Exception as e:
        app.logger.error("find_hospitals error: %s", e)
        return jsonify({"error": "Unable to find hospitals", "details": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    """
    Query params: user_id (optional), limit (optional)
    Example: /history?user_id=guest&limit=10
    """
    try:
        user_id = request.args.get("user_id", "guest")
        limit = int(request.args.get("limit", 10))
        if not supabase:
            return jsonify({"error": "Supabase not configured"}), 500
        res = supabase.table("chat_history").select("*").eq("user_id", user_id).order("id", desc=True).limit(limit).execute()
        return jsonify({"history": res.data or []}), 200
    except Exception as e:
        app.logger.error("history error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/summary", methods=["GET"])
def summary():
    """
    Simple summary endpoint.
    If SUMMARY_MODEL and HF_API_TOKEN are set, calls HF summarization model (optional).
    Otherwise returns a short aggregated summary of last N chats.
    Query params: user_id (optional), limit (optional)
    """
    try:
        user_id = request.args.get("user_id", "guest")
        limit = int(request.args.get("limit", 10))

        recent = []
        if supabase:
            recent = supabase.table("chat_history").select("user_message, bot_response, diagnosis, created_at").eq("user_id", user_id).order("id", desc=True).limit(limit).execute().data or []
        if not recent:
            return jsonify({"summary": "No history for this user."}), 200

        # Build a plain-text summary candidate
        bullets = []
        diagnoses = {}
        for r in recent:
            u = r.get("user_message", "")
            d = r.get("diagnosis", "Unknown")
            ts = r.get("created_at", "")
            bullets.append(f"- {ts}: \"{u}\" -> {d}")
            diagnoses[d] = diagnoses.get(d, 0) + 1

        simple_summary = "Recent activity:\n" + "\n".join(bullets) + "\n\nDiagnosis counts: " + ", ".join(f"{k}({v})" for k,v in diagnoses.items())

        # Optionally call HF summarization model if configured
        if USE_HF_SUMMARY:
            try:
                # prepare text for summarization
                long_text = "\n".join([f"User: {r.get('user_message','')} Assistant: {r.get('bot_response','')}" for r in reversed(recent)])
                hf_summary_url = f"https://api-inference.huggingface.co/models/{SUMMARY_MODEL}"
                resp = requests.post(hf_summary_url, headers=HF_HEADERS, json={"inputs": long_text}, timeout=30)
                resp.raise_for_status()
                summ = resp.json()
                # many summarization models return [{"summary_text":"..."}] or a string
                summary_text = ""
                if isinstance(summ, list) and len(summ) and "summary_text" in summ[0]:
                    summary_text = summ[0]["summary_text"]
                elif isinstance(summ, str):
                    summary_text = summ
                else:
                    summary_text = simple_summary
                return jsonify({"summary": summary_text}), 200
            except Exception as e:
                app.logger.error("HF summary failed: %s", e)
                # fallback to simple summary
                return jsonify({"summary": simple_summary}), 200

        # default fallback
        return jsonify({"summary": simple_summary}), 200

    except Exception as e:
        app.logger.error("summary error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Save user feedback: { user_id, message, rating }
    """
    try:
        data = request.get_json(force=True)
        user_id = data.get("user_id", "guest")
        message = data.get("message", "")
        rating = data.get("rating", None)
        if not supabase:
            return jsonify({"error": "Supabase not configured"}), 500
        supabase.table("feedback").insert({
            "user_id": user_id,
            "message": message,
            "rating": rating
        }).execute()
        return jsonify({"ok": True}), 200
    except Exception as e:
        app.logger.error("feedback error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)








 

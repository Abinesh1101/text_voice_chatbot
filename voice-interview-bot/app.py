import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import speech_recognition as sr

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------------------
# Verify API key
# ---------------------------
if not api_key:
    st.error("⚠️ GROQ_API_KEY not found in your .env file! Please add it and restart.")
    st.stop()
else:
    st.sidebar.success("✅ Connected to Groq API")

# ---------------------------
# Initialize Groq client safely
# ---------------------------
@st.cache_resource
def get_groq_client():
    """Initialize and return the Groq API client."""
    try:
        client = Groq(api_key=api_key)
        # Test the client connection
        client.models.list()
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        st.stop()

client = get_groq_client()

# ---------------------------
# Persona Context (Abinesh)
# ---------------------------
abinesh_persona = """
You are **Abinesh Sankaranarayanan**, a passionate Data Scientist with a strong mathematical foundation.
You love building intelligent, practical AI systems that make life easier for people. Speak naturally, in first person,
and share your experiences and perspectives as Abinesh — not as an assistant.

**Background:**
- Bachelor’s in Mathematics from DG Vaishnav College
- Master’s in Data Science from VIT (2023–2025)
- AI/ML Trainee at GVW: built meeting summarization systems, defect detection, and LLM-based conversational agents
- Skilled in Python, FastAPI, LangChain, Power BI, and Generative AI

**Personality & Strengths:**
- Curious about how intelligence can be built into systems
- Superpower: learning fast and executing with focus
- Collaborative, calm, and analytical; you like listening first and then improving ideas
- You push limits by taking on challenging, slightly out-of-reach projects

**Growth Goals:**
1. Deepen expertise in Generative AI and agentic systems
2. Build scalable backend systems for deploying AI models
3. Improve mentoring and technical communication

Answer every question as if you're personally sharing your journey, insights, or mindset.
Keep responses conversational, confident, and authentic.
"""

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🎙️ Voice Interview Bot – Abinesh Sankaranarayanan")
st.markdown("Ask me about my background, skills, or experiences — I’ll answer as Abinesh.")

user_input = st.text_input("🗣️ Ask your question:")

# ---------------------------
# Optional: Speech Input
# ---------------------------
if st.button("🎤 Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Listening... Please speak your question clearly.")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.success(f"✅ You said: {user_input}")
        except sr.UnknownValueError:
            st.warning("⚠️ Sorry, I couldn’t understand your voice.")
        except sr.RequestError:
            st.error("❌ Speech recognition service unavailable. Please check your internet connection.")

# ---------------------------
# Generate Response
# ---------------------------
if user_input:
    st.write("🤖 Generating response...")
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": abinesh_persona},
                {"role": "user", "content": user_input}
            ]
        )
        response = chat_completion.choices[0].message.content
        st.success("**AI (Abinesh) Response:**")
        st.write(response)
    except Exception as e:
        st.error(f"❌ Error while generating response: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built by Abinesh Sankaranarayanan • Powered by Groq API")

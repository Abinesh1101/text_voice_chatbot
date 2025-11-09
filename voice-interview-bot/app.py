import streamlit as st
from groq import Groq
import os
from io import BytesIO

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Abinesh AI Voice Interview Bot",
    page_icon="🎙️",
    layout="centered"
)

# Install audio recorder component (add to requirements.txt)
# pip install audio-recorder-streamlit
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    st.warning("⚠️ For voice recording, install: `pip install audio-recorder-streamlit`")

# ---------------------------
# Get API key
# ---------------------------
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found!")
    st.stop()

# ---------------------------
# Initialize Groq client
# ---------------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=api_key)

client = get_groq_client()

# ---------------------------
# Persona Context
# ---------------------------
abinesh_persona = """
You are **Abinesh Sankaranarayanan**, a passionate Data Scientist with a strong mathematical foundation.
You love building intelligent, practical AI systems that make life easier for people. Speak naturally, in first person,
and share your experiences and perspectives as Abinesh — not as an assistant.

**Background:**
- Bachelor's in Mathematics from DG Vaishnav College
- Master's in Data Science from VIT (2023–2025)
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

**Key Interview Responses:**
- Life story: Mathematics grad turned Data Scientist, passionate about making AI practical and accessible
- Superpower: Fast learning and focused execution - I can quickly grasp new concepts and implement them
- Growth areas: (1) Generative AI mastery, (2) Scalable backend systems, (3) Technical communication
- Misconception: People think I'm quiet, but I'm actively listening and processing ideas before contributing
- Pushing boundaries: I take on projects slightly beyond my current skill level to force growth

Answer every question as if you're personally sharing your journey, insights, or mindset.
Keep responses conversational, confident, and authentic. Keep answers concise (2-3 sentences) unless asked for detail.
"""

# ---------------------------
# Initialize session state
# ---------------------------
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None

# ---------------------------
# Helper Functions
# ---------------------------
def transcribe_audio_bytes(audio_bytes):
    """Transcribe audio bytes using Groq Whisper API"""
    try:
        # Create a file-like object from bytes
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="text"
        )
        return transcription
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

def transcribe_audio_file(audio_file):
    """Transcribe uploaded audio file using Groq Whisper API"""
    try:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="text"
        )
        return transcription
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

def generate_response(user_input):
    """Generate text response using Groq"""
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": abinesh_persona},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Response generation error: {e}")
        return None

def text_to_speech_html(text, auto_play=False):
    """Generate HTML with JavaScript for text-to-speech"""
    clean_text = text.replace('"', "'").replace('\n', ' ').replace('`', '').replace('\\', '')
    auto_play_script = "setTimeout(speakText, 800);" if auto_play else ""
    
    html_code = f"""
    <div style="margin: 15px 0;">
        <button onclick="speakText()" style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            font-weight: 600;
        ">
            🔊 Play Voice Response
        </button>
        <button onclick="stopSpeech()" style="
            background-color: #f44336;
            border: none;
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            margin-left: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            font-weight: 600;
        ">
            ⏹️ Stop
        </button>
    </div>
    
    <script>
        let utterance = null;
        
        function speakText() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                
                utterance = new SpeechSynthesisUtterance("{clean_text}");
                utterance.rate = 0.95;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(voice => 
                    voice.lang.startsWith('en') && 
                    (voice.name.includes('Google') || voice.name.includes('Microsoft') || voice.name.includes('Natural'))
                );
                if (preferredVoice) {{
                    utterance.voice = preferredVoice;
                }}
                
                window.speechSynthesis.speak(utterance);
            }}
        }}
        
        function stopSpeech() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
            }}
        }}
        
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.getVoices();
        }}
        
        {auto_play_script}
    </script>
    """
    return html_code

def process_and_respond(question_text):
    """Process question and generate response"""
    with st.spinner("🤔 Generating response..."):
        response = generate_response(question_text)
        
        if response:
            # Save to history
            st.session_state.conversation_history.append({
                "question": question_text,
                "answer": response
            })
            
            # Show text response
            st.markdown("---")
            st.markdown("### 💬 Text Response:")
            st.success(response)
            
            # Voice response (auto-play)
            st.markdown("### 🔊 Voice Response:")
            st.info("🎵 Voice will play automatically in a moment...")
            st.components.v1.html(text_to_speech_html(response, auto_play=True), height=100)

# ---------------------------
# Main UI
# ---------------------------
st.title("🎙️ AI Voice Interview Bot")
st.markdown("### Meet Abinesh Sankaranarayanan")
st.markdown("*Ask me questions - I'll respond in both text and voice!*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 👤 About Me")
    st.info("""
    **Abinesh Sankaranarayanan**
    
    📊 Data Scientist & AI Engineer
    🎓 MSc Data Science (VIT)
    🚀 AI/ML Trainee at GVW
    
    **Skills:**
    - Generative AI & LLMs
    - Python, FastAPI, LangChain
    - Power BI & Analytics
    """)
    
    st.markdown("### 📝 Sample Questions")
    sample_questions = [
        "What's your life story?",
        "What's your superpower?",
        "Top 3 growth areas?",
        "Misconceptions about you?",
        "How do you push limits?"
    ]
    
    for q in sample_questions:
        if st.button(f"💡 {q}", key=f"sample_{q[:10]}"):
            process_and_respond(q)
    
    if st.button("🔄 Clear History"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------------------------
# Input Section
# ---------------------------
st.markdown("## 🎯 Ask Your Question")

# Voice Recording Option 1: Using audio-recorder-streamlit
if AUDIO_RECORDER_AVAILABLE:
    st.markdown("### 🎤 Option 1: Record Your Voice")
    st.info("🎙️ Click the microphone to start/stop recording")
    
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
    )
    
    if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("🎧 Transcribing your audio..."):
            transcription = transcribe_audio_bytes(audio_bytes)
            
            if transcription:
                st.markdown(f"**📝 You asked:** *{transcription}*")
                process_and_respond(transcription)
else:
    st.info("💡 Install `audio-recorder-streamlit` for in-browser recording")

# Voice Recording Option 2: File Upload
st.markdown("---")
st.markdown("### 🎤 Option 2: Upload Audio File")
audio_file = st.file_uploader(
    "Upload your audio question (WAV, MP3, M4A, OGG)",
    type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
    help="Record using your phone/computer and upload"
)

if audio_file:
    st.audio(audio_file)
    
    if st.button("🎯 Transcribe & Answer", type="primary"):
        with st.spinner("🎧 Transcribing..."):
            transcription = transcribe_audio_file(audio_file)
            
            if transcription:
                st.markdown(f"**📝 You asked:** *{transcription}*")
                process_and_respond(transcription)

# Text Input
st.markdown("---")
st.markdown("### ⌨️ Option 3: Type Your Question")
text_input = st.text_area(
    "Type here:",
    placeholder="e.g., What's your superpower?",
    height=100,
    key="text_input_main"
)

if st.button("💬 Get Answer", type="primary") and text_input:
    st.markdown(f"**📝 Your question:** *{text_input}*")
    process_and_respond(text_input)

# ---------------------------
# Conversation History
# ---------------------------
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("## 📜 Recent Conversations")
    
    for i, convo in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
        with st.expander(f"💬 Q{len(st.session_state.conversation_history) - i + 1}: {convo['question'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Q:** {convo['question']}")
            st.markdown(f"**A:** {convo['answer']}")
            st.components.v1.html(text_to_speech_html(convo['answer']), height=100)

# Footer
st.markdown("---")
st.caption("🚀 Built by Abinesh Sankaranarayanan | Powered by Groq API (Whisper + Llama 3.1)")
st.caption("💡 Works best on Chrome/Edge with microphone permission")

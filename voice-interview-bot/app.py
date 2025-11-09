import streamlit as st
from groq import Groq
import os
import base64
from io import BytesIO

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Abinesh AI Voice Interview Bot",
    page_icon="🎙️",
    layout="centered"
)

# ---------------------------
# Get API key from Streamlit secrets or environment
# ---------------------------
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found! Please add it to Streamlit secrets.")
    st.stop()

# ---------------------------
# Initialize Groq client
# ---------------------------
@st.cache_resource
def get_groq_client():
    try:
        client = Groq(api_key=api_key)
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
if 'audio_response' not in st.session_state:
    st.session_state.audio_response = None

# ---------------------------
# Helper Functions
# ---------------------------
def transcribe_audio(audio_file):
    """Transcribe audio using Groq Whisper API"""
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

def text_to_speech_html(text):
    """Generate HTML with JavaScript for text-to-speech"""
    # Clean text for speech
    clean_text = text.replace('"', "'").replace('\n', ' ')
    
    html_code = f"""
    <div style="margin: 20px 0;">
        <button onclick="speakText()" style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        ">
            🔊 Play Voice Response
        </button>
        <button onclick="stopSpeech()" style="
            background-color: #f44336;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            margin-left: 10px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        ">
            ⏹️ Stop
        </button>
    </div>
    
    <script>
        let utterance = null;
        
        function speakText() {{
            if ('speechSynthesis' in window) {{
                // Stop any ongoing speech
                window.speechSynthesis.cancel();
                
                utterance = new SpeechSynthesisUtterance("{clean_text}");
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                
                // Try to use a natural voice
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(voice => 
                    voice.lang.startsWith('en') && 
                    (voice.name.includes('Google') || voice.name.includes('Microsoft'))
                );
                if (preferredVoice) {{
                    utterance.voice = preferredVoice;
                }}
                
                window.speechSynthesis.speak(utterance);
            }} else {{
                alert('Text-to-speech is not supported in your browser.');
            }}
        }}
        
        function stopSpeech() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
            }}
        }}
        
        // Load voices
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.getVoices();
        }}
    </script>
    """
    return html_code

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🎙️ AI Voice Interview Bot")
st.markdown("### Meet Abinesh Sankaranarayanan - AI/ML Engineer")
st.markdown("*Ask me anything about my background, skills, and experiences!*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.success("✅ Voice Bot Active")
    st.markdown("### 👤 About Me")
    st.info("""
    **Abinesh Sankaranarayanan**
    
    📊 Data Scientist & AI Engineer
    
    🎓 MSc Data Science (VIT)
    
    🔧 Expertise:
    - Generative AI & LLMs
    - LangChain & FastAPI
    - Power BI & Analytics
    - Machine Learning
    
    🚀 Currently building intelligent AI systems at GVW
    """)
    
    st.markdown("### 🎤 How to Use")
    st.markdown("""
    1. **Upload Audio**: Record your question and upload
    2. **Or Type**: Type your question directly
    3. **Listen**: Click play to hear my response
    """)
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------------------------
# Input Methods
# ---------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 🎤 Voice Input")
    audio_file = st.file_uploader(
        "Upload your audio question (WAV, MP3, M4A, OGG)",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Record your question using your phone or computer and upload it here"
    )

with col2:
    st.markdown("#### ⌨️ Text Input")
    text_input = st.text_input(
        "Or type your question:",
        placeholder="e.g., What's your superpower?",
        label_visibility="collapsed"
    )

# ---------------------------
# Process Input
# ---------------------------
user_question = None

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("🎯 Transcribe & Answer", type="primary"):
        with st.spinner("🎧 Listening to your question..."):
            # Transcribe audio
            transcribed_text = transcribe_audio(audio_file)
            
            if transcribed_text:
                user_question = transcribed_text
                st.success(f"📝 You asked: *{transcribed_text}*")

elif text_input:
    if st.button("💬 Get Answer", type="primary"):
        user_question = text_input

# ---------------------------
# Generate Response
# ---------------------------
if user_question:
    with st.spinner("🤔 Thinking..."):
        response = generate_response(user_question)
        
        if response:
            # Save to history
            st.session_state.conversation_history.append({
                "question": user_question,
                "answer": response
            })
            
            # Display response
            st.markdown("---")
            st.markdown("### 💬 Response:")
            st.markdown(f"**Q:** {user_question}")
            st.markdown(f"**A:** {response}")
            
            # Add text-to-speech
            st.markdown("### 🔊 Voice Response:")
            st.components.v1.html(text_to_speech_html(response), height=100)

# ---------------------------
# Conversation History
# ---------------------------
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    
    for i, convo in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
        with st.expander(f"💬 Question {len(st.session_state.conversation_history) - i + 1}: {convo['question'][:60]}..."):
            st.markdown(f"**Q:** {convo['question']}")
            st.markdown(f"**A:** {convo['answer']}")
            # Add play button for history items too
            st.components.v1.html(text_to_speech_html(convo['answer']), height=100)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("🚀 Built by Abinesh Sankaranarayanan | Powered by Groq API (Whisper + Llama 3.1)")
st.caption("💡 Tip: Use your phone's voice recorder to create audio files, then upload here!")

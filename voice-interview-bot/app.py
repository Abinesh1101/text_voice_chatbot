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

def text_to_speech_html(text, auto_play=False):
    """Generate HTML with JavaScript for text-to-speech"""
    clean_text = text.replace('"', "'").replace('\n', ' ').replace('`', '')
    auto_play_script = "setTimeout(speakText, 800);" if auto_play else ""
    
    html_code = f"""
    <div style="margin: 15px 0;">
        <button onclick="speakText()" style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        ">
            🔊 Play Response
        </button>
        <button onclick="stopSpeech()" style="
            background-color: #f44336;
            border: none;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            margin-left: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
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
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(voice => 
                    voice.lang.startsWith('en') && 
                    (voice.name.includes('Google') || voice.name.includes('Microsoft'))
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

def get_voice_recorder_html():
    """Simple in-browser voice recorder"""
    return """
    <div style="padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="color: white; margin-bottom: 20px;">🎤 Voice Recording</h3>
        
        <button id="recordBtn" onclick="toggleRecording()" style="
            background-color: #ff4b4b;
            border: none;
            color: white;
            padding: 18px 40px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 50px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: all 0.3s;
        ">
            <span id="recordIcon">🎙️</span>
            <span id="recordText">Start Recording</span>
        </button>
        
        <div id="status" style="color: white; margin-top: 15px; font-size: 16px;">
            Click to start speaking
        </div>
        
        <div id="timer" style="color: white; font-size: 28px; font-weight: bold; 
                                  margin-top: 10px; font-family: monospace;">
        </div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let timer;
        let seconds = 0;
        
        function toggleRecording() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            value: reader.result
                        }, '*');
                    };
                };
                
                mediaRecorder.start();
                isRecording = true;
                seconds = 0;
                
                document.getElementById('recordBtn').style.backgroundColor = '#4CAF50';
                document.getElementById('recordIcon').textContent = '⏹️';
                document.getElementById('recordText').textContent = 'Stop Recording';
                document.getElementById('status').textContent = '🔴 Recording...';
                
                timer = setInterval(() => {
                    seconds++;
                    const mins = Math.floor(seconds / 60);
                    const secs = seconds % 60;
                    document.getElementById('timer').textContent = 
                        `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                }, 1000);
                
            } catch (err) {
                document.getElementById('status').textContent = '❌ Microphone access denied';
                console.error(err);
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                clearInterval(timer);
                
                document.getElementById('recordBtn').style.backgroundColor = '#ff4b4b';
                document.getElementById('recordIcon').textContent = '🎙️';
                document.getElementById('recordText').textContent = 'Start Recording';
                document.getElementById('status').textContent = '✅ Processing your audio...';
            }
        }
    </script>
    """

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
    st.markdown("""
    - What's your life story?
    - What's your superpower?
    - Top 3 growth areas?
    - Misconceptions about you?
    - How do you push limits?
    """)
    
    if st.button("🔄 Clear History"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------------------------
# Input Section
# ---------------------------
st.markdown("## 🎯 Ask Your Question")

# Voice Input
st.markdown("### 🎤 Option 1: Speak Your Question")
voice_data = st.components.v1.html(get_voice_recorder_html(), height=250)

st.markdown("**OR upload audio file:**")
audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3', 'm4a', 'ogg'], label_visibility="collapsed")

if audio_file:
    st.audio(audio_file)
    if st.button("🎯 Process Audio", type="primary"):
        with st.spinner("🎧 Transcribing..."):
            transcription = transcribe_audio(audio_file)
            
            if transcription:
                st.markdown(f"**📝 You asked:** *{transcription}*")
                
                with st.spinner("🤔 Generating response..."):
                    response = generate_response(transcription)
                    
                    if response:
                        # Save to history
                        st.session_state.conversation_history.append({
                            "question": transcription,
                            "answer": response
                        })
                        
                        # Show text response
                        st.markdown("---")
                        st.markdown("### 💬 Text Response:")
                        st.success(response)
                        
                        # Voice response (auto-play)
                        st.markdown("### 🔊 Voice Response:")
                        st.components.v1.html(text_to_speech_html(response, auto_play=True), height=80)

# Text Input
st.markdown("---")
st.markdown("### ⌨️ Option 2: Type Your Question")
text_input = st.text_area(
    "Type here:",
    placeholder="e.g., What's your superpower?",
    height=80,
    label_visibility="collapsed"
)

if st.button("💬 Get Answer", type="primary") and text_input:
    with st.spinner("🤔 Generating response..."):
        response = generate_response(text_input)
        
        if response:
            # Save to history
            st.session_state.conversation_history.append({
                "question": text_input,
                "answer": response
            })
            
            # Show text response
            st.markdown("---")
            st.markdown("### 💬 Text Response:")
            st.success(response)
            
            # Voice response (auto-play)
            st.markdown("### 🔊 Voice Response:")
            st.components.v1.html(text_to_speech_html(response, auto_play=True), height=80)

# ---------------------------
# Conversation History
# ---------------------------
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("## 📜 Conversation History")
    
    for i, convo in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
        with st.expander(f"💬 {convo['question'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Q:** {convo['question']}")
            st.markdown(f"**A:** {convo['answer']}")
            st.components.v1.html(text_to_speech_html(convo['answer']), height=80)

# Footer
st.markdown("---")
st.caption("🚀 Built by Abinesh | Powered by Groq API (Whisper + Llama 3.1)")
st.caption("💡 Works best on Chrome/Edge with microphone permission")

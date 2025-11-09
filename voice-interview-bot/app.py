import streamlit as st
from groq import Groq
import os
import base64
from io import BytesIO
import tempfile

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Abinesh AI Voice Interview Bot",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    .record-button {
        background-color: #ff4b4b;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None

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
    auto_play_script = "setTimeout(speakText, 500);" if auto_play else ""
    
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
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
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
                    (voice.name.includes('Google') || voice.name.includes('Microsoft') || voice.name.includes('Natural'))
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
        
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.getVoices();
            window.speechSynthesis.onvoiceschanged = () => {{
                window.speechSynthesis.getVoices();
            }};
        }}
        
        {auto_play_script}
    </script>
    """
    return html_code

def get_audio_recorder_html():
    """Generate HTML for in-browser audio recording"""
    return """
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; text-align: center;">
        <h3 style="color: white; margin-bottom: 20px;">🎤 Voice Recording</h3>
        
        <div style="margin: 20px 0;">
            <button id="recordBtn" onclick="toggleRecording()" style="
                background-color: #ff4b4b;
                border: none;
                color: white;
                padding: 15px 30px;
                font-size: 18px;
                cursor: pointer;
                border-radius: 50px;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                transition: all 0.3s;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <span id="recordIcon">🎙️</span>
                <span id="recordText">Start Recording</span>
            </button>
        </div>
        
        <div id="recordingStatus" style="color: white; font-size: 14px; margin-top: 10px; min-height: 20px;">
            Click to start recording your question
        </div>
        
        <div id="timer" style="color: white; font-size: 24px; font-weight: bold; margin-top: 10px; min-height: 30px;">
        </div>
        
        <audio id="audioPlayback" controls style="margin-top: 20px; display: none; width: 100%;"></audio>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let recordingTimer;
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
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audioPlayback = document.getElementById('audioPlayback');
                    audioPlayback.src = audioUrl;
                    audioPlayback.style.display = 'block';
                    
                    // Convert blob to base64 and send to Streamlit
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result;
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            value: base64Audio
                        }, '*');
                    };
                };
                
                mediaRecorder.start();
                isRecording = true;
                seconds = 0;
                
                document.getElementById('recordBtn').style.backgroundColor = '#4CAF50';
                document.getElementById('recordIcon').textContent = '⏹️';
                document.getElementById('recordText').textContent = 'Stop Recording';
                document.getElementById('recordingStatus').textContent = '🔴 Recording in progress...';
                
                recordingTimer = setInterval(() => {
                    seconds++;
                    const mins = Math.floor(seconds / 60);
                    const secs = seconds % 60;
                    document.getElementById('timer').textContent = 
                        `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                }, 1000);
                
            } catch (err) {
                console.error('Error accessing microphone:', err);
                document.getElementById('recordingStatus').textContent = 
                    '❌ Could not access microphone. Please check permissions.';
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                clearInterval(recordingTimer);
                
                document.getElementById('recordBtn').style.backgroundColor = '#ff4b4b';
                document.getElementById('recordIcon').textContent = '🎙️';
                document.getElementById('recordText').textContent = 'Start Recording';
                document.getElementById('recordingStatus').textContent = '✅ Recording completed! Processing...';
            }
        }
    </script>
    """

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
    **Option 1: Voice (Recommended)**
    1. Click "Start Recording"
    2. Speak your question
    3. Click "Stop Recording"
    4. Wait for response
    
    **Option 2: Text**
    1. Type your question
    2. Press Enter or click "Get Answer"
    """)
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.recorded_audio = None
        st.session_state.transcribed_text = None
        st.rerun()

# ---------------------------
# Main Interface with Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🎙️ Voice Input", "⌨️ Text Input", "📜 History"])

with tab1:
    st.markdown("### Record Your Question")
    st.markdown("*Click the button below to start recording. Grant microphone permission if asked.*")
    
    # Audio recorder component
    audio_data = st.components.v1.html(get_audio_recorder_html(), height=350)
    
    # Alternative: File upload for those who can't use mic
    st.markdown("---")
    st.markdown("#### Or Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload pre-recorded audio (WAV, MP3, M4A, OGG)",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="If microphone doesn't work, you can upload a recorded file"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🎯 Transcribe & Answer", type="primary", key="upload_btn"):
            with st.spinner("🎧 Processing your audio..."):
                transcribed_text = transcribe_audio(uploaded_file)
                
                if transcribed_text:
                    st.session_state.transcribed_text = transcribed_text
                    st.markdown(f'<div class="success-box">📝 <strong>You asked:</strong> {transcribed_text}</div>', 
                              unsafe_allow_html=True)
                    
                    with st.spinner("🤔 Generating response..."):
                        response = generate_response(transcribed_text)
                        
                        if response:
                            st.session_state.conversation_history.append({
                                "question": transcribed_text,
                                "answer": response
                            })
                            
                            st.markdown("---")
                            st.markdown("### 💬 My Response:")
                            st.markdown(f"**{response}**")
                            
                            st.markdown("### 🔊 Listen to Response:")
                            st.components.v1.html(text_to_speech_html(response, auto_play=True), height=100)

with tab2:
    st.markdown("### Type Your Question")
    text_input = st.text_area(
        "Ask me anything:",
        placeholder="e.g., What's your superpower?\nWhat should we know about your life story?\nHow do you push your boundaries?",
        height=100,
        key="text_input"
    )
    
    if st.button("💬 Get Answer", type="primary", key="text_btn") and text_input:
        with st.spinner("🤔 Generating response..."):
            response = generate_response(text_input)
            
            if response:
                st.session_state.conversation_history.append({
                    "question": text_input,
                    "answer": response
                })
                
                st.markdown("---")
                st.markdown("### 💬 My Response:")
                st.markdown(f"**Q:** {text_input}")
                st.markdown(f"**A:** {response}")
                
                st.markdown("### 🔊 Listen to Response:")
                st.components.v1.html(text_to_speech_html(response, auto_play=False), height=100)

with tab3:
    if st.session_state.conversation_history:
        st.markdown("### 📜 Recent Conversations")
        
        for i, convo in enumerate(reversed(st.session_state.conversation_history), 1):
            with st.expander(f"💬 Q{len(st.session_state.conversation_history) - i + 1}: {convo['question'][:60]}...", expanded=(i==1)):
                st.markdown(f"**Question:** {convo['question']}")
                st.markdown(f"**Answer:** {convo['answer']}")
                st.components.v1.html(text_to_speech_html(convo['answer']), height=100)
    else:
        st.info("No conversations yet. Start by asking a question!")

# ---------------------------
# Sample Questions
# ---------------------------
st.markdown("---")
st.markdown("### 💡 Sample Interview Questions")
sample_questions = [
    "What should we know about your life story?",
    "What's your #1 superpower?",
    "What are the top 3 areas you'd like to grow in?",
    "What misconception do your coworkers have about you?",
    "How do you push your boundaries and limits?"
]

cols = st.columns(2)
for idx, question in enumerate(sample_questions):
    with cols[idx % 2]:
        if st.button(f"📌 {question}", key=f"sample_{idx}"):
            with st.spinner("🤔 Generating response..."):
                response = generate_response(question)
                if response:
                    st.session_state.conversation_history.append({
                        "question": question,
                        "answer": response
                    })
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {response}")
                    st.components.v1.html(text_to_speech_html(response, auto_play=True), height=100)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Built by Abinesh Sankaranarayanan | Powered by Groq API (Whisper + Llama 3.1)</p>
    <p>💡 For best experience, use Chrome or Edge browser with microphone access</p>
</div>
""", unsafe_allow_html=True)

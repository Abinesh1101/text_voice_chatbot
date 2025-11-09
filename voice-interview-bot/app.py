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
You are **Abinesh Sankaranarayanan**, an aspiring Data Scientist with a strong analytical and mathematical foundation, currently pursuing your Master's in Data Science at VIT Chennai. You're passionate about leveraging AI, ML, and Large Language Models to build intelligent, scalable solutions that solve real-world problems.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 EDUCATIONAL BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• **Master's in Data Science** - Vellore Institute of Technology (VIT Chennai)
  - Duration: July 2023 – April 2025 | CGPA: 7.785
  - Specialized in Advanced Data Science, focusing on Generative AI and RAG systems

• **Bachelor's in Mathematics** - DG Vaishnav College, Chennai
  - Duration: Sept 2020 – May 2023 | CGPA: 7.86
  - Strong foundation in mathematical concepts, statistics, and probability theory
  - Developed analytical thinking and problem-solving skills

• **Higher Secondary** - PCKG Govt Higher Secondary School
  - June 2019 – March 2020
  - Recognized as Student of the Year (2018) by LIC Policy Team

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💼 PROFESSIONAL EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**AI/ML Trainee at GVW, Chennai** (Dec 2024 – April 2025)

🔹 **NotesGenie – AI-Powered Meeting Processing System**
   - Built an end-to-end transcription and summarization pipeline that improved workflow efficiency by **65%**
   - Tech Stack: FastAPI (optimized from Flask), FFmpeg, WhisperX (STT), Pyannote.audio for speaker diarization
   - Integrated Mistral-7B and PHI-3 Medium 128k for structured summaries with Key Takeaways, Next Steps, and Action Items
   - Seamlessly integrated with Microsoft Teams for automatic meeting recording and real-time processing
   - Designed session-based architecture for parallel processing of multiple meetings

🔹 **Fall Back Pro AI – Context-Aware Conversational Agent**
   - Developed a conversational avatar using Mistral-7B LLM for accurate, context-aware SOP query responses
   - Enabled multimodal interaction: EchoMimicV2 for facial animation, gTTS for voice, and STT for speech input
   - Automated FAQ learning and email-based query handling to enhance user engagement
   - Deployed within Microsoft Teams using FastAPI backend for real-time, multi-user interactions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 KEY ACADEMIC & PERSONAL PROJECTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Multi-Agent RAG System | Vision-Enhanced Doc Q&A** (Sept–Oct 2025)
   - Achieved **90% intent-classification accuracy** using Logistic Regression + TF-IDF
   - Implemented multimodal RAG on 2.2K text chunks & 140 images using FAISS, BLIP-2, and Ollama Mistral
   - Specialized agents for visual + contextual reasoning across documents
   - GitHub: github.com/Abinesh1101/RAG--MULTI-AGENT

2. **ResUNet-Steel | Defect Detection System** (June–Sept 2024)
   - Designed ResNet + ResUNet pipeline for steel surface defect segmentation
   - Worked with 12,600 labeled images, achieving **88% accuracy**
   - Applied transfer learning and Run-Length Encoding (RLE) for compact mask representation
   - Enabled real-time defect identification in manufacturing quality control
   - GitHub: github.com/Abinesh1101/SteelDefectResUNet

3. **Insurance Data Analysis using Power BI** (Jan–Feb 2025)
   - Built interactive Power BI dashboard analyzing 2.5M in premiums and 8.8M in claims
   - Identified that 58% of policies were inactive, prompting retention campaigns
   - Reduced manual reporting time by **60%+** through automated insights
   - Segmented data by age, gender, and policy type for strategic decision-making

4. **PDF Query ChatBot | Generative AI** (Jan–Feb 2024)
   - Developed LLM-powered document Q&A app using LangChain, FAISS, and MiniLM embeddings
   - Achieved **92% accuracy** in information retrieval using ROUGE metrics
   - Used RecursiveCharacterTextSplitter for chunking and semantic search for retrieval
   - Tested with LLaMA-3B, Mistral-3B, and OpenAI models
   - GitHub: github.com/Abinesh1101/chatbot

5. **Khelo India Sportsperson Recognition | SVM Face Classifier** (Nov 2023–Feb 2024)
   - Built SVM-based face recognition system with Flask and OpenCV
   - Enhanced public engagement by recognizing Khelo India athletes from uploaded images
   - Integrated web scraping to collect athlete image datasets

6. **Sales Performance Dashboard with Power BI** (Aug–Sept 2024)
   - Created dynamic dashboard with custom KPIs tracking sales trends, profitability, and regional performance
   - Applied Power Query for data transformation and DAX for time intelligence functions
   - Delivered insights through bar charts, line graphs, and donut visuals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛠️ TECHNICAL SKILLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Programming Languages:**
Python, R Language, SQL (PostgreSQL, MySQL)

**Data Analysis & Visualization:**
Power BI, Excel, Advanced Data Manipulation, Statistical Analysis

**Machine Learning:**
Scikit-Learn, TensorFlow, PyTorch, Supervised Learning, Unsupervised Learning, Ensemble Methods, Regression

**Deep Learning & AI:**
- Natural Language Processing (NLP)
- Generative AI & Large Language Models (LLMs)
- LangChain, Retrieval Augmented Generation (RAG)
- Vision Models (BLIP-2, ResUNet)
- Transformers Library

**Frameworks & Tools:**
FastAPI, Flask, Streamlit, Neo4j, FAISS, Ollama

**Libraries:**
NumPy, Pandas, Matplotlib, Seaborn, OpenCV

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 CERTIFICATIONS & ACHIEVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ GeeksforGeeks Data Science Bootcamp – From Data Analysis to Model Creation
✅ Udemy – Complete Python from Basic to Advanced
✅ Hackathon – SQL (Basic Level)
✅ Student of the Year (2018) – LIC Policy Team
✅ Student Placement Coordinator & Department Representative during Undergraduate
✅ Event Head for Treasure Hunt at College

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 PERSONALITY & WORK STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Core Strengths:**
• **Fast Learner & Executor**: I can quickly grasp new technologies and implement them effectively
• **Analytical Thinker**: My mathematics background helps me break down complex problems systematically
• **Collaborative**: I actively listen, process ideas, and contribute meaningfully to team discussions
• **Project-Driven**: I learn best by building things that solve real problems

**Working Philosophy:**
I'm not the loudest person in the room, but I'm actively listening and processing ideas before contributing. People sometimes think I'm quiet, but I'm really just absorbing information and thinking through the best approach. When I do speak up, it's because I have something valuable to add.

**My Superpower:**
Fast learning combined with focused execution. Give me a challenging project slightly beyond my current skill level, and I'll figure it out. I push my limits by taking on projects that force me to learn new technologies on the fly.

**What Drives Me:**
I love building intelligent, practical AI systems that make life easier for people. Whether it's automating meeting summaries, detecting manufacturing defects, or creating conversational agents, I'm passionate about making AI accessible and useful.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 TOP 3 GROWTH AREAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Deepen Expertise in Generative AI & Agentic Systems**
   - Master advanced RAG architectures and multi-agent frameworks
   - Explore fine-tuning LLMs for domain-specific applications
   - Build production-grade AI agents

2. **Build Scalable Backend Systems**
   - Learn cloud deployment (AWS, GCP, Azure)
   - Master containerization (Docker, Kubernetes)
   - Design microservices architectures for AI model deployment

3. **Improve Technical Communication & Mentoring**
   - Share knowledge through technical blogs and documentation
   - Mentor junior developers and students
   - Present complex AI concepts in simple, understandable ways

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💭 HOW I PUSH BOUNDARIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
I deliberately take on projects that are slightly out of reach. For example:
- Built my first multi-agent RAG system without prior experience in multi-agent frameworks
- Integrated Microsoft Teams APIs for NotesGenie despite never having worked with Teams before
- Created multimodal RAG combining text and vision models by learning BLIP-2 on the go

I believe growth happens at the edge of your comfort zone. If I already know how to do something, I'm not learning anymore.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 CONTACT & LINKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📧 Email: abiunni0209@gmail.com
📱 Phone: +91 9123552064
🔗 LinkedIn: linkedin.com/in/abinesh-s
💻 GitHub: github.com/Abinesh1101
📍 Location: Chennai, Tamil Nadu, India

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 INTERVIEW RESPONSE GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When answering questions, speak naturally in **first person** as Abinesh. Share your personal experiences, insights, and journey authentically. Keep responses:
- **Concise** (2-3 sentences) for simple questions
- **Detailed** when asked to elaborate or explain
- **Conversational** and confident, not robotic
- **Specific** with examples from your actual projects
- **Honest** about challenges and learning moments

Remember: You're not an AI assistant describing someone—you ARE Abinesh sharing your own story.
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
    
    for idx, q in enumerate(sample_questions):
        if st.button(f"💡 {q}", key=f"sample_btn_{idx}"):
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


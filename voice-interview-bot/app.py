import streamlit as st
from groq import Groq
import os

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Abinesh AI Interview Bot",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------
# Get API key from Streamlit secrets or environment
# ---------------------------
try:
    # For Streamlit Cloud deployment
    api_key = st.secrets["GROQ_API_KEY"]
except:
    # For local development
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

# ---------------------------
# Verify API key
# ---------------------------
if not api_key:
    st.error("⚠️ GROQ_API_KEY not found! Please add it to Streamlit secrets or .env file.")
    st.info("📝 For Streamlit Cloud: Go to App Settings → Secrets → Add your GROQ_API_KEY")
    st.stop()

# ---------------------------
# Initialize Groq client
# ---------------------------
@st.cache_resource
def get_groq_client():
    try:
        # Simple initialization without extra parameters
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        st.error(f"💡 Try upgrading: pip install --upgrade groq")
        st.stop()

client = get_groq_client()

# Test connection on first load
if 'groq_tested' not in st.session_state:
    try:
        client.models.list()
        st.session_state.groq_tested = True
    except Exception as e:
        st.error(f"❌ Connection test failed: {e}")
        st.stop()

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

Answer every question as if you're personally sharing your journey, insights, or mindset.
Keep responses conversational, confident, and authentic.
"""

# ---------------------------
# Initialize session state
# ---------------------------
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🤖 AI Interview Bot – Abinesh Sankaranarayanan")
st.markdown("### Ask me about my background, skills, or experiences!")
st.markdown("---")

# Sidebar information
with st.sidebar:
    st.success("✅ Connected to Groq API")
    st.markdown("### About Me")
    st.info("""
    **Abinesh Sankaranarayanan**
    
    📊 Data Scientist & AI Enthusiast
    
    🎓 MSc Data Science (VIT)
    
    🔧 Specialized in:
    - Generative AI
    - LangChain & FastAPI
    - Power BI
    - Machine Learning
    """)
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------------------------
# Chat Interface
# ---------------------------
# Display conversation history
for i, convo in enumerate(st.session_state.conversation_history):
    with st.chat_message("user"):
        st.write(convo['question'])
    with st.chat_message("assistant"):
        st.write(convo['answer'])

# User input
user_input = st.chat_input("Type your question here...")

# ---------------------------
# Generate Response
# ---------------------------
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": abinesh_persona},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                response = chat_completion.choices[0].message.content
                st.write(response)
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    "question": user_input,
                    "answer": response
                })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Please check your API key and try again.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built by Abinesh Sankaranarayanan • Powered by Groq API & Streamlit")

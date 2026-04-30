import streamlit as st
from vid_whisper import VidWhisper
from langchain_core.messages import HumanMessage, AIMessage
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="VidWhisper AI", page_icon="🎙️", layout="centered")

# Custom CSS for modern look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        background-color: #ff4b4b;
        color: white;
    }
    .chat-bubble {
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ VidWhisper AI")
st.markdown("Turn YouTube videos into AI conversations.")

# --- CACHING ---
@st.cache_resource
def get_vid_whisper(video_id):
    """Initialize VidWhisper with caching to avoid redundant reruns."""
    return VidWhisper(video_id)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    video_id = st.text_input("YouTube Video ID", placeholder="e.g., FQp7iI9vjl8")
    
    # Check for API Key in .env
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success("✅ API Key found in .env")
    else:
        st.error("❌ API Key not found in .env. Please add it to your environment variables.")

    if st.button("🚀 Process Video"):
        if not video_id:
            st.warning("Please enter a Video ID first.")
        elif not api_key:
            st.error("Missing OpenAI API Key in .env")
        else:
            # Initialize VidWhisper (using cache)
            vw = get_vid_whisper(video_id)
            
            # Progress tracking
            progress_bar = st.progress(0, text="Starting...")
            
            def update_progress(msg, val):
                progress_bar.progress(val, text=msg)

            try:
                # Process the video
                db = vw.process_video(progress_callback=update_progress)
                st.session_state.db = db
                st.session_state.current_video = video_id
                st.session_state.vw = vw
                st.session_state.messages = [] # Reset chat for new video
                st.success("🎉 Video processed and indexed!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.markdown("### 💡 Tips")
    st.info("Ask for a summary to understand the whole video, or ask specific questions to learn about any particular part.")

# --- MAIN CHAT AREA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize VidWhisper for existing video ID in sidebar if not in session_state
if "current_video" in st.session_state and st.session_state.current_video == video_id:
    # Everything is already set up in session state
    pass
elif video_id:
    # Try loading existing DB if it's already indexed on disk
    vw_temp = get_vid_whisper(video_id)
    if vw_temp.is_indexed():
        if "db" not in st.session_state or st.session_state.current_video != video_id:
             with st.spinner("📂 Loading existing database..."):
                db = vw_temp.load_db()
                st.session_state.db = db
                st.session_state.current_video = video_id
                st.session_state.vw = vw_temp
                st.session_state.messages = []
                st.info(f"Loaded existing index for Video: {video_id}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask something about the video..."):
    if "db" not in st.session_state:
        st.warning("Please process a video first in the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    # 1. Format history for LangChain (limited to last 10 messages)
                    # Note: We take last 10 messages, but since messages include the current 'user' query,
                    # we should take up to 11 if we count the just-added one, 
                    # or simply take the previous 10.
                    # Let's take the last 10 from session_state before the current query was added 
                    # OR just slice the current list.
                    history_to_pass = []
                    # st.session_state.messages contains [..., old_ai, new_user]
                    # We take the messages BEFORE the current user query for the history argument
                    raw_history = st.session_state.messages[:-1][-10:] 
                    
                    for m in raw_history:
                        if m["role"] == "user":
                            history_to_pass.append(HumanMessage(content=m["content"]))
                        else:
                            history_to_pass.append(AIMessage(content=m["content"]))
                    
                    # 2. Get response with history
                    response = st.session_state.vw.ask(query, st.session_state.db, chat_history=history_to_pass)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # 3. Optional: Clean up session state if it gets too long
                    if len(st.session_state.messages) > 20: # 10 exchanges = 20 messages
                        st.session_state.messages = st.session_state.messages[-20:]
                        
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

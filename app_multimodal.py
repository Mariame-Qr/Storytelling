"""
AutoStory Multimodal - Streamlit Frontend
Advanced Multimodal Automotive Storytelling Interface
"""

import os
import time
import streamlit as st
from pathlib import Path
from backend_multimodal import run_autostory_multimodal_crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AutoStory Multimodal - AI Storyteller",
    page_icon="🚗🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .multimodal-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-text { background: #e3f2fd; color: #1976d2; }
    .badge-audio { background: #f3e5f5; color: #7b1fa2; }
    .badge-image { background: #fff3e0; color: #e65100; }
    .badge-video { background: #e8f5e9; color: #2e7d32; }
    
    .story-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .decision-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "HUGGINGFACE_API_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please create a `.env` file with your API keys.")
        return False
    return True


def check_qdrant_db():
    """Check if Qdrant database exists"""
    if not Path("qdrant_db").exists():
        st.warning("⚠️ Qdrant database not found. Please run `python ingest.py` first.")
        return False
    return True


def display_multimodal_outputs(result: dict):
    """Display multimodal outputs"""
    outputs = result.get("outputs", {})
    
    # Display strategy decision
    st.markdown('<div class="decision-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Orchestrator Decision")
    st.markdown(f"**Strategy**: {result.get('strategy', 'UNKNOWN')}")
    
    decision = result.get("decision", {})
    if decision:
        st.markdown(f"**Feature**: {decision.get('feature_name', 'N/A')}")
        st.markdown(f"**Classification**: {decision.get('intent_classification', 'N/A')}")
        if decision.get("reasoning"):
            st.markdown(f"**Reasoning**: {decision['reasoning']}")
    
    modalities = decision.get("modalities", [])
    st.markdown("**Modalities Generated**:")
    for mod in modalities:
        badge_class = f"badge-{mod.lower()}"
        st.markdown(f'<span class="multimodal-badge {badge_class}">{mod}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display story
    st.markdown("### 📖 Automotive Story")
    story = result.get("story", "")
    st.markdown(f'<div class="story-container">{story}</div>', unsafe_allow_html=True)
    
    # Display outputs in columns
    st.markdown("### 🎨 Generated Outputs")
    
    # Check what outputs are available
    has_audio = "audio" in outputs and os.path.exists(outputs.get("audio", ""))
    has_image = "image" in outputs and os.path.exists(outputs.get("image", ""))
    has_video = "final_video" in outputs and os.path.exists(outputs.get("final_video", ""))
    has_simple_video = "video" in outputs and os.path.exists(outputs.get("video", ""))
    
    # Determine layout based on available outputs
    if has_audio and (has_image or has_video or has_simple_video):
        col1, col2 = st.columns(2)
    else:
        col1, col2 = st.columns(1), None
    
    with col1:
        # Audio
        if "audio" in outputs:
            audio_path = outputs["audio"]
            if os.path.exists(audio_path):
                st.markdown("#### 🎤 Audio Narration")
                st.audio(audio_path)
                
                with open(audio_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Audio",
                        f.read(),
                        file_name=f"narration_{int(time.time())}.mp3",
                        mime="audio/mpeg"
                    )
        
        # Image
        if "image" in outputs:
            image_path = outputs["image"]
            if os.path.exists(image_path):
                st.markdown("#### 🖼️ Generated Image")
                st.image(image_path, use_container_width=True)
                
                with open(image_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Image",
                        f.read(),
                        file_name=f"image_{int(time.time())}.png",
                        mime="image/png"
                    )
    
    if col2:
        with col2:
            # Video
            if "final_video" in outputs:
                video_path = outputs["final_video"]
                if os.path.exists(video_path):
                    st.markdown("#### 🎬 Final Narrated Video")
                    st.video(video_path)
                    
                    with open(video_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Video",
                            f.read(),
                            file_name=f"autostory_{int(time.time())}.mp4",
                            mime="video/mp4"
                        )
            
            elif "video" in outputs:
                video_path = outputs["video"]
                if os.path.exists(video_path):
                    st.markdown("#### 🎥 Video")
                    st.video(video_path)
                    
                    with open(video_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Video",
                            f.read(),
                            file_name=f"video_{int(time.time())}.mp4",
                            mime="video/mp4"
                        )
    else:
        # Single column layout for single output
        if has_audio and not (has_image or has_video or has_simple_video):
            audio_path = outputs["audio"]
            st.markdown("#### 🎤 Audio Narration")
            st.audio(audio_path)
            with open(audio_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Audio",
                    f.read(),
                    file_name=f"narration_{int(time.time())}.mp3",
                    mime="audio/mpeg"
                )
        
        elif has_image and not (has_audio or has_video or has_simple_video):
            image_path = outputs["image"]
            st.markdown("#### 🖼️ Generated Image")
            st.image(image_path, use_container_width=True)
            with open(image_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Image",
                    f.read(),
                    file_name=f"image_{int(time.time())}.png",
                    mime="image/png"
                )
        
        elif (has_video or has_simple_video) and not (has_audio or has_image):
            video_path = outputs.get("final_video") or outputs.get("video")
            st.markdown("#### 🎥 Generated Video")
            st.video(video_path)
            with open(video_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Video",
                    f.read(),
                    file_name=f"video_{int(time.time())}.mp4",
                    mime="video/mp4"
                )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗🎬 AutoStory Multimodal</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent Multimodal Automotive Storytelling Agent</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/car.png", width=150)
        st.markdown("### ⚙️ System Status")
        
        env_ok = check_environment()
        db_ok = check_qdrant_db()
        
        if env_ok:
            st.success("✅ API Keys Configured")
        if db_ok:
            st.success("✅ Knowledge Base Ready")
        
        if not (env_ok and db_ok):
            st.error("❌ System Not Ready")
            st.stop()
        
        st.markdown("---")
        
        st.markdown("### 🎭 Multimodal Capabilities")
        st.markdown("""
        <span class="multimodal-badge badge-text">📝 TEXT</span>
        <span class="multimodal-badge badge-audio">🎤 AUDIO</span>
        <span class="multimodal-badge badge-image">🖼️ IMAGE</span>
        <span class="multimodal-badge badge-video">🎬 VIDEO</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🤖 Agentic System")
        st.markdown("""
        - **Orchestrator**: Strategy decision
        - **Technical Expert**: Retrieves specs
        - **Storyteller**: Crafts narrative
        - **Audio Agent**: Generates narration
        - **Creative Director**: Generates visuals
        - **Video Assembler**: Merges outputs
        """)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Output Format Selection")
        
        output_format = st.radio(
            "Choose your preferred output format:",
            ["🎬 Full Multimodal (Audio + Image + Video)", 
             "🎤 Audio Only (Narration)", 
             "🖼️ Image Only (Visual)", 
             "🎥 Video Only (Silent Video)"],
            index=0
        )
        
        # Parse selection
        if "Audio Only" in output_format:
            format_preference = "audio"
        elif "Image Only" in output_format:
            format_preference = "image"
        elif "Video Only" in output_format:
            format_preference = "video"
        else:
            format_preference = "full"
        
        st.markdown("---")
        
        st.markdown("### 💡 Example Prompts")
        examples = [
            "Show me how ABS prevents wheel lockup during emergency braking",
            "Visualize the airbag deployment system in action",
            "Explain how the all-wheel drive system distributes torque",
            "Show the adaptive cruise control maintaining distance",
        ]
        
        for example in examples:
            if st.button(f"📝 {example[:35]}...", key=example, use_container_width=True):
                st.session_state.selected_example = example
    
    # Initialize session state
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Main input area
    st.markdown("### 🎤 What would you like to visualize?")
    
    default_value = st.session_state.selected_example or ""
    if st.session_state.selected_example:
        st.session_state.selected_example = None
    
    user_query = st.text_area(
        "Describe the automotive feature or system:",
        value=default_value,
        height=100,
        placeholder="Example: Show me how the ABS system works during emergency braking"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Multimodal Story", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Generate AutoStory
    if generate_button and user_query.strip():
        st.markdown("---")
        
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Multimodal Workflow in Progress...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("**Phase 1/5**: 🧠 Orchestrator analyzing strategy...")
            progress_bar.progress(10)
            
            time.sleep(0.5)
            status_text.markdown("**Phase 2/5**: 🔍 Technical Expert retrieving specifications...")
            progress_bar.progress(25)
            
            time.sleep(0.5)
            status_text.markdown("**Phase 3/5**: ✍️ Storyteller crafting narrative...")
            progress_bar.progress(40)
            
            time.sleep(0.5)
            status_text.markdown("**Phase 4/5**: 🎨 Generating multimodal content...")
            progress_bar.progress(60)
            
            # Execute crew
            try:
                result = run_autostory_multimodal_crew(
                    user_query, 
                    format_preference=format_preference
                )
                progress_bar.progress(100)
                status_text.markdown("**✅ Workflow Complete!**")
                time.sleep(1)
                progress_container.empty()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
                st.stop()
        
        # Display results
        if result.get("success"):
            display_multimodal_outputs(result)
            
            # Add to history
            st.session_state.history.append({
                "query": user_query,
                "result": result,
                "timestamp": time.time()
            })
        else:
            st.error("❌ Workflow failed.")
            st.code(result.get("error", "Unknown error"))
    
    # History section
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📚 Session History")
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"🚗 {item['query'][:60]}..."):
                st.markdown(f"**Query**: {item['query']}")
                st.markdown(f"**Strategy**: {item['result'].get('strategy', 'N/A')}")
                st.markdown(f"**Execution Time**: {item['result'].get('execution_time', 0):.1f}s")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">Powered by CrewAI • Google Gemini • SDXL • gTTS • MoviePy • Qdrant</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

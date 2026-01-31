"""
AutoStory - Streamlit Frontend
Intelligent Automotive Storytelling Agent Interface
"""

import os
import time
import streamlit as st
from pathlib import Path
from backend import run_autostory_crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AutoStory - Automotive AI Storyteller",
    page_icon="🚗",
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
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .tech-specs {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    .example-prompt {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        cursor: pointer;
        border: 1px solid #dee2e6;
        transition: all 0.3s;
    }
    
    .example-prompt:hover {
        background: #e9ecef;
        border-color: #adb5bd;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY", "HUGGINGFACE_API_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please create a `.env` file with your API keys. See `.env.example` for reference.")
        return False
    return True


def check_qdrant_db():
    """Check if Qdrant database exists"""
    if not Path("qdrant_db").exists():
        st.warning("⚠️ Qdrant database not found. Please run `python ingest.py` first.")
        return False
    return True


def display_example_prompts():
    """Display example prompts users can try"""
    st.sidebar.markdown("### 💡 Example Prompts")
    
    examples = [
        "Show me how ABS prevents wheel lockup during emergency braking",
        "Visualize the airbag deployment system in action",
        "Explain how the all-wheel drive system distributes torque",
        "Show the engine delivering maximum torque",
        "Demonstrate how electronic stability control prevents skidding",
        "Visualize the adaptive cruise control maintaining distance",
        "Show the interior dashboard and infotainment system",
        "Illustrate how the suspension adapts to road conditions"
    ]
    
    for example in examples:
        if st.sidebar.button(f"📝 {example[:40]}...", key=example, use_container_width=True):
            st.session_state.selected_example = example


def format_story_text(story: str) -> str:
    """Format story text for better readability"""
    # Remove excessive newlines
    story = "\n\n".join([p.strip() for p in story.split("\n\n") if p.strip()])
    return story


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 AutoStory</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent Automotive Storytelling Agent</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/car.png", width=150)
        st.markdown("### ⚙️ System Status")
        
        # Environment check
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
        
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Enter your request** about any automotive feature
        2. **AI analyzes** the technical requirements
        3. **Agents decide** whether to create an image or video
        4. **Visual generated** showing the exact feature
        """)
        
        st.markdown("---")
        
        display_example_prompts()
        
        st.markdown("---")
        
        st.markdown("### 🧠 Agentic System")
        st.markdown("""
        - **Technical Expert**: Retrieves specs
        - **Storyteller**: Crafts narrative
        - **Creative Director**: Generates visuals
        """)
    
    # Initialize session state
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Main input area
    st.markdown("### 🎤 What would you like to visualize?")
    
    # Use selected example if available
    default_value = st.session_state.selected_example or ""
    if st.session_state.selected_example:
        st.session_state.selected_example = None  # Clear after using
    
    user_query = st.text_area(
        "Describe the automotive feature or system you want to see:",
        value=default_value,
        height=100,
        placeholder="Example: Show me how the ABS system works during emergency braking"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate AutoStory", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    with col3:
        show_tech = st.checkbox("Show Technical Details", value=False)
    
    # Generate AutoStory
    if generate_button and user_query.strip():
        st.markdown("---")
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Agentic Workflow in Progress...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 1: Research
            status_text.markdown("**Phase 1/3**: 🔍 Technical Expert retrieving specifications...")
            progress_bar.progress(10)
            
            # Phase 2: Storytelling
            time.sleep(1)
            status_text.markdown("**Phase 2/3**: ✍️ Storyteller crafting narrative...")
            progress_bar.progress(30)
            
            # Phase 3: Visual Generation
            time.sleep(1)
            status_text.markdown("**Phase 3/3**: 🎨 Creative Director generating visuals...")
            progress_bar.progress(50)
            
            # Execute crew
            try:
                result = run_autostory_crew(user_query)
                progress_bar.progress(100)
                status_text.markdown("**✅ Workflow Complete!**")
                time.sleep(1)
                progress_container.empty()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()
        
        # Display results
        if result.get("success"):
            # Format decision
            st.markdown('<div class="decision-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 Creative Director Decision")
            
            format_used = result.get("format_used", "image").upper()
            format_icon = "🖼️" if format_used == "IMAGE" else "🎬"
            
            st.markdown(f"**Visual Format Selected**: {format_icon} **{format_used}**")
            
            # Parse decision rationale from visual_decision
            decision_text = result.get("visual_decision", "")
            if "RATIONALE" in decision_text or "DECISION" in decision_text:
                st.markdown("**Rationale**:")
                # Extract rationale section
                lines = decision_text.split("\n")
                for line in lines:
                    if "rationale" in line.lower() or "decision" in line.lower():
                        st.markdown(f"- {line.strip()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Story
            st.markdown("### 📖 Automotive Story")
            story_text = format_story_text(result.get("story", ""))
            st.markdown(f'<div class="story-container">{story_text}</div>', unsafe_allow_html=True)
            
            # Visual output
            st.markdown("### 🎨 Generated Visual")
            
            visual_path = result.get("visual_path", "")
            
            if visual_path and os.path.exists(visual_path):
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    if visual_path.endswith(".mp4"):
                        st.video(visual_path)
                        st.success("🎬 Video generated successfully!")
                    else:
                        st.image(visual_path, use_container_width=True)
                        st.success("🖼️ Image generated successfully!")
                
                with col_right:
                    st.markdown("**Output Details**")
                    st.markdown(f"- Format: {format_used}")
                    st.markdown(f"- File: `{os.path.basename(visual_path)}`")
                    st.markdown(f"- Time: {result.get('execution_time', 0):.1f}s")
                    
                    # Download button
                    with open(visual_path, "rb") as file:
                        file_bytes = file.read()
                        file_ext = "mp4" if format_used == "VIDEO" else "png"
                        st.download_button(
                            label=f"⬇️ Download {format_used}",
                            data=file_bytes,
                            file_name=f"autostory_{int(time.time())}.{file_ext}",
                            mime=f"{'video' if format_used == 'VIDEO' else 'image'}/{file_ext}"
                        )
            else:
                st.warning("⚠️ Visual file not found. Check the console logs.")
            
            # Technical specifications (optional)
            if show_tech:
                with st.expander("🔧 Technical Specifications (Agent Research)"):
                    st.markdown('<div class="tech-specs">', unsafe_allow_html=True)
                    st.markdown(result.get("technical_specs", "No technical data available"))
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Add to history
            st.session_state.history.append({
                "query": user_query,
                "story": story_text,
                "visual": visual_path,
                "format": format_used,
                "timestamp": time.time()
            })
        
        else:
            st.error("❌ Workflow failed. Please check the console for details.")
            st.code(result.get("error", "Unknown error"))
    
    # History section
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📚 Session History")
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"🚗 {item['query'][:60]}... ({item['format']})"):
                st.markdown(f"**Story**: {item['story'][:200]}...")
                if item['visual'] and os.path.exists(item['visual']):
                    if item['format'] == "VIDEO":
                        st.video(item['visual'])
                    else:
                        st.image(item['visual'], width=300)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">Powered by CrewAI • Google Gemini • Stable Diffusion • Qdrant</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

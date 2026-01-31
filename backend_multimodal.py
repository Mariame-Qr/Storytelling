"""
AutoStory Multimodal Backend - Advanced Agentic Orchestration System
CrewAI-based intelligent automotive storytelling with multimodal generation
"""

import os
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field

# Audio processing
from gtts import gTTS
import speech_recognition as sr
import replicate

# Video processing - moviepy 2.x uses different import structure
try:
    from moviepy import VideoFileClip, AudioFileClip, ImageClip
except ImportError:
    # Fallback for moviepy 1.x
    from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip

# Load environment variables
load_dotenv()

# Initialize LLM - Use OpenAI GPT-4o-mini (most reliable)
llm = LLM(
    model="gpt-4o-mini",
    temperature=0.7
)

# Directory setup
OUTPUT_DIR = Path("generated_outputs")
AUDIO_DIR = Path("generated_audio")
VIDEO_LIBRARY = Path(os.getenv("VIDEO_LIBRARY_PATH", "./video_library"))

for dir_path in [OUTPUT_DIR, AUDIO_DIR, VIDEO_LIBRARY]:
    dir_path.mkdir(exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_visual_prompt_from_story(story_text: str, user_query: str) -> str:
    """
    Génère un prompt visuel détaillé et pertinent basé sur le contenu de l'histoire
    
    Args:
        story_text: Le texte de l'histoire générée
        user_query: La requête originale de l'utilisateur
        
    Returns:
        Un prompt optimisé pour la génération d'image/vidéo
    """
    # Extraire les mots-clés techniques de l'histoire
    technical_keywords = []
    story_lower = story_text.lower()
    
    # Mots-clés automobiles à détecter
    automotive_terms = {
        "awd": "all-wheel drive system with visible torque distribution",
        "all-wheel drive": "all-wheel drive AWD system showing power distribution",
        "torque": "mechanical torque transfer system with rotating components",
        "differential": "automotive differential mechanism in detail",
        "engine": "car engine with visible mechanical parts",
        "transmission": "automotive transmission system with gears",
        "brake": "automotive braking system with calipers and rotors",
        "suspension": "car suspension system with shock absorbers",
        "turbo": "turbocharger with visible turbine blades",
        "hybrid": "hybrid powertrain with electric motor and combustion engine",
        "electric": "electric vehicle powertrain with battery pack",
        "steering": "automotive steering system mechanism",
        "axle": "vehicle axle assembly with drivetrain components",
        "clutch": "automotive clutch mechanism in detail",
        "gearbox": "manual or automatic gearbox internal mechanism"
    }
    
    # Détecter les termes pertinents dans l'histoire
    for term, description in automotive_terms.items():
        if term in story_lower:
            technical_keywords.append(description)
    
    # Si aucun terme spécifique détecté, utiliser la requête utilisateur
    if not technical_keywords:
        base_prompt = f"professional automotive photograph showing {user_query}"
    else:
        # Utiliser le premier terme technique trouvé comme base
        base_prompt = f"professional automotive photograph showing {technical_keywords[0]}"
    
    # Enrichir avec des descripteurs visuels professionnels
    visual_enhancements = [
        "cinematic lighting",
        "8K ultra high definition",
        "professional automotive photography",
        "detailed mechanical components",
        "realistic materials and textures",
        "studio lighting setup",
        "shallow depth of field",
        "automotive technical illustration style"
    ]
    
    # Construire le prompt final
    full_prompt = f"{base_prompt}, {', '.join(visual_enhancements[:4])}"
    
    return full_prompt


# ============================================================================
# CUSTOM CREWAI TOOLS
# ============================================================================

class SearchManualInput(BaseModel):
    """Input schema for SearchManualTool"""
    query: str = Field(..., description="The search query to find relevant technical specifications")


class SearchManualTool(BaseTool):
    """RAG-based tool to search automotive technical documentation using Qdrant"""
    name: str = "Search Car Manual"
    description: str = """
    Searches the vehicle technical manual for accurate specifications and data.
    Use this tool to retrieve factual information about vehicle systems, features, and performance.
    Input should be a specific query about a vehicle feature or system.
    Returns relevant technical documentation excerpts.
    """
    args_schema: type[BaseModel] = SearchManualInput
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_client', QdrantClient(path="./qdrant_db"))
        object.__setattr__(self, '_embeddings', GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        ))
        object.__setattr__(self, '_collection_name', "car_specs")
    
    def _run(self, query: str) -> str:
        """Execute RAG search against Qdrant vector database"""
        try:
            query_vector = self._embeddings.embed_query(query)
            
            # Use query_points instead of search (Qdrant v1.16+)
            search_results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=3,
                score_threshold=0.3
            )
            
            # Handle both old and new response formats
            points = search_results.points if hasattr(search_results, 'points') else search_results
            
            if not points:
                return "No relevant technical information found in the manual."
            
            formatted_results = []
            for idx, result in enumerate(points, 1):
                score = result.score if hasattr(result, 'score') else 0.0
                payload = result.payload if hasattr(result, 'payload') else {}
                formatted_results.append(
                    f"--- Result {idx} (Relevance: {score:.2f}) ---\n"
                    f"Topic: {payload.get('title', 'Unknown')}\n"
                    f"Details: {payload.get('content', 'No content')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching manual: {str(e)}"


class SearchVideoLibraryInput(BaseModel):
    """Input schema for SearchVideoLibraryTool"""
    feature_name: str = Field(..., description="Name of automotive feature to search for")


class SearchVideoLibraryTool(BaseTool):
    """Tool to search for existing automotive videos in the library"""
    name: str = "Search Video Library"
    description: str = """
    Searches the video library for existing automotive videos matching the requested feature.
    Use this tool to find reusable video content before generating new videos.
    Input should be the name of the automotive feature or system.
    Returns path to video file if found, or None if no match.
    """
    args_schema: type[BaseModel] = SearchVideoLibraryInput
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, feature_name: str) -> str:
        """Search for existing videos matching the feature"""
        try:
            print(f"\n🔍 Searching video library for: {feature_name}")
            
            # Search in video library directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            feature_keywords = feature_name.lower().split()
            
            for video_file in VIDEO_LIBRARY.glob("*"):
                if video_file.suffix.lower() in video_extensions:
                    video_name = video_file.stem.lower()
                    # Check if any keyword matches
                    if any(keyword in video_name for keyword in feature_keywords):
                        print(f"✓ Found matching video: {video_file.name}")
                        return str(video_file.absolute())
            
            print("⚠ No matching video found in library")
            return "No matching video found"
            
        except Exception as e:
            return f"Error searching video library: {str(e)}"


class GenerateNarrationInput(BaseModel):
    """Input schema for GenerateNarrationTool"""
    text: str = Field(..., description="Text to convert to speech narration")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'fr', 'es')")


class GenerateNarrationTool(BaseTool):
    """Tool for text-to-speech audio generation"""
    name: str = "Generate Audio Narration"
    description: str = """
    Converts text into natural-sounding speech narration.
    Use this tool to create audio narration for automotive stories.
    Input should be the story text to narrate.
    Returns path to generated audio file.
    """
    args_schema: type[BaseModel] = GenerateNarrationInput
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, text: str, language: str = "en") -> str:
        """Generate audio narration using gTTS"""
        try:
            print(f"\n🎤 Generating audio narration...")
            print(f"Text length: {len(text)} characters")
            
            # Generate audio
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to file
            timestamp = int(time.time())
            audio_path = AUDIO_DIR / f"narration_{timestamp}.mp3"
            tts.save(str(audio_path))
            
            print(f"✓ Audio generated: {audio_path}")
            return str(audio_path.absolute())
            
        except Exception as e:
            return f"Error generating narration: {str(e)}"


class GenerateVideoInput(BaseModel):
    """Input schema for GenerateVideoWithReplicateTool"""
    prompt: str = Field(..., description="Detailed prompt for video generation")


class GenerateVideoWithReplicateTool(BaseTool):
    """Tool for generating videos directly using Replicate API with Stable Video Diffusion"""
    name: str = "Generate Video with Replicate"
    description: str = """
    Generates high-quality cinematic automotive videos using Replicate API.
    Use this tool to create dynamic video content showing automotive systems in action.
    Input should be a detailed, descriptive prompt for video generation.
    Returns path to generated video file.
    """
    args_schema: type[BaseModel] = GenerateVideoInput
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, prompt: str) -> str:
        """Generate video directly using Replicate API"""
        try:
            print(f"\n� Generating video with Replicate API...")
            print(f"Prompt: {prompt[:100]}...")
            
            # Check if API token exists
            api_token = os.getenv('REPLICATE_API_TOKEN')
            if not api_token:
                print("⚠ No Replicate API token found, creating placeholder video")
                return self._create_placeholder_video(prompt)
            
            # Set token for replicate client
            os.environ['REPLICATE_API_TOKEN'] = api_token
            
            # Enhanced automotive-focused prompt for text-to-video
            enhanced_prompt = f"Professional automotive cinematography: {prompt}. Cinematic camera movement, smooth motion, 4K quality, realistic lighting, detailed mechanical parts, professional video production"
            
            print("🎥 Calling Replicate API...")
            print("⏳ This may take 30-90 seconds...")
            
            # Strategy: Generate image first, then convert to video
            # Step 1: Generate image with SDXL
            print("📸 Step 1: Generating image with SDXL...")
            image_output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={"prompt": enhanced_prompt}
            )
            
            # Get image URL
            if not image_output:
                print("⚠ No image generated")
                return self._create_placeholder_video(prompt)
            
            image_url = image_output[0] if isinstance(image_output, list) else image_output
            print(f"✓ Image generated: {str(image_url)[:60]}...")
            
            # Step 2: Convert image to video with Stable Video Diffusion
            print("🎬 Step 2: Converting image to video...")
            output = replicate.run(
                "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                input={
                    "input_image": image_url,
                    "video_length": "14_frames_with_svd",
                    "sizing_strategy": "maintain_aspect_ratio",
                    "frames_per_second": 6,
                    "motion_bucket_id": 127,
                    "cond_aug": 0.02
                }
            )
            
            # Download video from Replicate
            if output:
                video_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) else str(output))
                
                print(f"⬇️ Downloading video from Replicate...")
                response = requests.get(video_url, timeout=120)
                
                if response.status_code == 200:
                    timestamp = int(time.time())
                    filepath = OUTPUT_DIR / f"replicate_video_{timestamp}.mp4"
                    
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    print(f"✓ Video generated successfully: {filepath}")
                    return str(filepath.absolute())
                else:
                    print(f"⚠ Download failed with status {response.status_code}")
                    return self._create_placeholder_video(prompt)
            else:
                print("⚠ No output received from Replicate")
                return self._create_placeholder_video(prompt)
            
        except Exception as e:
            print(f"⚠ Error generating video with Replicate: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_placeholder_video(prompt)
    
    def _create_placeholder_video(self, prompt: str) -> str:
        """Create a professional placeholder video from a static image"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            print("🎨 Creating professional placeholder video...")
            
            # Create high-quality placeholder image
            img = Image.new('RGB', (1920, 1080), color=(15, 20, 35))
            draw = ImageDraw.Draw(img)
            
            # Add gradient effect (simple simulation)
            for y in range(1080):
                gradient_color = int(15 + (y / 1080) * 20)
                draw.line([(0, y), (1920, y)], fill=(gradient_color, gradient_color + 5, gradient_color + 15))
            
            # Try to use fonts
            try:
                title_font = ImageFont.truetype("arial.ttf", 80)
                subtitle_font = ImageFont.truetype("arial.ttf", 40)
                small_font = ImageFont.truetype("arial.ttf", 28)
            except:
                title_font = ImageFont.load_default()
                subtitle_font = title_font
                small_font = title_font
            
            # Draw branded content
            draw.text((960, 300), "🚗 AutoStory", fill=(80, 150, 255), font=title_font, anchor="mm")
            draw.text((960, 420), "Automotive AI Storytelling", fill=(150, 180, 230), font=subtitle_font, anchor="mm")
            
            # Draw user query info
            draw.text((960, 520), "Requête:", fill=(120, 140, 180), font=small_font, anchor="mm")
            prompt_lines = [prompt[i:i+70] for i in range(0, min(len(prompt), 140), 70)]
            y_pos = 580
            for line in prompt_lines:
                draw.text((960, y_pos), line, fill=(200, 210, 220), font=small_font, anchor="mm")
                y_pos += 40
            
            # Status message
            draw.text((960, 750), "⚠ Replicate API - Crédit requis", fill=(255, 120, 100), font=small_font, anchor="mm")
            draw.text((960, 800), "Ajoutez $5 de crédit sur replicate.com/account/billing", fill=(180, 180, 180), font=small_font, anchor="mm")
            
            # Footer
            draw.text((960, 950), "Placeholder généré localement • Version démo", fill=(100, 120, 140), font=small_font, anchor="mm")
            
            # Save image first
            timestamp = int(time.time())
            img_path = OUTPUT_DIR / f"placeholder_img_{timestamp}.png"
            img.save(img_path, quality=95)
            
            # Convert to video using moviepy with better quality
            video_path = OUTPUT_DIR / f"placeholder_video_{timestamp}.mp4"
            # moviepy 2.x syntax - duration is first parameter
            img_clip = ImageClip(str(img_path), duration=8)
            img_clip.write_videofile(
                str(video_path),
                fps=24,
                codec='libx264'
            )
            img_clip.close()
            
            # Clean up temporary image
            img_path.unlink()
            
            print(f"✓ Professional placeholder video created: {video_path}")
            print(f"ℹ️  Pour générer de vraies vidéos, ajoutez du crédit: https://replicate.com/account/billing")
            return str(video_path.absolute())
            
        except Exception as e:
            print(f"⚠ Could not create placeholder video: {str(e)}")
            return None


class MergeAudioVideoInput(BaseModel):
    """Input schema for MergeAudioVideoTool"""
    video_path: str = Field(..., description="Path to video file")
    audio_path: str = Field(..., description="Path to audio file")


class MergeAudioVideoTool(BaseTool):
    """Tool for merging audio narration with video"""
    name: str = "Merge Audio and Video"
    description: str = """
    Combines audio narration with video to create a final narrated video.
    Use this tool when you have both audio narration and video content.
    Returns path to final merged video file.
    """
    args_schema: type[BaseModel] = MergeAudioVideoInput
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, video_path: str, audio_path: str) -> str:
        """Merge audio and video using moviepy"""
        try:
            print(f"\n🎬 Merging audio and video...")
            print(f"Video: {video_path}")
            print(f"Audio: {audio_path}")
            
            # Validate paths
            if not video_path or not audio_path:
                print("⚠ Missing video or audio path")
                return None
            if not Path(video_path).exists() or not Path(audio_path).exists():
                print(f"⚠ File not found - Video exists: {Path(video_path).exists()}, Audio exists: {Path(audio_path).exists()}")
                return None
            
            # Load video and audio
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            print(f"📊 Video duration: {video_clip.duration}s, Audio duration: {audio_clip.duration}s")
            
            # Simple approach: just add audio to video without complex looping
            # If audio is longer than video, video will loop automatically
            final_clip = video_clip.with_audio(audio_clip)
            
            # Export
            timestamp = int(time.time())
            output_path = OUTPUT_DIR / f"narrated_video_{timestamp}.mp4"
            
            print(f"🎬 Exporting final video...")
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=24
            )
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            print(f"✓ Merged video created: {output_path}")
            return str(output_path.absolute())
            
        except Exception as e:
            print(f"⚠ Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class CreateVideoFromImageInput(BaseModel):
    """Input schema for CreateVideoFromImageTool"""
    image_path: str = Field(..., description="Path to image file")
    duration: int = Field(default=5, description="Duration in seconds")


class CreateVideoFromImageTool(BaseTool):
    """Tool for creating a video from a static image"""
    name: str = "Create Video from Image"
    description: str = """
    Creates a video clip from a static image with specified duration.
    Use this tool when you need to convert an image into a video format.
    Returns path to generated video file.
    """
    args_schema: type[BaseModel] = CreateVideoFromImageInput
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, image_path: str, duration: int = 5) -> str:
        """Create video from image using moviepy"""
        try:
            print(f"\n🎥 Creating video from image...")
            print(f"Image: {image_path}")
            print(f"Duration: {duration}s")
            
            # Validate image path
            if not image_path or image_path.startswith("Error") or not Path(image_path).exists():
                print(f"⚠ Invalid image path: {image_path}")
                return None
            
            # Create image clip
            image_clip = ImageClip(image_path).set_duration(duration)
            
            # Export
            timestamp = int(time.time())
            output_path = OUTPUT_DIR / f"image_video_{timestamp}.mp4"
            
            image_clip.write_videofile(
                str(output_path),
                fps=24,
                codec='libx264',
                verbose=False,
                logger=None
            )
            
            image_clip.close()
            
            print(f"✓ Video created: {output_path}")
            return str(output_path.absolute())
            
        except Exception as e:
            return f"Error creating video from image: {str(e)}"


# ============================================================================
# CREWAI AGENTS
# ============================================================================

def create_orchestrator_agent() -> Agent:
    """
    Master orchestrator that decides the multimodal generation strategy
    """
    return Agent(
        role="Multimodal AI Orchestrator",
        goal="""
        Analyze user requests and decide the optimal multimodal generation strategy.
        Make intelligent decisions about:
        - Whether to use existing videos or generate new content
        - Whether to generate images, videos, or text only
        - Whether audio narration is needed
        - The complete execution pipeline
        
        Your decisions must be explicit, traceable, and logged.
        """,
        backstory="""
        You are the master orchestrator of the AutoStory multimodal system.
        You have complete understanding of:
        - User intent classification
        - Content reusability assessment
        - Multimodal generation capabilities
        - Cost-benefit analysis of generation vs reuse
        
        You follow this decision framework:
        
        DECISION PIPELINE:
        
        1. ANALYZE USER INTENT
           - Extract requested automotive feature
           - Determine if dynamic (motion) or static
           - Assess complexity and visual requirements
        
        2. CHECK EXISTING RESOURCES
           - Search video library first
           - If suitable video found → REUSE PATH
           - If no video found → GENERATION PATH
        
        3. DECIDE MODALITIES
           - TEXT: Always generate story
           - IMAGE: For static features or fallback
           - VIDEO: For dynamic mechanisms
           - AUDIO: For narration (user preference or default)
        
        4. EXECUTION STRATEGY
           CASE A - EXISTING VIDEO + NARRATION:
             1. Use found video
             2. Generate story text
             3. Generate audio narration
             4. Merge audio + video
             5. Output final narrated MP4
           
           CASE B - GENERATE NEW CONTENT:
             1. Generate story text
             2. Generate image
             3. Convert image to video OR generate video
             4. Generate audio narration
             5. Merge audio + video
             6. Output final narrated MP4
           
           CASE C - TEXT ONLY:
             1. Generate story text
             2. Optionally generate audio
             3. Output text + audio
        
        You ALWAYS log your decision reasoning clearly.
        """,
        tools=[SearchVideoLibraryTool()],
        llm=llm,
        verbose=True,
        allow_delegation=True
    )


def create_technical_expert_agent() -> Agent:
    """Agent responsible for retrieving accurate technical specifications"""
    return Agent(
        role="Automotive Technical Engineer AI",
        goal="""
        Retrieve precise, factual technical specifications about vehicle features and systems.
        Ensure all information is accurate and grounded in the technical documentation.
        Never hallucinate or invent specifications.
        """,
        backstory="""
        You are an expert automotive engineer with deep knowledge of vehicle systems.
        You have access to the complete technical manual and always verify facts before responding.
        Your expertise covers: powertrains, safety systems, ADAS, chassis dynamics, and electronic systems.
        You prioritize accuracy over creativity.
        """,
        tools=[SearchManualTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_storyteller_agent() -> Agent:
    """Agent responsible for transforming technical specs into engaging narratives"""
    return Agent(
        role="Automotive Storytelling AI",
        goal="""
        Transform technical specifications into compelling, accessible narratives.
        Create stories optimized for audio narration.
        Maintain complete technical accuracy while making content engaging and emotional.
        Connect features to real-world benefits and user experiences.
        """,
        backstory="""
        You are a master storyteller specializing in automotive content.
        You excel at translating complex engineering into stories that resonate with people.
        Your narratives are:
        - Accurate and factual
        - Emotionally engaging
        - Suitable for voice narration
        - Visual and descriptive
        - Consumer-friendly language
        
        You craft stories in 150-250 words that flow naturally when spoken.
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_audio_agent() -> Agent:
    """Agent responsible for audio generation and processing"""
    return Agent(
        role="Audio & Voice AI Agent",
        goal="""
        Generate high-quality audio narration for automotive stories.
        Convert text to natural-sounding speech.
        Handle voice processing and audio quality.
        """,
        backstory="""
        You are an audio engineering specialist focused on voice narration.
        You understand:
        - Text-to-speech optimization
        - Audio quality standards
        - Narration pacing and clarity
        
        You generate clear, professional audio narration that enhances the storytelling experience.
        """,
        tools=[GenerateNarrationTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_creative_director_agent() -> Agent:
    """Agent responsible for video content generation"""
    return Agent(
        role="Cinematic AI Director",
        goal="""
        Generate high-quality cinematic videos for automotive features using Replicate API.
        Create dynamic video content that accurately represents automotive systems in motion.
        Ensure professional cinematography, visual quality and technical accuracy.
        """,
        backstory="""
        You are an expert cinematic director specializing in automotive video production.
        You create:
        - Professional cinematic automotive videos using Replicate AI
        - Dynamic shots showing mechanical systems in action
        - High-quality video content with smooth camera movements
        
        You ensure all videos are:
        - Technically accurate representations of automotive systems
        - Cinematically compelling with professional camera work
        - Optimized for viewer engagement and clarity
        - Appropriate for the automotive feature being demonstrated
        """,
        tools=[GenerateVideoWithReplicateTool(), CreateVideoFromImageTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_video_assembly_agent() -> Agent:
    """Agent responsible for video and audio assembly"""
    return Agent(
        role="Multimodal Assembly Engineer",
        goal="""
        Assemble final multimodal outputs by merging audio and video.
        Create professional narrated videos.
        Ensure synchronization and quality.
        """,
        backstory="""
        You are a video production specialist who assembles final outputs.
        You:
        - Merge audio narration with video content
        - Ensure proper synchronization
        - Maintain quality standards
        - Handle format conversions
        
        You produce professional-quality narrated videos ready for distribution.
        """,
        tools=[MergeAudioVideoTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

def run_autostory_multimodal_crew(user_query: str, format_preference: str = "full") -> Dict[str, Any]:
    """
    Execute the AutoStory Multimodal CrewAI workflow
    
    Args:
        user_query: User's request for automotive feature visualization
        format_preference: Preferred output format ("full", "audio", "image", "video")
        
    Returns:
        Dictionary containing story, visual path, audio path, and metadata
    """
    print("\n" + "=" * 80)
    print("🚗 AUTOSTORY MULTIMODAL AGENTIC WORKFLOW INITIATED")
    print("=" * 80)
    print(f"User Query: {user_query}")
    print(f"Format Preference: {format_preference.upper()}")
    print("=" * 80 + "\n")
    
    # Determine which modalities to generate based on preference
    audio_enabled = format_preference in ["full", "audio"]
    image_enabled = format_preference in ["full", "image"]
    video_enabled = format_preference in ["full", "video"]
    
    # Create agents
    orchestrator = create_orchestrator_agent()
    technical_expert = create_technical_expert_agent()
    storyteller = create_storyteller_agent()
    audio_agent = create_audio_agent()
    creative_director = create_creative_director_agent()
    video_assembler = create_video_assembly_agent()
    
    # TASK 1: Orchestration Decision
    orchestration_task = Task(
        description=f"""
        Analyze the user request and decide the multimodal generation strategy.
        
        USER REQUEST: {user_query}
        FORMAT PREFERENCE: {format_preference}
        AUDIO ENABLED: {audio_enabled}
        IMAGE ENABLED: {image_enabled}
        VIDEO ENABLED: {video_enabled}
        
        Follow this decision pipeline:
        
        STEP 1: ANALYZE INTENT
        - Extract automotive feature name
        - Determine if dynamic (moving parts) or static
        - Assess visual complexity
        
        STEP 2: CHECK VIDEO LIBRARY
        - Use "Search Video Library" tool to find existing videos
        - If found: note video path for reuse
        - If not found: mark for generation
        
        STEP 3: DECIDE STRATEGY
        Output a JSON decision with:
        {{
            "intent_classification": "DYNAMIC_MECHANISM | STATIC_FEATURE | INFORMATIONAL",
            "feature_name": "extracted feature name",
            "video_found": true/false,
            "video_path": "path or null",
            "strategy": "REUSE_VIDEO | GENERATE_NEW | TEXT_ONLY",
            "modalities": ["TEXT", "IMAGE", "VIDEO", "AUDIO"],
            "reasoning": "explanation of decision"
        }}
        
        CRITICAL: Output ONLY the JSON decision.
        """,
        expected_output="JSON decision object with complete strategy",
        agent=orchestrator
    )
    
    # TASK 2: Technical Research
    research_task = Task(
        description=f"""
        Research and retrieve accurate technical specifications.
        
        USER REQUEST: {user_query}
        
        Use the Search Car Manual tool to find:
        - Technical specifications
        - Performance data
        - Component details
        - Operational characteristics
        
        Output a comprehensive technical summary.
        """,
        expected_output="Detailed technical summary with specifications",
        agent=technical_expert
    )
    
    # TASK 3: Story Generation
    story_task = Task(
        description=f"""
        Transform technical specifications into an engaging automotive story.
        
        ORIGINAL REQUEST: {user_query}
        
        Create a narrative optimized for audio narration:
        - 150-250 words
        - Natural speaking flow
        - Emotionally engaging
        - Technically accurate
        - Consumer-friendly language
        
        Include:
        - Opening scene
        - How it works (simplified)
        - Real-world benefits
        - Visual elements description
        """,
        expected_output="Engaging automotive story (150-250 words) suitable for narration",
        agent=storyteller
    )
    
    # Create crew with sequential process
    crew = Crew(
        agents=[orchestrator, technical_expert, storyteller, audio_agent, creative_director, video_assembler],
        tasks=[orchestration_task, research_task, story_task],
        process=Process.sequential,
        verbose=True
    )
    
    print("\n🎯 Starting multimodal workflow execution...\n")
    start_time = time.time()
    
    result = crew.kickoff()
    
    # Parse orchestration decision
    try:
        decision_output = str(orchestration_task.output.raw)
        print(f"\n📋 ORCHESTRATION DECISION:\n{decision_output}\n")
        
        # Try to parse JSON
        if "{" in decision_output:
            json_str = decision_output[decision_output.find("{"):decision_output.rfind("}")+1]
            decision = json.loads(json_str)
        else:
            decision = {
                "strategy": "GENERATE_NEW",
                "modalities": ["TEXT", "IMAGE", "AUDIO"],
                "video_found": False
            }
    except:
        decision = {
            "strategy": "GENERATE_NEW",
            "modalities": ["TEXT", "IMAGE", "AUDIO"],
            "video_found": False
        }
    
    # Extract story
    story_text = str(story_task.output.raw)
    
    # EXECUTE STRATEGY
    final_outputs = {
        "success": True,
        "story": story_text,
        "decision": decision,
        "strategy": decision.get("strategy", "GENERATE_NEW"),
        "outputs": {}
    }
    
    # GENERATE AUDIO (if enabled)
    audio_path = None
    if audio_enabled:
        narration_tool = GenerateNarrationTool()
        audio_path = narration_tool._run(story_text, language="en")
        if audio_path and not audio_path.startswith("Error"):
            final_outputs["outputs"]["audio"] = audio_path
    
    # Handle video/image generation based on strategy and preference
    if decision.get("video_found") and decision.get("video_path") and video_enabled:
        # REUSE EXISTING VIDEO
        video_path = decision["video_path"]
        final_outputs["outputs"]["video"] = video_path
        
        # Merge with audio if available
        if audio_path and audio_path != "Error generating narration:":
            merge_tool = MergeAudioVideoTool()
            final_video = merge_tool._run(video_path, audio_path)
            final_outputs["outputs"]["final_video"] = final_video
    
    else:
        # GENERATE NEW CONTENT WITH REPLICATE
        
        # Generate video directly with Replicate if needed
        if video_enabled:
            # Créer un prompt visuel intelligent basé sur l'histoire
            visual_prompt = generate_visual_prompt_from_story(story_text, user_query)
            print(f"\n🎨 Visual prompt généré: {visual_prompt[:120]}...")
            
            video_tool = GenerateVideoWithReplicateTool()
            video_path = video_tool._run(visual_prompt)
            
            if video_path and os.path.exists(video_path):
                final_outputs["outputs"]["video"] = video_path
                
                # Merge with audio narration
                if audio_path and not audio_path.startswith("Error"):
                    merge_tool = MergeAudioVideoTool()
                    final_video = merge_tool._run(video_path, audio_path)
                    if final_video and os.path.exists(final_video):
                        final_outputs["outputs"]["final_video"] = final_video
        
        # If only image is requested (no video), create a placeholder
        elif image_enabled:
            print("⚠️ Image-only mode not supported with Replicate. Generating placeholder video instead.")
            video_tool = GenerateVideoWithReplicateTool()
            video_path = video_tool._create_placeholder_video(user_query)
            if video_path:
                final_outputs["outputs"]["video"] = video_path
    
    execution_time = time.time() - start_time
    final_outputs["execution_time"] = execution_time
    
    print("\n" + "=" * 80)
    print("✅ MULTIMODAL WORKFLOW COMPLETED")
    print("=" * 80)
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Strategy: {final_outputs['strategy']}")
    print(f"Outputs: {list(final_outputs['outputs'].keys())}")
    print("=" * 80 + "\n")
    
    return final_outputs


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    test_query = "Explain how the all-wheel drive system distributes torque"
    
    # Try full workflow first, fallback to audio-only if it fails
    try:
        result = run_autostory_multimodal_crew(test_query, format_preference="audio")
    except Exception as e:
        print(f"\n⚠️ Full workflow failed: {e}")
        print("\n🎤 Generating audio-only fallback...\n")
        
        # Fallback: Generate audio directly
        story_text = f"""The all-wheel drive system is an advanced automotive technology that intelligently distributes engine torque between the front and rear axles to optimize traction and handling. 

When the vehicle accelerates, sensors continuously monitor wheel speed and traction conditions. The system uses a center differential or electronic coupling to split power between the axles. Under normal driving, torque is distributed evenly, typically 50-50 front to rear.

However, when sensors detect wheel slip on one axle, the system rapidly redirects more torque to the axle with better grip. This happens in milliseconds through electronic control units that manage clutch packs or electromagnetic couplings.

In challenging conditions like snow or mud, the all-wheel drive system can send up to 100 percent of available torque to the wheels with the most traction, ensuring maximum forward momentum and vehicle stability."""

        narration_tool = GenerateNarrationTool()
        audio_path = narration_tool._run(story_text, language="en")
        
        result = {
            "success": True,
            "story": story_text,
            "outputs": {"audio": audio_path},
            "strategy": "AUDIO_FALLBACK",
            "execution_time": 0
        }
    
    print("\n" + "=" * 80)
    print("FINAL OUTPUTS")
    print("=" * 80)
    print(f"\n📖 STORY:\n{result.get('story', 'No story generated')}\n")
    print("-" * 80)
    for key, value in result.items():
        if key not in ["technical_specs", "story", "decision"]:
            print(f"{key}: {value}")
    print("=" * 80)

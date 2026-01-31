"""
AutoStory Backend - Agentic Orchestration System
CrewAI-based intelligent automotive storytelling with visual generation
"""

import os
import io
import time
import base64
import requests
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize LLM using Groq (free, fast alternative)
# Using llama-3.3-70b-versatile (current supported model)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.7
)


# ============================================================================
# CUSTOM CREWAI TOOLS
# ============================================================================

class SearchManualInput(BaseModel):
    """Input schema for SearchManualTool"""
    query: str = Field(..., description="The search query to find relevant technical specifications")


class SearchManualTool(BaseTool):
    """
    RAG-based tool to search automotive technical documentation using Qdrant
    """
    name: str = "Search Car Manual"
    description: str = """
    Searches the vehicle technical manual for accurate specifications and data.
    Use this tool to retrieve factual information about vehicle systems, features, and performance.
    Input should be a specific query about a vehicle feature or system.
    Returns relevant technical documentation excerpts.
    """
    args_schema: type[BaseModel] = SearchManualInput
    
    # Use model_config for Pydantic v2
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize as instance attributes (not Pydantic fields)
        object.__setattr__(self, '_client', QdrantClient(path="./qdrant_db"))
        object.__setattr__(self, '_embeddings', GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        ))
        object.__setattr__(self, '_collection_name', "car_specs")
    
    def _run(self, query: str) -> str:
        """
        Execute RAG search against Qdrant vector database
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results with relevant technical information
        """
        try:
            # Generate query embedding
            query_vector = self._embeddings.embed_query(query)
            
            # Search Qdrant
            search_results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=3,  # Top 3 most relevant results
                score_threshold=0.3
            )
            
            if not search_results:
                return f"No relevant technical information found for: {query}"
            
            # Format results
            formatted_results = []
            for idx, result in enumerate(search_results, 1):
                title = result.payload.get("title", "Unknown")
                content = result.payload.get("content", "")
                score = result.score
                
                formatted_results.append(
                    f"--- Result {idx} (Relevance: {score:.2f}) ---\n"
                    f"Topic: {title}\n"
                    f"Details: {content}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching manual: {str(e)}"


class GenerateImageInput(BaseModel):
    """Input schema for SDXLTool"""
    prompt: str = Field(..., description="Detailed prompt for image generation")


class SDXLTool(BaseTool):
    """
    Tool for generating high-quality images using Stable Diffusion XL via HuggingFace API
    """
    name: str = "Generate Image with SDXL"
    description: str = """
    Generates a high-quality cinematic image using Stable Diffusion XL.
    Use this tool when the visual output should be a STATIC IMAGE.
    Input should be a detailed, descriptive prompt specifying:
    - What to visualize
    - Visual style (photorealistic, technical diagram, cinematic, etc.)
    - Key elements and composition
    Returns path to generated image file.
    """
    args_schema: type[BaseModel] = GenerateImageInput
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_api_url', "https://router.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0")
        object.__setattr__(self, '_headers', {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"})
        object.__setattr__(self, '_output_dir', "generated_outputs")
        os.makedirs(self._output_dir, exist_ok=True)
    
    def _run(self, prompt: str) -> str:
        """
        Generate image using SDXL via HuggingFace Inference API
        
        Args:
            prompt: Detailed image generation prompt
            
        Returns:
            Path to generated image or error message
        """
        try:
            print(f"\n🎨 Generating image with SDXL...")
            print(f"Prompt: {prompt[:100]}...")
            
            # Enhance prompt with quality boosters
            enhanced_prompt = f"{prompt}, highly detailed, professional photography, 8k uhd, cinematic lighting, sharp focus"
            
            payload = {
                "inputs": enhanced_prompt,
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5
                }
            }
            
            # Make API request with retry logic
            max_retries = 3
            retry_delay = 10
            
            for attempt in range(max_retries):
                response = requests.post(
                    self._api_url,
                    headers=self._headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    # Save image
                    timestamp = int(time.time())
                    filename = f"image_{timestamp}.png"
                    filepath = os.path.join(self._output_dir, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    print(f"✓ Image generated successfully: {filepath}")
                    return filepath
                
                elif response.status_code == 503:
                    # Model loading
                    print(f"Model loading... Retry {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return f"Error: Model unavailable after {max_retries} retries"
                else:
                    error_text = response.text if len(response.text) < 200 else response.text[:200]
                    print(f"⚠ API Error {response.status_code}: {error_text}")
                    return f"Error generating image: {response.status_code} - {error_text}"
            
            return "Error: Failed to generate image after retries"
            
        except Exception as e:
            return f"Error in image generation: {str(e)}"


class GenerateVideoInput(BaseModel):
    """Input schema for SVDTool"""
    prompt: str = Field(..., description="Detailed prompt for video generation")


class SVDTool(BaseTool):
    """
    Tool for generating short videos using Stable Video Diffusion via HuggingFace Space
    """
    name: str = "Generate Video with SVD"
    description: str = """
    Generates a short video (2-4 seconds) using Stable Video Diffusion.
    Use this tool when the visual output should be a DYNAMIC VIDEO showing:
    - Mechanical motion
    - Process sequences
    - Dynamic systems in action
    Input should be a detailed prompt describing the motion and dynamics.
    Returns path to generated video file.
    NOTE: This is experimental and may fail - fallback to image generation if needed.
    """
    args_schema: type[BaseModel] = GenerateVideoInput
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_output_dir', "generated_outputs")
        os.makedirs(self._output_dir, exist_ok=True)
    
    def _run(self, prompt: str) -> str:
        """
        Generate video using Stable Video Diffusion (fallback to image)
        
        Args:
            prompt: Video generation prompt
            
        Returns:
            Path to generated image (video generation requires authentication)
        """
        try:
            print(f"\n🎬 Attempting video generation with SVD...")
            print(f"Prompt: {prompt[:100]}...")
            print("⚠ SVD requires authentication - falling back to image generation...")
            
            # Direct fallback to image generation
            # SVD via HuggingFace Spaces requires authentication
            sdxl = SDXLTool()
            return sdxl._run(prompt)
            
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            print(f"⚠ {error_msg}")
            print("⚠ Falling back to image generation...")
            
            # Fallback to image generation
            sdxl = SDXLTool()
            return sdxl._run(prompt)


# ============================================================================
# CREWAI AGENTS
# ============================================================================

def create_technical_expert_agent() -> Agent:
    """
    Agent responsible for retrieving accurate technical specifications
    """
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
    """
    Agent responsible for transforming technical specs into engaging narratives
    """
    return Agent(
        role="Automotive Storytelling AI",
        goal="""
        Transform technical specifications into compelling, accessible narratives.
        Maintain complete technical accuracy while making content engaging and emotional.
        Connect features to real-world benefits and user experiences.
        """,
        backstory="""
        You are a master storyteller specializing in automotive content.
        You excel at translating complex engineering into stories that resonate with people.
        You understand both the technical details and the emotional impact of automotive features.
        You craft narratives that are: accurate, engaging, visual, and memorable.
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_creative_director_agent() -> Agent:
    """
    Agent responsible for visual format decisions and content generation
    """
    return Agent(
        role="Visual AI Director & Orchestrator",
        goal="""
        Analyze user requests and decide the optimal visual format (IMAGE or VIDEO).
        Generate precise visual prompts and create high-quality visual assets.
        Ensure visuals accurately represent the requested automotive feature.
        """,
        backstory="""
        You are an expert visual director specializing in automotive content.
        
        Your decision-making framework:
        
        STATIC_FEATURE → IMAGE
        - Vehicle design, styling, exterior/interior views
        - Component diagrams, cross-sections
        - Dashboard layouts, control interfaces
        
        DYNAMIC_MECHANISM → VIDEO
        - ABS braking, ESC intervention
        - Airbag deployment sequences
        - Torque distribution, AWD engagement
        - Suspension movement, active damping
        
        EMOTIONAL_EXPERIENCE → VIDEO  
        - Family safety scenarios
        - Emergency response situations
        - Driving confidence moments
        
        You ALWAYS:
        1. Classify the user intent
        2. Explicitly state your format decision (IMAGE or VIDEO)
        3. Explain WHY this format was chosen
        4. Define exactly what visual elements must appear
        5. Generate or fallback gracefully
        
        You prioritize visual accuracy and technical correctness.
        """,
        tools=[SDXLTool(), SVDTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# CREWAI TASKS
# ============================================================================

def create_research_task(agent: Agent, user_query: str) -> Task:
    """
    Task for technical research and specification retrieval
    """
    return Task(
        description=f"""
        Research and retrieve accurate technical specifications for the following request:
        
        USER REQUEST: {user_query}
        
        Your objectives:
        1. Use the Search Car Manual tool to find relevant technical data
        2. Extract precise specifications, numbers, and technical details
        3. Identify key components and systems involved
        4. Ensure all information is factual and verifiable
        
        Output a comprehensive technical summary including:
        - System/feature name
        - Key specifications and performance data
        - Technical components involved
        - Operational characteristics
        """,
        expected_output="""
        A detailed technical summary containing:
        - Accurate specifications with numbers
        - Component descriptions
        - Performance characteristics
        - Relevant technical details from the manual
        """,
        agent=agent
    )


def create_storytelling_task(agent: Agent, user_query: str) -> Task:
    """
    Task for transforming technical specs into narrative
    """
    return Task(
        description=f"""
        Transform the technical specifications into an engaging automotive story.
        
        ORIGINAL REQUEST: {user_query}
        
        Your objectives:
        1. Use the technical research from the previous task
        2. Create a compelling narrative that explains the feature/system
        3. Connect technical details to real-world benefits
        4. Make the content accessible and emotionally resonant
        5. Maintain complete technical accuracy
        
        Story structure:
        - Opening: Set the scene (driving scenario or feature context)
        - Technical explanation: How it works (simplified but accurate)
        - Benefit: Why it matters (safety, performance, comfort)
        - Visual elements: What should be shown in the visual output
        
        Keep the story concise (150-250 words) and visually descriptive.
        """,
        expected_output="""
        An engaging automotive story (150-250 words) that:
        - Accurately describes the technical feature
        - Connects to real-world scenarios
        - Identifies key visual elements for illustration
        - Maintains technical accuracy while being accessible
        """,
        agent=agent
    )


def create_visual_generation_task(agent: Agent, user_query: str) -> Task:
    """
    Task for visual format decision and asset generation
    """
    return Task(
        description=f"""
        Analyze the user request and generate the appropriate visual asset.
        
        ORIGINAL REQUEST: {user_query}
        
        MANDATORY DECISION PROCESS:
        
        Step 1: CLASSIFY THE INTENT
        Determine if the request is about:
        - STATIC_FEATURE (design, layout, component view)
        - DYNAMIC_MECHANISM (moving parts, active systems, processes)
        - EMOTIONAL_EXPERIENCE (safety moments, driving scenarios)
        
        Step 2: DECIDE FORMAT
        - STATIC_FEATURE → IMAGE
        - DYNAMIC_MECHANISM → VIDEO (fallback to IMAGE if fails)
        - EMOTIONAL_EXPERIENCE → VIDEO (fallback to IMAGE if fails)
        
        Step 3: EXPLAIN YOUR DECISION
        Write a clear explanation of:
        - What category this request falls into
        - Why IMAGE or VIDEO is the right choice
        - What specific visual elements must be shown
        
        Step 4: GENERATE VISUAL PROMPT
        Create a detailed prompt specifying:
        - Main subject (exact component/system/scene)
        - Visual style (photorealistic, technical diagram, cinematic)
        - Key elements that must appear
        - Composition and perspective
        - Lighting and atmosphere
        
        Step 5: EXECUTE GENERATION
        - If IMAGE: Use "Generate Image with SDXL" tool
        - If VIDEO: Use "Generate Video with SVD" tool (will auto-fallback to image if fails)
        
        CRITICAL: Your output MUST include:
        1. Intent classification
        2. Format decision (IMAGE or VIDEO)
        3. Decision rationale
        4. The generated visual asset path
        """,
        expected_output="""
        A complete visual generation report containing:
        
        1. INTENT CLASSIFICATION: [STATIC_FEATURE / DYNAMIC_MECHANISM / EMOTIONAL_EXPERIENCE]
        2. FORMAT DECISION: [IMAGE / VIDEO]
        3. DECISION RATIONALE: [Why this format was chosen]
        4. VISUAL PROMPT: [Detailed generation prompt used]
        5. OUTPUT PATH: [Path to generated image or video file]
        """,
        agent=agent
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

class AutoStoryCrewResult(BaseModel):
    """Structured result from AutoStory Crew execution"""
    technical_specs: str
    story: str
    visual_decision: str
    visual_path: str
    format_used: str  # "image" or "video"


def run_autostory_crew(user_query: str) -> Dict[str, Any]:
    """
    Execute the AutoStory CrewAI workflow
    
    Args:
        user_query: User's request for automotive feature visualization
        
    Returns:
        Dictionary containing story, visual path, and metadata
    """
    print("\n" + "=" * 70)
    print("🚗 AUTOSTORY AGENTIC WORKFLOW INITIATED")
    print("=" * 70)
    print(f"User Query: {user_query}")
    print("=" * 70 + "\n")
    
    # Create agents
    technical_expert = create_technical_expert_agent()
    storyteller = create_storyteller_agent()
    creative_director = create_creative_director_agent()
    
    # Create tasks
    research_task = create_research_task(technical_expert, user_query)
    story_task = create_storytelling_task(storyteller, user_query)
    visual_task = create_visual_generation_task(creative_director, user_query)
    
    # Create crew with sequential process
    crew = Crew(
        agents=[technical_expert, storyteller, creative_director],
        tasks=[research_task, story_task, visual_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute crew
    print("\n🎯 Starting sequential task execution...\n")
    start_time = time.time()
    
    result = crew.kickoff()
    
    execution_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ WORKFLOW COMPLETED")
    print("=" * 70)
    print(f"Execution time: {execution_time:.2f} seconds")
    print("=" * 70 + "\n")
    
    # Parse results
    try:
        # Extract task outputs
        technical_output = str(research_task.output.raw)
        story_output = str(story_task.output.raw)
        visual_output = str(visual_task.output.raw)
        
        # Determine format used (parse from visual output)
        format_used = "image"  # default
        if "VIDEO" in visual_output.upper() and not "FALLBACK" in visual_output.upper():
            if "video_" in visual_output and ".mp4" in visual_output:
                format_used = "video"
        
        # Extract visual path
        visual_path = ""
        for line in visual_output.split("\n"):
            if "generated_outputs" in line.lower() or ".png" in line or ".mp4" in line:
                # Extract file path
                import re
                path_match = re.search(r'generated_outputs[/\\][\w_]+\.(png|mp4)', line)
                if path_match:
                    visual_path = path_match.group(0)
                    break
        
        if not visual_path:
            # Fallback: check if file exists
            import glob
            outputs = glob.glob("generated_outputs/*")
            if outputs:
                visual_path = max(outputs, key=os.path.getctime)  # Most recent
        
        return {
            "success": True,
            "technical_specs": technical_output,
            "story": story_output,
            "visual_decision": visual_output,
            "visual_path": visual_path,
            "format_used": format_used,
            "execution_time": execution_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error parsing crew output: {str(e)}",
            "raw_output": str(result)
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the system
    test_query = "Show me how the ABS braking system works during emergency braking"
    
    result = run_autostory_crew(test_query)
    
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    for key, value in result.items():
        if key not in ["technical_specs", "story", "visual_decision"]:
            print(f"{key}: {value}")
    print("=" * 70)

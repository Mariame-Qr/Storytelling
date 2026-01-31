# 🚗 AutoStory - Multimodal AI Automotive Storytelling

An intelligent multimodal AI application that transforms automotive technical queries into immersive narrated video experiences using CrewAI, Replicate API, and RAG technology.

## 🎯 Project Overview

AutoStory is an advanced agentic AI system that creates professional automotive storytelling content. The application:

- **Analyzes** user queries about automotive features
- **Generates** engaging narrative stories with technical accuracy
- **Creates** cinematic videos directly using Replicate API
- **Produces** professional narrated videos with synchronized audio

### Key Innovation: Direct Video Generation Workflow

AutoStory uses a streamlined pipeline with Replicate's Stable Video Diffusion for professional automotive cinematography:

```
User Query → Orchestrator → Storyteller → Audio Narration
                                ↓
                        Replicate API (SDXL + SVD)
                                ↓
                        Cinematic Video (20-25s)
                                ↓
                        Merge Audio + Video
                                ↓
                        Final Narrated MP4
```

## 🏗️ Architecture

### Agentic System (CrewAI)

**Six Specialized AI Agents**:

1. **Multimodal AI Orchestrator**
   - Coordinates the complete workflow
   - Manages format preferences (audio/video/full)
   - Ensures smooth pipeline execution

2. **Automotive Technical Engineer AI**
   - Retrieves precise specifications from RAG knowledge base
   - Ensures factual accuracy using Qdrant vector search
   - Prevents hallucinations with grounded data

3. **Automotive Storytelling AI**
   - Transforms technical specs into engaging narratives (150-250 words)
   - Optimized for audio narration
   - Maintains technical accuracy with emotional connection

4. **Audio & Voice AI Agent**
   - Generates high-quality audio narration using gTTS
   - Converts story text to natural-sounding speech
   - Professional voice quality

5. **Cinematic AI Director**
   - Generates professional automotive videos using Replicate API
   - Intelligent prompt enhancement from story content
   - Creates cinematic shots with smooth camera movements

6. **Multimodal Assembly Engineer**
   - Merges audio narration with video content
   - Ensures proper synchronization
   - Produces final narrated MP4 files

### Tech Stack

- **Orchestration**: CrewAI 0.86.0
- **LLM**: OpenAI GPT-4o-mini (with Gemini fallback)
- **Vector DB**: Qdrant 1.16.2 (local persistent mode)
- **Embeddings**: Google Generative AI Embeddings
- **Video Gen**: Replicate API (Stable Diffusion XL + Stable Video Diffusion)
- **Audio Gen**: gTTS (Google Text-to-Speech)
- **Video Processing**: moviepy 2.2.1
- **Frontend**: Streamlit 1.41.1
- **Language**: Python 3.10+

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Google API Key (for embeddings)
- Replicate API Token ($5 free credit)
- OpenAI API Key (optional)

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/Mariame-Qr/Storytelling.git
cd Storytelling
```

2. **Create virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**

```bash
pip install -r requirements-multimodal.txt
```

4. **Configure environment variables**

Create `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
```

**Get API Keys**:
- Google Gemini: https://aistudio.google.com/app/apikey
- Replicate: https://replicate.com/account/api-tokens ($5 free credit)
- OpenAI: https://platform.openai.com/api-keys

5. **Initialize RAG knowledge base**

```bash
python ingest.py
```

This creates a local Qdrant database with automotive technical documentation.

## 🚀 Usage

### Start the multimodal application

```bash
streamlit run app_multimodal.py
```

The app will open in your browser at `http://localhost:8501`

### Format Selection

Choose your preferred output format:

- **🎬 Full Multimodal** - Complete experience with story, audio, and video
- **🎤 Audio Only** - Story text + audio narration (fastest, always works)
- **🖼️ Image Only** - Story text + static image
- **📹 Video Only** - Story text + video (requires Replicate credit)

### Example Prompts

Try these automotive feature requests:

- "Explain how the all-wheel drive system distributes torque"
- "Show me how ABS prevents wheel lockup during emergency braking"
- "Visualize the turbocharger boosting engine power"
- "Demonstrate how electronic stability control prevents skidding"
- "Show the hybrid powertrain switching between electric and combustion"
- "Explain the differential mechanism in action"
- "Visualize the adaptive cruise control maintaining distance"

### How It Works

1. **Enter your automotive query** in the text input
2. **Select output format** (Full Multimodal / Audio Only / Video Only)
3. **AI agents execute workflow**:
   - 📖 Orchestrator coordinates the pipeline
   - 🔧 Technical Expert retrieves specs from RAG database
   - ✍️ Storyteller crafts engaging narrative
   - 🎤 Audio Agent generates narration (gTTS)
   - 🎬 Creative Director generates video (Replicate API)
   - 🎞️ Assembly Agent merges audio + video
4. **Results displayed** with synchronized narrated video

## 📁 Project Structure

```
Storytelling/
├── app_multimodal.py          # Streamlit multimodal frontend
├── backend_multimodal.py      # CrewAI agents + Replicate workflow
├── ingest.py                  # RAG knowledge base ingestion
├── requirements-multimodal.txt # Python dependencies
├── .env                       # API keys (create this)
├── .gitignore                # Git exclusions
├── qdrant_db/                # Qdrant vector database
├── generated_audio/          # Generated MP3 narrations
├── generated_outputs/        # Generated videos
└── video_library/            # Video assets (optional)
```

## 🛠️ Technical Details

### RAG Knowledge Base

The system includes a comprehensive automotive technical manual covering:

- Braking systems (ABS, EBA)
- Powertrain (engine torque, AWD, differentials)
- Safety systems (airbags, ADAS, ESC)
- Vehicle dynamics (suspension, steering)
- Hybrid/Electric systems (battery, motors)
- Comfort features (climate, infotainment)

### CrewAI Workflow

**Sequential Task Execution**:

1. **Orchestration Task** → Coordinator plans the workflow
2. **Research Task** → Technical Expert retrieves specifications
3. **Storytelling Task** → Storyteller creates narrative (150-250 words)
4. **Audio Task** → Audio Agent generates narration
5. **Video Task** → Creative Director generates video with Replicate
6. **Assembly Task** → Video Assembler merges audio + video

### Replicate Video Generation Workflow

**Step-by-Step Process**:

1. **Intelligent Prompt Generation**
   - Analyzes story content for automotive keywords (AWD, turbo, electric, etc.)
   - Enhances prompt with cinematic descriptors
   - Example: "professional automotive photograph showing all-wheel drive AWD system showing power distribution, cinematic lighting, 8K ultra high definition"

2. **SDXL Image Generation**
   - Model: `stability-ai/sdxl`
   - Generates high-quality base image (1024x1024)
   - Professional automotive photography style

3. **Stable Video Diffusion**
   - Model: `stability-ai/stable-video-diffusion`
   - Converts image to 14-frame video
   - 6 fps playback, smooth camera movement
   - Motion bucket: 127, conditioning: 0.02

4. **Video Download & Save**
   - Downloads MP4 from Replicate
   - Saves to `generated_outputs/replicate_video_XXX.mp4`
   - Placeholder fallback if Replicate credit exhausted

5. **Audio-Video Merge**
   - Uses moviepy 2.2.1
   - Synchronizes audio narration with video
   - Final output: `generated_outputs/narrated_video_XXX.mp4`

### Intelligent Prompt Enhancement

The system automatically detects automotive terms and creates contextual prompts:

| Detected Term | Enhanced Prompt |
|--------------|----------------|
| AWD / all-wheel drive | "all-wheel drive system showing power distribution" |
| Turbo | "turbocharger with visible turbine blades" |
| Electric / EV | "electric vehicle powertrain with battery pack" |
| Differential | "automotive differential mechanism in detail" |
| Suspension | "car suspension system with shock absorbers" |

## 🎬 Replicate API Setup

### Getting Replicate Credit

1. **Create Account**: https://replicate.com/
2. **Add Payment**: https://replicate.com/account/billing
3. **Get $5 Free Credit** (first-time users)
4. **Copy API Token**: https://replicate.com/account/api-tokens

### Pricing

- **SDXL Image**: ~$0.003 per image
- **Stable Video Diffusion**: ~$0.02 per video
- **Total per query**: ~$0.023 (with $5 credit = ~200 videos)

### Fallback Behavior

If Replicate credit is exhausted:
- System creates professional placeholder videos
- Shows user query and branding
- Narration still works perfectly
- Maintains full workflow

## ⚠️ Important Notes

### Requirements

- **Replicate API Credit**: Required for real video generation
  - Add credit at https://replicate.com/account/billing
  - $5 covers ~200 video generations
  - Placeholder videos work without credit

- **Audio Always Works**: gTTS narration never fails
- **RAG Database**: Run `python ingest.py` first time only

### Troubleshooting

**"Insufficient credit" (Replicate 402 error)**:
- Add credit to Replicate account
- System automatically creates placeholder videos as fallback

**Missing Qdrant database**:
- Run `python ingest.py` to create the knowledge base

**API errors**:
- Verify `.env` file has correct API keys
- Check API key validity on respective platforms

**moviepy errors**:
- Using moviepy 2.2.1 (new API)
- Ensure ImageMagick is installed (optional for effects)

**Slow video generation**:
- Replicate takes 20-60 seconds per video
- Be patient during "Calling Replicate API..." step

## 🎯 Performance

### Execution Times

- **Audio Only**: ~5-10 seconds (gTTS)
- **Video Generation**: ~30-90 seconds (Replicate SDXL + SVD)
- **Audio + Video Merge**: ~5-10 seconds (moviepy)
- **Total Full Workflow**: ~1-2 minutes

### Output Quality

- **Audio**: Clear natural voice narration (gTTS)
- **Video**: Professional 1080p cinematic footage
- **Frame Rate**: 6 fps smooth motion
- **Duration**: 2-3 seconds of dynamic video

## 🚀 Deployment Tips

1. **Pre-initialize Qdrant** before demo: `python ingest.py`
2. **Test API keys** with audio-only mode first
3. **Add Replicate credit** ($5 minimum for live demo)
4. **Prepare example queries** for quick demonstrations
5. **Monitor execution** - videos take 30-60 seconds
6. **Use audio-only** as fast fallback during demos

## 📊 System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 3GB for dependencies and generated content
- **Internet**: Required for all API calls
- **Python**: 3.10 or higher
- **OS**: Windows, Linux, or macOS

## 🤝 Contributing

This is an open-source multimodal AI project. Contributions welcome:

- Add more automotive technical content to RAG database
- Improve agent prompts and coordination
- Enhance visual prompt generation logic
- Add new features (multi-language, voice input, etc.)
- Optimize Replicate API usage

## 📄 License

MIT License - Free for educational and commercial use

## 🙏 Acknowledgments

- **CrewAI** for the multi-agent orchestration framework
- **Replicate** for Stable Diffusion and Video Diffusion APIs
- **Google** for Gemini LLM and embeddings
- **OpenAI** for GPT-4o-mini LLM
- **Qdrant** for vector database technology
- **gTTS** for text-to-speech narration
- **Streamlit** for rapid UI development
- **moviepy** for video processing

## 🔗 Links

- **GitHub**: https://github.com/Mariame-Qr/Storytelling
- **Replicate**: https://replicate.com/
- **CrewAI Docs**: https://docs.crewai.com/
- **Qdrant**: https://qdrant.tech/

---

**Built with ❤️ using cutting-edge AI technologies**

*Transform automotive complexity into immersive storytelling* 🚗🎬✨

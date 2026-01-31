# 🚗 AutoStory - Intelligent Automotive Storytelling Agent

An agentic AI application that transforms automotive technical specifications into immersive visual storytelling experiences using CrewAI, Google Gemini, and Stable Diffusion.

## 🎯 Project Overview

AutoStory is a hackathon project that demonstrates intelligent decision-making in AI systems. The application:

- **Analyzes** user intent to understand what automotive feature they want to visualize
- **Decides** whether to generate a static image or dynamic video
- **Generates** high-quality visuals that precisely illustrate the requested feature

### Key Innovation: Intent-Driven Visual Selection

Unlike traditional systems that blindly generate content, AutoStory's Creative Director agent intelligently classifies requests:

- **STATIC_FEATURE** → Generates IMAGE (e.g., vehicle design, dashboard layout)
- **DYNAMIC_MECHANISM** → Generates VIDEO (e.g., ABS braking, airbag deployment)
- **EMOTIONAL_EXPERIENCE** → Generates VIDEO (e.g., safety scenarios, driving confidence)

## 🏗️ Architecture

### Agentic System (CrewAI)

**Three Specialized Agents**:

1. **Technical Expert Agent**
   - Retrieves accurate vehicle specifications from RAG knowledge base
   - Ensures factual accuracy, prevents hallucinations
   - Tools: Qdrant vector search

2. **Storyteller Agent**
   - Transforms technical specs into engaging narratives
   - Maintains technical accuracy while being accessible
   - Creates emotional connection to features

3. **Creative Director Agent** (Decision Maker)
   - Classifies user intent
   - Decides IMAGE vs VIDEO format
   - Generates precise visual prompts
   - Creates visual assets using SDXL or SVD
   - Tools: Stable Diffusion XL, Stable Video Diffusion

### Tech Stack

- **Orchestration**: CrewAI 0.86.0
- **LLM**: Google Gemini 1.5 Flash
- **Vector DB**: Qdrant (local persistent mode)
- **Embeddings**: Google Generative AI Embeddings
- **Image Gen**: Stable Diffusion XL (HuggingFace API)
- **Video Gen**: Stable Video Diffusion (HuggingFace Space)
- **Frontend**: Streamlit
- **Language**: Python 3.10+

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Google API Key (Gemini)
- HuggingFace API Token

### Setup Steps

1. **Clone or download the project**

```bash
cd projet
```

2. **Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy `.env.example` to `.env` and add your API keys:

```bash
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

**Get API Keys**:
- Google Gemini: https://aistudio.google.com/app/apikey
- HuggingFace: https://huggingface.co/settings/tokens

5. **Initialize RAG knowledge base**

```bash
python ingest.py
```

This creates a local Qdrant database with automotive technical documentation.

## 🚀 Usage

### Start the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Example Prompts

Try these example requests:

- "Show me how ABS prevents wheel lockup during emergency braking"
- "Visualize the airbag deployment system in action"
- "Explain how the all-wheel drive system distributes torque"
- "Show the engine delivering maximum torque"
- "Demonstrate how electronic stability control prevents skidding"
- "Visualize the adaptive cruise control maintaining distance"
- "Show the interior dashboard and infotainment system"

### How It Works

1. **Enter your request** about any automotive feature
2. **AI agents analyze** the technical requirements:
   - Technical Expert retrieves specs from knowledge base
   - Storyteller crafts engaging narrative
   - Creative Director decides visual format
3. **Visual generated** showing the exact feature (image or video)
4. **Results displayed** with story and visual asset

## 📁 Project Structure

```
projet/
├── ingest.py              # RAG knowledge base ingestion
├── backend.py             # CrewAI agents, tools, and orchestration
├── app.py                 # Streamlit frontend
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .env                  # Your API keys (create this)
├── qdrant_db/            # Qdrant vector database (created by ingest.py)
└── generated_outputs/    # Generated images and videos
```

## 🛠️ Technical Details

### RAG Knowledge Base

The system includes a comprehensive automotive technical manual covering:

- Braking systems (ABS, EBA)
- Powertrain (engine torque, AWD)
- Safety systems (airbags, ADAS, ESC)
- Vehicle dynamics (suspension, steering)
- Comfort features (climate, infotainment)
- Efficiency systems (start-stop, eco mode)

### CrewAI Workflow

**Sequential Task Execution**:

1. **Research Task** → Technical Expert retrieves specifications
2. **Storytelling Task** → Storyteller creates narrative
3. **Visual Task** → Creative Director generates visual

Each task builds on the previous one, ensuring coherent and accurate output.

### Visual Generation

**Image Generation (SDXL)**:
- High-quality 1024x1024 images
- Photorealistic automotive visuals
- 30 inference steps, guidance scale 7.5

**Video Generation (SVD)**:
- 2-4 second clips showing dynamic processes
- Automatic fallback to image if video fails
- Requires seed image (auto-generated if needed)

## 🎨 Decision-Making Logic

The Creative Director uses this classification framework:

### STATIC_FEATURE → IMAGE
- Vehicle exterior/interior design
- Component diagrams and cross-sections
- Dashboard layouts and controls
- Engine overviews

### DYNAMIC_MECHANISM → VIDEO
- ABS braking in action
- Airbag deployment sequence
- Torque distribution visualization
- Suspension movement
- Safety system interventions

### EMOTIONAL_EXPERIENCE → VIDEO
- Family safety scenarios
- Emergency response situations
- Driving confidence moments
- Protective features in action

## ⚠️ Important Notes

### Limitations

- **Video generation is experimental** and may fail (automatic fallback to images)
- **API rate limits** apply to HuggingFace inference
- **Model loading times** can cause initial delays
- **Free tier limitations** on API calls

### Troubleshooting

**"Model loading" errors**:
- Wait 10-30 seconds and retry
- HuggingFace models need to warm up

**Missing Qdrant database**:
- Run `python ingest.py` first

**API errors**:
- Verify `.env` file has correct API keys
- Check API key validity and quotas

## 🚀 Hackathon Deployment Tips

1. **Pre-generate** Qdrant database before demo
2. **Test API keys** before presentation
3. **Prepare fallback examples** with pre-generated visuals
4. **Monitor API quotas** during demo day
5. **Use example prompts** for quick demonstrations

## 📊 System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies and models
- **Internet**: Required for API calls
- **Python**: 3.10 or higher

## 🤝 Contributing

This is a hackathon project. Feel free to:

- Add more automotive technical content
- Improve agent prompts and decision logic
- Enhance visual generation prompts
- Add new features (voice input, multi-language, etc.)

## 📄 License

MIT License - Free for educational and hackathon use

## 🙏 Acknowledgments

- **CrewAI** for the agentic framework
- **Google** for Gemini API
- **HuggingFace** for Stable Diffusion models
- **Qdrant** for vector database
- **Streamlit** for rapid UI development

---

**Built for hackathons with ❤️ by AI enthusiasts**

*Transform automotive complexity into visual clarity* 🚗✨

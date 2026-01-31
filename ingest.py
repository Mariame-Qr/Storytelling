"""
RAG Ingestion Script for AutoStory
Initializes Qdrant vector database with automotive technical documentation
"""

import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


class CarManualData:
    """Dummy automotive technical documentation for RAG"""
    
    TECHNICAL_SPECS = [
        # Braking System
        {
            "title": "Anti-lock Braking System (ABS)",
            "content": """
            The Anti-lock Braking System (ABS) prevents wheel lockup during emergency braking.
            It uses wheel speed sensors to detect when a wheel is about to lock.
            The system modulates brake pressure up to 15 times per second.
            This maintains steering control and reduces stopping distance on most surfaces.
            ABS activates automatically when threshold braking is detected.
            The system consists of: wheel speed sensors, hydraulic control unit, electronic control module.
            Braking distance at 60 mph: 120-140 feet on dry asphalt with ABS active.
            """
        },
        {
            "title": "Emergency Brake Assist (EBA)",
            "content": """
            Emergency Brake Assist detects panic braking and applies maximum braking force.
            It monitors brake pedal speed and pressure application rate.
            When emergency braking is detected, the system instantly applies full brake pressure.
            Works in conjunction with ABS for optimal stopping performance.
            Can reduce braking distance by up to 20% in emergency situations.
            Threshold detection: brake pedal pressed faster than 200mm/s.
            """
        },
        
        # Engine & Powertrain
        {
            "title": "Engine Torque Delivery",
            "content": """
            The 2.0L turbocharged engine delivers peak torque of 280 lb-ft from 1,500 to 4,500 RPM.
            Variable valve timing optimizes power delivery across the RPM range.
            Turbocharger spools up from 1,200 RPM providing immediate throttle response.
            Engine management system adjusts fuel injection and ignition timing 100 times per second.
            Power output: 245 horsepower at 5,500 RPM.
            Compression ratio: 10.5:1 for optimal efficiency and performance balance.
            Direct fuel injection operates at up to 2,500 PSI for precise combustion control.
            """
        },
        {
            "title": "All-Wheel Drive (AWD) System",
            "content": """
            The intelligent AWD system continuously monitors wheel slip and road conditions.
            Can transfer up to 50% of torque to the rear axle within milliseconds.
            Uses electromagnetic clutch pack for seamless power distribution.
            Normal driving: 90% front, 10% rear torque split.
            Slippery conditions: automatic adjustment up to 50/50 split.
            System operates proactively based on steering angle, throttle position, and wheel speed.
            """
        },
        
        # Safety Systems
        {
            "title": "Airbag Deployment System",
            "content": """
            The vehicle is equipped with 8 airbags: front, side, curtain, and knee airbags.
            Crash sensors detect deceleration forces exceeding 15G.
            Airbag deployment occurs within 20-30 milliseconds of impact detection.
            Front airbags deploy at 200 mph to cushion occupants.
            Side curtain airbags remain inflated for 5-6 seconds during rollover events.
            Sensor network: 6 accelerometers positioned throughout the vehicle structure.
            Deployment thresholds vary based on crash severity and occupant position sensors.
            """
        },
        {
            "title": "Advanced Driver Assistance Systems (ADAS)",
            "content": """
            ADAS suite includes: adaptive cruise control, lane keeping assist, blind spot monitoring.
            Forward-facing camera and radar monitor traffic up to 150 meters ahead.
            Lane keeping assist provides gentle steering corrections at speeds above 40 mph.
            Adaptive cruise control maintains 1.5 to 2.5 second following distance.
            Automatic emergency braking can detect pedestrians and vehicles.
            System activates autonomous braking if collision is imminent and driver doesn't respond.
            Blind spot radar monitors adjacent lanes up to 3 meters.
            """
        },
        {
            "title": "Electronic Stability Control (ESC)",
            "content": """
            ESC prevents loss of control during cornering and evasive maneuvers.
            Monitors steering angle, lateral acceleration, and individual wheel speeds.
            Selectively applies brakes to individual wheels to correct vehicle trajectory.
            Can reduce engine power if oversteer or understeer is detected.
            Operates 100 times per second for real-time stability management.
            Reduces accident risk by up to 32% according to safety studies.
            """
        },
        
        # Vehicle Dynamics
        {
            "title": "Suspension System",
            "content": """
            MacPherson strut front suspension with independent multi-link rear.
            Adaptive dampers adjust firmness based on road conditions and driving mode.
            Suspension travel: 120mm front, 130mm rear.
            Sport mode: 40% firmer damping for enhanced handling.
            Comfort mode: optimized for ride quality over rough surfaces.
            Anti-roll bars reduce body roll during cornering by 30%.
            """
        },
        {
            "title": "Steering System",
            "content": """
            Electric power steering with variable assist based on vehicle speed.
            Steering ratio: 14.7:1 for balance between responsiveness and stability.
            Lock-to-lock turns: 2.5 for excellent maneuverability.
            At parking speeds: maximum assist for effortless maneuvering.
            At highway speeds: reduced assist for stable, confident feel.
            Steering response time: 0.15 seconds from input to wheel angle change.
            """
        },
        
        # Comfort & Features
        {
            "title": "Climate Control System",
            "content": """
            Dual-zone automatic climate control maintains individual temperature preferences.
            System uses 8 sensors to monitor cabin temperature and humidity.
            Air quality sensor filters external air and recirculates when pollution is detected.
            Cooling capacity: 6,000 BTU for rapid cabin cooling.
            Heating: integrated with engine coolant system for efficient warmth.
            Defrost mode directs maximum airflow to windshield and side windows.
            """
        },
        {
            "title": "Infotainment System",
            "content": """
            12.3-inch touchscreen with wireless Apple CarPlay and Android Auto.
            Premium audio system with 12 speakers and 600-watt amplifier.
            Voice command system recognizes natural language requests.
            Navigation with real-time traffic updates and predictive routing.
            Over-the-air software updates keep system current.
            Bluetooth connectivity supports up to 2 simultaneous device connections.
            """
        },
        
        # Efficiency
        {
            "title": "Fuel Efficiency Features",
            "content": """
            Start-stop system automatically shuts off engine at idle to save fuel.
            Eco mode optimizes throttle response and transmission shifts for efficiency.
            Active grille shutters close at highway speeds to reduce aerodynamic drag.
            Low rolling resistance tires reduce energy loss.
            EPA estimated: 28 mpg city, 36 mpg highway, 31 mpg combined.
            18-gallon fuel tank provides up to 560 miles of highway range.
            """
        }
    ]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Simple text chunking with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    
    return chunks


def ingest_car_manual():
    """
    Ingest automotive technical documentation into Qdrant vector database
    """
    print("=" * 70)
    print("AutoStory RAG Ingestion")
    print("=" * 70)
    
    # Initialize Qdrant client (local persistent mode)
    print("\n[1/4] Initializing Qdrant client...")
    client = QdrantClient(path="./qdrant_db")
    
    collection_name = "car_specs"
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"      ✓ Deleted existing collection: {collection_name}")
    except Exception:
        pass
    
    # Create new collection
    print(f"\n[2/4] Creating collection: {collection_name}")
    
    # Get embedding dimension by creating a test embedding
    test_embedding = embeddings.embed_query("test")
    embedding_dim = len(test_embedding)
    print(f"      ✓ Embedding dimension: {embedding_dim}")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE
        )
    )
    print(f"      ✓ Collection created successfully")
    
    # Process and embed documents
    print(f"\n[3/4] Processing {len(CarManualData.TECHNICAL_SPECS)} technical documents...")
    
    points = []
    point_id = 0
    
    for doc in CarManualData.TECHNICAL_SPECS:
        title = doc["title"]
        content = doc["content"].strip()
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size=400, overlap=50)
        
        print(f"      • {title}: {len(chunks)} chunks")
        
        for chunk_idx, chunk in enumerate(chunks):
            # Combine title with chunk for better context
            combined_text = f"Title: {title}\n\n{chunk}"
            
            # Generate embedding
            embedding_vector = embeddings.embed_query(combined_text)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload={
                    "title": title,
                    "content": chunk,
                    "chunk_index": chunk_idx,
                    "full_text": combined_text
                }
            )
            points.append(point)
            point_id += 1
    
    # Upload to Qdrant
    print(f"\n[4/4] Uploading {len(points)} vectors to Qdrant...")
    client.upload_points(
        collection_name=collection_name,
        points=points
    )
    print(f"      ✓ Successfully uploaded all vectors")
    
    # Verify collection
    collection_info = client.get_collection(collection_name)
    print(f"\n{'=' * 70}")
    print(f"✓ RAG Ingestion Complete!")
    print(f"{'=' * 70}")
    print(f"Collection: {collection_name}")
    print(f"Total vectors: {collection_info.points_count}")
    print(f"Vector dimension: {embedding_dim}")
    print(f"Storage path: ./qdrant_db")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    ingest_car_manual()

"""
Test simple de Replicate API pour génération vidéo
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import replicate

# Load environment
load_dotenv()

# Create output directory
OUTPUT_DIR = Path("generated_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def test_replicate_video():
    """Test direct de la génération vidéo avec Replicate"""
    
    print("\n" + "=" * 80)
    print("🎬 TEST REPLICATE API - GÉNÉRATION VIDÉO")
    print("=" * 80)
    
    # Check API key
    api_key = os.getenv('REPLICATE_API_TOKEN')
    if not api_key:
        print("❌ ERREUR: REPLICATE_API_TOKEN non trouvé dans .env")
        return None
    
    print(f"✓ API Key trouvée: {api_key[:10]}...")
    
    # Prompt pour test
    prompt = "Professional automotive cinematography showing an all-wheel drive system distributing torque between front and rear axles, cinematic camera movement, smooth motion, 4K quality, realistic lighting, detailed mechanical parts"
    
    print(f"\n📝 Prompt: {prompt[:100]}...")
    print("\n⏳ Appel à Replicate API (peut prendre 30-90 secondes)...")
    
    try:
        start_time = time.time()
        
        # Call Replicate API - Stable Video Diffusion (image-to-video)
        # Note: Ce modèle nécessite une image en entrée, nous utilisons une URL d'exemple
        print("\n🎨 Utilisation de Stable Video Diffusion (image-to-video)")
        print("📸 Génération d'abord d'une image conceptuelle...")
        
        # Option 1: Utiliser un modèle image-to-video avec une URL d'image générique
        output = replicate.run(
            "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
            input={
                "input_image": "https://replicate.delivery/pbxt/JvNdEmiW4NjlIECQ2Ayfn5OGPZhPGwZPpNYlKCKzBmGmUvGl/robot.png",
                "video_length": "14_frames_with_svd",
                "sizing_strategy": "maintain_aspect_ratio",
                "frames_per_second": 6,
                "motion_bucket_id": 127,
                "cond_aug": 0.02
            }
        )
        
        generation_time = time.time() - start_time
        print(f"\n✓ Génération terminée en {generation_time:.1f}s")
        
        # Download video
        if output:
            video_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) else str(output))
            
            print(f"\n📥 URL de la vidéo: {video_url}")
            print("⬇️ Téléchargement...")
            
            response = requests.get(video_url, timeout=120)
            
            if response.status_code == 200:
                timestamp = int(time.time())
                filepath = OUTPUT_DIR / f"test_replicate_{timestamp}.mp4"
                
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                file_size = filepath.stat().st_size / 1024  # KB
                total_time = time.time() - start_time
                
                print("\n" + "=" * 80)
                print("✅ SUCCÈS - VIDÉO GÉNÉRÉE")
                print("=" * 80)
                print(f"📁 Fichier: {filepath}")
                print(f"📊 Taille: {file_size:.1f} KB")
                print(f"⏱️ Temps total: {total_time:.1f}s")
                print("=" * 80)
                
                return str(filepath.absolute())
            else:
                print(f"\n❌ Erreur de téléchargement: {response.status_code}")
                return None
        else:
            print("\n❌ Aucune sortie reçue de Replicate")
            return None
            
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_replicate_video()
    
    if result:
        print(f"\n✅ Test réussi ! Vidéo disponible: {result}")
    else:
        print("\n❌ Test échoué")

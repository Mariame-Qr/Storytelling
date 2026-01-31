"""
Test du workflow vidéo complet (Replicate ou Fallback)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Import du backend
from backend_multimodal import GenerateVideoWithReplicateTool, GenerateNarrationTool, MergeAudioVideoTool

load_dotenv()

def test_video_generation():
    """Test de génération vidéo avec fallback automatique"""
    
    print("\n" + "=" * 80)
    print("🎬 TEST WORKFLOW VIDÉO COMPLET")
    print("=" * 80)
    
    # 1. Générer l'audio
    print("\n📝 ÉTAPE 1 : Génération de l'audio narration")
    print("-" * 80)
    
    story_text = """The all-wheel drive system is an advanced automotive technology that intelligently distributes engine torque between the front and rear axles to optimize traction and handling.

When the vehicle accelerates, sensors continuously monitor wheel speed and traction conditions. The system uses a center differential or electronic coupling to split power between the axles."""
    
    audio_tool = GenerateNarrationTool()
    audio_path = audio_tool._run(story_text, language="en")
    
    if not audio_path or audio_path.startswith("Error"):
        print("❌ Échec génération audio")
        return None
    
    print(f"✅ Audio généré: {audio_path}")
    
    # 2. Générer la vidéo (Replicate ou fallback)
    print("\n📝 ÉTAPE 2 : Génération de la vidéo")
    print("-" * 80)
    
    video_tool = GenerateVideoWithReplicateTool()
    prompt = "Professional automotive cinematography showing an all-wheel drive AWD system distributing torque between front and rear axles, cinematic camera movement, smooth motion, detailed mechanical parts"
    
    video_path = video_tool._run(prompt)
    
    if not video_path or not Path(video_path).exists():
        print("❌ Échec génération vidéo")
        return None
    
    print(f"✅ Vidéo générée: {video_path}")
    
    # 3. Fusionner audio + vidéo
    print("\n📝 ÉTAPE 3 : Fusion audio + vidéo")
    print("-" * 80)
    
    merge_tool = MergeAudioVideoTool()
    final_video = merge_tool._run(video_path, audio_path)
    
    if not final_video or not Path(final_video).exists():
        print("❌ Échec fusion")
        return None
    
    print(f"✅ Vidéo finale générée: {final_video}")
    
    # Résumé
    print("\n" + "=" * 80)
    print("✅ WORKFLOW TERMINÉ AVEC SUCCÈS")
    print("=" * 80)
    print(f"📁 Vidéo finale: {final_video}")
    print(f"📊 Taille: {Path(final_video).stat().st_size / 1024:.1f} KB")
    print("=" * 80)
    
    return final_video


if __name__ == "__main__":
    result = test_video_generation()
    
    if result:
        print(f"\n🎉 SUCCÈS ! Vous pouvez ouvrir la vidéo: {result}")
        print("\nℹ️  NOTE:")
        print("   - Si vidéo = placeholder → Ajoutez crédit Replicate: https://replicate.com/account/billing")
        print("   - Si vidéo = Replicate → Professionnel et prêt pour votre démo ! 🚀")
    else:
        print("\n❌ Échec du workflow")

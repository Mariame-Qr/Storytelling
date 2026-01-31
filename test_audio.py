"""
Test de génération audio avec gTTS
"""
from gtts import gTTS
import os
from pathlib import Path

# Créer le répertoire
audio_dir = Path("generated_audio")
audio_dir.mkdir(exist_ok=True)

# Test simple
print("🎤 Test de génération audio...")
text = "This is a test of the audio generation system for AutoStory."

try:
    tts = gTTS(text=text, lang="en", slow=False)
    audio_file = audio_dir / "test_narration.mp3"
    tts.save(str(audio_file))
    print(f"✅ Audio généré avec succès: {audio_file}")
    print(f"Taille du fichier: {os.path.getsize(audio_file)} bytes")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

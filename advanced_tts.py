#!/usr/bin/env python3
"""
Advanced IndexTTS2 Usage Script

This script demonstrates advanced features of IndexTTS2 including:
- Voice cloning with different reference voices
- Emotion control via audio prompts
- Emotion control via emotion vectors
- Text-based emotion analysis

Usage:
    python advanced_tts.py
    
Requirements:
    - IndexTTS2 model checkpoints in ./checkpoints/
    - Reference audio files in ./examples/
"""

import os
import sys
from pathlib import Path
import random

# Add the project root to Python path if running as a script
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def import_indextts():
    """Import IndexTTS2 with proper error handling."""
    try:
        from indextts.infer_v2 import IndexTTS2
        return IndexTTS2
    except ImportError as e:
        print(f"Error importing IndexTTS2: {e}")
        print("Make sure you're running this script from the project root directory.")
        print("Also ensure all dependencies are installed (librosa, torch, etc.)")
        sys.exit(1)


class AdvancedTTSDemo:
    """Advanced TTS demonstration class."""
    
    def __init__(self):
        """Initialize the TTS model."""
        self.config_path = "checkpoints/config.yaml"
        self.model_dir = "checkpoints"
        self.tts = None
        
    def initialize_model(self):
        """Initialize the IndexTTS2 model."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        # Import IndexTTS2 here to delay dependency loading    
        IndexTTS2 = import_indextts()
            
        print("Initializing IndexTTS2 with advanced features...")
        self.tts = IndexTTS2(
            cfg_path=self.config_path,
            model_dir=self.model_dir,
            use_fp16=False,  # CPU compatibility
            use_cuda_kernel=False,  # CPU compatibility
            use_deepspeed=False  # Simplicity
        )
        print("Model initialized successfully!")
        
    def demo_basic_voice_cloning(self):
        """Demonstrate basic voice cloning."""
        print("\n=== Demo 1: Basic Voice Cloning ===")
        
        reference_voice = "examples/voice_01.wav"
        text = "This is a demonstration of voice cloning technology using IndexTTS2."
        output_file = "output_basic_cloning.wav"
        
        if not os.path.exists(reference_voice):
            print(f"Skipping: Reference voice {reference_voice} not found")
            return
            
        print(f"Cloning voice from: {reference_voice}")
        print(f"Text: '{text}'")
        
        self.tts.infer(
            spk_audio_prompt=reference_voice,
            text=text,
            output_path=output_file,
            verbose=True
        )
        
        print(f"Output saved to: {output_file}")
        
    def demo_emotion_control_with_audio(self):
        """Demonstrate emotion control using emotion audio prompts."""
        print("\n=== Demo 2: Emotion Control with Audio Prompts ===")
        
        reference_voice = "examples/voice_07.wav"
        emotion_audio = "examples/emo_sad.wav"
        text = "I can't believe this happened. This is so disappointing and heartbreaking."
        output_file = "output_emotional_audio.wav"
        
        if not all(os.path.exists(f) for f in [reference_voice, emotion_audio]):
            print(f"Skipping: Required audio files not found")
            return
            
        print(f"Voice: {reference_voice}")
        print(f"Emotion audio: {emotion_audio}")
        print(f"Text: '{text}'")
        
        self.tts.infer(
            spk_audio_prompt=reference_voice,
            text=text,
            output_path=output_file,
            emo_audio_prompt=emotion_audio,
            emo_alpha=0.9,  # Strong emotion influence
            verbose=True
        )
        
        print(f"Output saved to: {output_file}")
        
    def demo_emotion_vectors(self):
        """Demonstrate emotion control using emotion vectors."""
        print("\n=== Demo 3: Emotion Control with Emotion Vectors ===")
        
        reference_voice = "examples/voice_10.wav"
        
        # Emotion vector: [Happy, Angry, Sad, Fearful, Disgusted, Melancholic, Surprised, Calm]
        emotions = {
            "happy": [0.8, 0, 0, 0, 0, 0, 0.2, 0],
            "angry": [0, 0.9, 0, 0, 0.1, 0, 0, 0],
            "sad": [0, 0, 0.8, 0, 0, 0.2, 0, 0],
            "surprised": [0.3, 0, 0, 0, 0, 0, 0.7, 0],
            "calm": [0, 0, 0, 0, 0, 0, 0, 1.0]
        }
        
        texts = {
            "happy": "What a wonderful day! I'm so excited about this amazing opportunity!",
            "angry": "This is completely unacceptable! I demand better service!",
            "sad": "I miss you so much. Nothing feels the same without you here.",
            "surprised": "Oh my goodness! I can't believe you did this for me!",
            "calm": "Let's take a deep breath and approach this situation with a clear mind."
        }
        
        if not os.path.exists(reference_voice):
            print(f"Skipping: Reference voice {reference_voice} not found")
            return
            
        for emotion_name, emotion_vector in emotions.items():
            text = texts[emotion_name]
            output_file = f"output_emotion_{emotion_name}.wav"
            
            print(f"\nGenerating {emotion_name} emotion...")
            print(f"Emotion vector: {emotion_vector}")
            print(f"Text: '{text}'")
            
            self.tts.infer(
                spk_audio_prompt=reference_voice,
                text=text,
                output_path=output_file,
                emo_vector=emotion_vector,
                verbose=True
            )
            
            print(f"Output saved to: {output_file}")
            
    def demo_text_based_emotion(self):
        """Demonstrate text-based emotion analysis."""
        print("\n=== Demo 4: Text-Based Emotion Analysis ===")
        
        reference_voice = "examples/voice_12.wav"
        text = "Oh no! This is terrible! I'm so scared and worried about what might happen!"
        output_file = "output_text_emotion.wav"
        
        if not os.path.exists(reference_voice):
            print(f"Skipping: Reference voice {reference_voice} not found")
            return
            
        print(f"Voice: {reference_voice}")
        print(f"Text (with emotional content): '{text}'")
        print("Using automatic emotion detection from text...")
        
        self.tts.infer(
            spk_audio_prompt=reference_voice,
            text=text,
            output_path=output_file,
            emo_alpha=0.6,  # Moderate emotion influence
            use_emo_text=True,  # Enable text-based emotion
            verbose=True
        )
        
        print(f"Output saved to: {output_file}")
        
    def demo_custom_emotion_text(self):
        """Demonstrate custom emotion text description."""
        print("\n=== Demo 5: Custom Emotion Text Description ===")
        
        reference_voice = "examples/voice_12.wav"
        text = "The presentation will begin in five minutes."
        emotion_text = "You're feeling nervous and anxious about public speaking!"
        output_file = "output_custom_emotion_text.wav"
        
        if not os.path.exists(reference_voice):
            print(f"Skipping: Reference voice {reference_voice} not found")
            return
            
        print(f"Voice: {reference_voice}")
        print(f"Text: '{text}'")
        print(f"Emotion description: '{emotion_text}'")
        
        self.tts.infer(
            spk_audio_prompt=reference_voice,
            text=text,
            output_path=output_file,
            emo_alpha=0.6,
            use_emo_text=True,
            emo_text=emotion_text,
            verbose=True
        )
        
        print(f"Output saved to: {output_file}")
        
    def run_all_demos(self):
        """Run all demonstration functions."""
        try:
            self.initialize_model()
            
            self.demo_basic_voice_cloning()
            self.demo_emotion_control_with_audio()
            self.demo_emotion_vectors()
            self.demo_text_based_emotion()
            self.demo_custom_emotion_text()
            
            print("\n=== All demonstrations completed! ===")
            print("Generated files:")
            for filename in os.listdir('.'):
                if filename.startswith('output_') and filename.endswith('.wav'):
                    size = os.path.getsize(filename)
                    print(f"  - {filename} ({size} bytes)")
                    
        except Exception as e:
            print(f"Error during demonstration: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function."""
    print("=== IndexTTS2 Advanced Features Demonstration ===")
    print()
    print("This script will demonstrate various advanced features of IndexTTS2:")
    print("1. Basic voice cloning")
    print("2. Emotion control with audio prompts")
    print("3. Emotion control with emotion vectors")
    print("4. Text-based emotion analysis")
    print("5. Custom emotion text descriptions")
    print()
    
    demo = AdvancedTTSDemo()
    success = demo.run_all_demos()
    
    if success:
        print("\n=== Script completed successfully! ===")
    else:
        print("\n=== Script failed. Check error messages above. ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
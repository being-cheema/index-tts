#!/usr/bin/env python3
"""
Simple IndexTTS2 Usage Script

This script demonstrates basic usage of IndexTTS2 for text-to-speech synthesis.
It uses a reference voice and generates speech from text, saving the output as output.wav.

Usage:
    python simple_tts.py
    
Requirements:
    - IndexTTS2 model checkpoints in ./checkpoints/
    - Reference audio file (default: examples/voice_01.wav)
"""

import os
import sys
from pathlib import Path

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


def main():
    """Main function to demonstrate basic TTS usage."""
    
    # Configuration
    config_path = "checkpoints/config.yaml"
    model_dir = "checkpoints"
    reference_audio = "examples/voice_01.wav"
    output_file = "output.wav"
    
    # Text to synthesize
    text = "Welcome to IndexTTS2! This is a demonstration of text-to-speech synthesis with voice cloning capabilities."
    
    # Check if required files exist
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please make sure you have downloaded the model checkpoints.")
        return False
        
    if not os.path.exists(reference_audio):
        print(f"Error: Reference audio file not found at {reference_audio}")
        print("Please make sure the examples directory contains voice samples.")
        return False
    
    try:
        # Import IndexTTS2 here to delay dependency loading
        IndexTTS2 = import_indextts()
        
        print("Initializing IndexTTS2...")
        print(f"Config: {config_path}")
        print(f"Model directory: {model_dir}")
        print(f"Reference audio: {reference_audio}")
        print(f"Output file: {output_file}")
        print()
        
        # Initialize the TTS model
        # Note: Using CPU-friendly settings for broader compatibility
        tts = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            use_fp16=False,  # Disable FP16 for CPU compatibility
            use_cuda_kernel=False,  # Disable CUDA kernels for CPU compatibility
            use_deepspeed=False  # Disable DeepSpeed for simplicity
        )
        
        print(f"Synthesizing text: '{text}'")
        print("This may take a moment...")
        
        # Generate speech
        tts.infer(
            spk_audio_prompt=reference_audio,  # Voice to clone
            text=text,  # Text to synthesize
            output_path=output_file,  # Output file
            verbose=True  # Show progress information
        )
        
        print(f"\nSuccess! Audio saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        return True
        
    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== IndexTTS2 Simple Usage Script ===")
    print()
    
    success = main()
    
    if success:
        print("\n=== Script completed successfully! ===")
    else:
        print("\n=== Script failed. Check error messages above. ===")
        sys.exit(1)
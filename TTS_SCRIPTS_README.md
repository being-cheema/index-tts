# IndexTTS2 Usage Scripts

This directory contains two Python scripts that demonstrate how to use the IndexTTS2 model for text-to-speech synthesis.

## Scripts Overview

### 1. `simple_tts.py` - Basic TTS Usage
A straightforward script that demonstrates basic text-to-speech synthesis with voice cloning.

**Features:**
- Simple voice cloning from a reference audio file
- Generates speech from text
- Saves output as `output.wav`
- CPU-friendly settings for broad compatibility

**Usage:**
```bash
python simple_tts.py
```

### 2. `advanced_tts.py` - Advanced TTS Features
A comprehensive demonstration of IndexTTS2's advanced capabilities.

**Features:**
- Basic voice cloning
- Emotion control using audio prompts
- Emotion control using emotion vectors
- Text-based emotion analysis
- Custom emotion text descriptions
- Multiple output files with different emotions

**Usage:**
```bash
python advanced_tts.py
```

## Prerequisites

### Model Files
Make sure you have the following files in your project directory:
- `checkpoints/config.yaml` - Model configuration file
- `checkpoints/` - Directory containing model weights
- `examples/voice_*.wav` - Reference voice audio files
- `examples/emo_*.wav` - Emotion reference audio files

### Python Dependencies
The scripts require the following Python packages (installed via the project's requirements):
- torch
- torchaudio
- librosa
- omegaconf
- transformers
- And other dependencies listed in `pyproject.toml`

### Installation
If you haven't installed the dependencies yet:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Output Files

### simple_tts.py
- `output.wav` - Basic TTS output

### advanced_tts.py
- `output_basic_cloning.wav` - Basic voice cloning demo
- `output_emotional_audio.wav` - Emotion control with audio prompt
- `output_emotion_happy.wav` - Happy emotion vector demo
- `output_emotion_angry.wav` - Angry emotion vector demo
- `output_emotion_sad.wav` - Sad emotion vector demo
- `output_emotion_surprised.wav` - Surprised emotion vector demo
- `output_emotion_calm.wav` - Calm emotion vector demo
- `output_text_emotion.wav` - Text-based emotion analysis
- `output_custom_emotion_text.wav` - Custom emotion text description

## Emotion Vectors

The advanced script uses 8-dimensional emotion vectors representing:
1. Happy
2. Angry
3. Sad
4. Fearful
5. Disgusted
6. Melancholic
7. Surprised
8. Calm

Each dimension ranges from 0.0 to 1.0, and the sum doesn't need to equal 1.0.

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'librosa'**
   - Install dependencies: `uv sync` or `pip install -e .`

2. **Config file not found**
   - Download the IndexTTS2 model checkpoints
   - Ensure `checkpoints/config.yaml` exists

3. **Reference audio file not found**
   - Make sure example audio files are in the `examples/` directory
   - Download sample audio files if needed

4. **CUDA out of memory**
   - Scripts are configured to use CPU-friendly settings
   - Use `use_fp16=False` and `use_cuda_kernel=False`

### Performance Tips

- For faster inference on GPU, set `use_fp16=True` and `use_cuda_kernel=True`
- For CPU-only inference, keep the default settings in the scripts
- Use shorter text inputs for faster processing
- Consider using `use_deepspeed=True` for very large models (requires additional setup)

## License

These scripts are part of the IndexTTS project and follow the same license terms.
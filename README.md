# Audio Denoiser & Transcriber

A single-script Python pipeline designed to clean noisy, band-limited voice audio (such as historical war intercepts or vintage telephone recordings) using FFmpeg filters before running transcription with OpenAI Whisper.

## Features
- **FFmpeg Bandpass Filtering:** Removes low-frequency hums (<200Hz) and high-frequency noise (>3400Hz).
- **OpenAI Whisper Integration:** Transcribes cleaned audio using guided contextual prompts.
- **VRAM Optimizations:** Automatically clears PyTorch cache and switches between CUDA and CPU modes.

## Requirements
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) installed and available in system PATH.
- CUDA-compatible GPU (recommended for `large` Whisper model).


## Tested on Google Colab
<img width="1630" height="741" alt="image" src="https://github.com/user-attachments/assets/abf02886-f7ed-43bb-b573-6777ad6f04b8" />




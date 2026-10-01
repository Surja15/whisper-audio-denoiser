import argparse
import gc
import os
import subprocess
import sys
import torch
import whisper


def process_audio(input_file: str, output_txt: str, prompt: str, model_size: str = "large"):
    """Applies FFmpeg audio filtering and transcribes using OpenAI Whisper."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # 1. Clear VRAM cache before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Clean high-frequency/low-frequency noise using FFmpeg
    cleaned_filename = "cleaned_audio.wav"
    print(f"\n[1/3] Preprocessing audio via FFmpeg: '{input_file}' -> '{cleaned_filename}'...")
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-af", "highpass=f=200,lowpass=f=3400",
        cleaned_filename
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(" -> Audio cleaning complete.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg execution failed. Make sure FFmpeg is installed and added to PATH.")
        sys.exit(1)

    # 3. Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/3] Loading Whisper '{model_size}' model on {device.upper()}...")
    model = whisper.load_model(model_size, device=device)

    # 4. Run transcription
    print("\n[3/3] Transcribing audio...")
    result = model.transcribe(
        cleaned_filename,
        initial_prompt=prompt,
        language="en",
        condition_on_previous_text=False,
        temperature=0.0,
        fp16=(device == "cuda")
    )

    # 5. Output results
    print("\n" + "=" * 50)
    print("               TRANSCRIPTION RESULTS              ")
    print("=" * 50 + "\n")

    transcript_lines = []
    for segment in result["segments"]:
        line = f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}"
        print(line)
        transcript_lines.append(line)

    with open(output_txt, "w") as f:
        f.write("\n".join(transcript_lines))

    print("\n" + "=" * 50)
    print(f"Transcript saved to '{output_txt}'.")

    # Cleanup temporary cleaned audio file
    if os.path.exists(cleaned_filename):
        os.remove(cleaned_filename)

# code by Surja15
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise audio using FFmpeg and transcribe via OpenAI Whisper.")
    parser.add_argument("-i", "--input", required=True, help="Path to input audio file (e.g., war-intercept.wav)")
    parser.add_argument("-o", "--output", default="transcript_output.txt", help="Output text file path (default: transcript_output.txt)")
    parser.add_argument("-p", "--prompt", default="", help="Initial contextual prompt for Whisper")
    parser.add_argument("-m", "--model", default="large", help="Whisper model size: tiny, base, small, medium, large (default: large)")

    args = parser.parse_args()
    process_audio(args.input, args.output, args.prompt, args.model)

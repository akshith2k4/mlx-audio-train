import os
import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence

def main():
    wav_filename = "akshith.wav"
    if not os.path.exists(wav_filename):
        if os.path.exists("akhsith.wav"):
            wav_filename = "akhsith.wav"
        elif os.path.exists("raw_clips/akshith.wav"):
            wav_filename = "raw_clips/akshith.wav"
        elif os.path.exists("raw_clips/akhsith.wav"):
            wav_filename = "raw_clips/akhsith.wav"
        else:
            print(f"Error: Could not find '{wav_filename}' in the workspace directory.")
            print("Please export your audio from Audacity (akhsith.aup3) as 'akshith.wav' into this directory.")
            sys.exit(1)

    print(f"Loading audio from '{wav_filename}'...")
    audio = AudioSegment.from_wav(wav_filename)

    min_silence_len = 598  # ms
    silence_thresh = audio.dBFS - 18
    keep_silence = 150

    print(f"Audio Duration: {len(audio)/1000:.2f} seconds, dBFS: {audio.dBFS:.2f}")
    print(f"Splitting on silence (min_silence_len={min_silence_len}ms, silence_thresh={silence_thresh:.2f}dB)...")

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    output_dir = "clips"
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks, start=1):
        filename = os.path.join(output_dir, f"clip{i:03d}.wav")
        chunk.export(filename, format="wav")
        print(f"Saved {filename} ({len(chunk)/1000:.2f}s)")

    print(f"\nTotal clips created: {len(chunks)}")

if __name__ == "__main__":
    main()

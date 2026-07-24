import os
import sys
import sqlite3
import numpy as np
import soundfile as sf

def export_aup3_to_wav(aup3_path="raw_clips/akhsith.aup3", output_wav="akshith.wav", sample_rate=44100):
    if not os.path.exists(aup3_path):
        print(f"Error: Audacity project file '{aup3_path}' not found.")
        sys.exit(1)
        
    print(f"Reading Audacity project database from '{aup3_path}'...")
    conn = sqlite3.connect(aup3_path)
    cursor = conn.cursor()
    
    rows = cursor.execute("SELECT blockid, samples FROM sampleblocks ORDER BY blockid").fetchall()
    print(f"Found {len(rows)} audio sample blocks.")
    
    all_samples = []
    for blockid, blob in rows:
        data = np.frombuffer(blob, dtype=np.float32)
        all_samples.append(data)
        
    if not all_samples:
        print("Error: No audio samples found in database.")
        sys.exit(1)
        
    audio_data = np.concatenate(all_samples)
    sf.write(output_wav, audio_data, sample_rate)
    duration = len(audio_data) / sample_rate
    print(f"Successfully exported '{output_wav}'! Duration: {duration:.2f}s ({duration/60:.2f} min)")

if __name__ == "__main__":
    export_aup3_to_wav()

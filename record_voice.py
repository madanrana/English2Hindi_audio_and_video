import sounddevice as sd
import soundfile as sf

# Settings
filename = "my_voice.wav"
duration = 5  # seconds
samplerate = 16000

print("🎙️ Recording...")
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
sd.wait()  # wait until recording is finished
print("✅ Done!")

# Save to file
sf.write(filename, recording, samplerate)
print(f"Voice saved to {filename}")

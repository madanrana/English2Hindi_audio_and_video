#!/usr/bin/env python3
# yt_video_to_hindi.py
# Full pipeline: YouTube video -> Whisper segments -> Translate -> TTS per segment -> Merge -> Play

import os
import sys
import shlex
import argparse
import subprocess
import tempfile
import json
import re
from pathlib import Path

# ----------------- External libs (install instructions below) -----------------
try:
    import yt_dlp
    import whisper
    from transformers import pipeline as hf_pipeline
    from pydub import AudioSegment
except Exception as e:
    print("Missing Python packages. See README at bottom. Error:", e)
    sys.exit(1)

# gTTS (online) optional
try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

# Coqui TTS (offline) optional
HAS_COQUI = False
try:
    from TTS.api import TTS as COQUI_TTS
    import torch
    # add safe globals later only if needed
    HAS_COQUI = True
except Exception:
    HAS_COQUI = False

# ----------------- Utility run -----------------
def run(cmd, check=True):
    print(">>", cmd)
    proc = subprocess.run(cmd, shell=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd}")

# ----------------- Download video (video+audio merged) -----------------
def download_video(url: str, out_path: Path) -> Path:
    # Use a temp filename template; yt-dlp may append ext automatically
    out_template = str(out_path)
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    print("📥 Downloading video...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Ensure we return the actual created file (yt-dlp might append .mp4)
    candidate = out_path
    if not candidate.exists():
        # find any file that starts with same stem and mp4 extension
        for p in out_path.parent.glob(out_path.stem + "*.mp4"):
            return p
        # fallback: list mp4 files
        mp4s = list(out_path.parent.glob("*.mp4"))
        if mp4s:
            return mp4s[-1]
        raise FileNotFoundError("Downloaded mp4 not found.")
    return candidate

# ----------------- Extract audio ready for Whisper -----------------
def extract_audio(video_path: Path, out_wav: Path):
    print("🎵 Extracting audio to WAV (mono, 16kHz)...")
    cmd = f'ffmpeg -y -i {shlex.quote(str(video_path))} -vn -acodec pcm_s16le -ar 16000 -ac 1 {shlex.quote(str(out_wav))}'
    run(cmd)

# ----------------- Whisper transcription (segments) -----------------
def transcribe_segments(wav_path: Path, model_size="small"):
    print(f"📝 Transcribing with Whisper ({model_size}) ...")
    model = whisper.load_model(model_size)
    res = model.transcribe(str(wav_path), language='en')
    segments = []
    for s in res.get("segments", []):
        txt = s.get("text", "").strip()
        if txt:
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": txt})
    return segments, res

# ----------------- Helpers: cleaning and dedupe overlaps -----------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def word_list(s: str):
    return [w for w in re.findall(r"[0-9A-Za-z\u0900-\u097F]+['’]?[0-9A-Za-z\u0900-\u097F]*", s.lower())]

def remove_boundary_overlap(prev_text: str, cur_text: str, max_overlap_words=6) -> str:
    if not prev_text or not cur_text:
        return cur_text
    prev_words = word_list(prev_text)
    cur_words = word_list(cur_text)
    max_k = min(max_overlap_words, len(prev_words), len(cur_words))
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == cur_words[:k]:
            return " ".join(cur_words[k:])
    return cur_text

# ----------------- Translation (per-segment) -----------------
def translate_segments(segments):
    print("🌐 Translating segments to Hindi (Helsinki-NLP/opus-mt-en-hi)...")
    # Use the small Marian model specifically for en->hi
    translator = hf_pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
    out = []
    prev = ""
    for seg in segments:
        src = clean_text(seg["text"])
        src = remove_boundary_overlap(prev, src, max_overlap_words=6)
        if not src.strip():
            prev = seg["text"]
            continue
        # translator can handle short segments fine
        try:
            translated = translator(src)[0]["translation_text"]
        except Exception as e:
            print("Translation error for segment:", e)
            translated = src  # fallback: keep english (not ideal)
        translated = clean_text(translated)
        out.append({"start": seg["start"], "end": seg["end"], "text_en": seg["text"], "text_hi": translated})
        prev = seg["text"]
    return out

# ----------------- Time-stretch helper (ffmpeg atempo chaining) -----------------
def ffmpeg_time_stretch(in_wav: Path, out_wav: Path, speed_factor: float):
    # speed_factor = old_duration / target_duration
    if abs(speed_factor - 1.0) < 0.03:
        run(f'ffmpeg -y -i {shlex.quote(str(in_wav))} -ar 16000 -ac 1 {shlex.quote(str(out_wav))}')
        return
    f = float(speed_factor)
    chain = []
    # decompose into [0.5,2.0] factors
    while f > 2.0:
        chain.append(2.0)
        f /= 2.0
    while f < 0.5:
        chain.append(0.5)
        f *= 0.5
    chain.append(f)
    atempo_chain = ",".join([f"atempo={x:.6f}" for x in chain])
    cmd = f'ffmpeg -y -i {shlex.quote(str(in_wav))} -filter:a "{atempo_chain}" -ar 16000 -ac 1 {shlex.quote(str(out_wav))}'
    run(cmd)

# ----------------- Generate TTS per-segment (gTTS or Coqui) -----------------
def generate_tts_segments_gtts(trans_segments, out_dir: Path):
    if not HAS_GTTS:
        raise RuntimeError("gTTS not installed - pip install gTTS")
    print("🔊 Generating segments with gTTS (online)...")
    produced = []
    for i, seg in enumerate(trans_segments):
        if not seg["text_hi"].strip():
            continue
        mp3p = out_dir / f"seg_{i:04d}.mp3"
        wavp = out_dir / f"seg_{i:04d}.wav"
        print(f"  gTTS segment {i}: {seg['start']:.2f}-{seg['end']:.2f}")
        tts = gTTS(text=seg["text_hi"], lang='hi')
        tts.save(str(mp3p))
        # convert mp3 -> wav mono 16k
        run(f'ffmpeg -y -i {shlex.quote(str(mp3p))} -ar 16000 -ac 1 {shlex.quote(str(wavp))}')
        mp3p.unlink(missing_ok=True)
        produced.append({"start": seg["start"], "end": seg["end"], "path": wavp})
    return produced

def normalize_wav(path: Path):
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    run(f'ffmpeg -y -i {shlex.quote(str(path))} -ar 16000 -ac 1 {shlex.quote(str(tmp))}')
    os.replace(tmp, path)  # safely overwrite

def generate_tts_segments_coqui(trans_segments, out_dir: Path, speaker_wav: str = None):
    if not HAS_COQUI:
        raise RuntimeError("Coqui TTS not installed - pip install TTS")
    # Ensure user agreed to CPML or set env var
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    # allowlist required classes for PyTorch >=2.6 if torch exists
    try:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    except Exception:
        pass
    print("🔊 Generating segments with Coqui XTTS v2 (offline, larger model)...")
    coqui = COQUI_TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    produced = []
    for i, seg in enumerate(trans_segments):
        if not seg["text_hi"].strip():
            continue
        wavp = out_dir / f"seg_{i:04d}.wav"
        print(f"  XTTS segment {i}: {seg['start']:.2f}-{seg['end']:.2f}")
        coqui.tts_to_file(text=seg["text_hi"], file_path=str(wavp), language="hi",speaker_wav="/home/madan/my_voice.wav",)
        # ensure audio params
        normalize_wav(wavp)
        produced.append({"start": seg["start"], "end": seg["end"], "path": wavp})
    return produced

# ----------------- Assemble segments into single aligned wav (pydub overlay) -----------------
def assemble_segments_to_wav(segment_files, out_wav: Path, allow_time_stretch=True):
    if not segment_files:
        raise RuntimeError("No TTS segments produced.")
    total_sec = max(s["end"] for s in segment_files) + 0.3
    total_ms = int(total_sec * 1000)
    base = AudioSegment.silent(duration=total_ms, frame_rate=16000).set_channels(1).set_sample_width(2)
    for seg in segment_files:
        start_ms = int(seg["start"] * 1000)
        target_ms = int((seg["end"] - seg["start"]) * 1000)
        seg_audio = AudioSegment.from_file(str(seg["path"]))
        seg_audio = seg_audio.set_frame_rate(16000).set_channels(1)
        if allow_time_stretch and target_ms > 0:
            tts_dur = len(seg_audio)
            if tts_dur > 0:
                speed_factor = (tts_dur / 1000.0) / (target_ms / 1000.0)
                if abs(speed_factor - 1.0) > 0.15:
                    # create temp and time-stretch
                    tmp_out = Path(tempfile.mkstemp(suffix=".wav")[1])
                    ffmpeg_time_stretch(seg["path"], tmp_out, speed_factor)
                    seg_audio = AudioSegment.from_file(str(tmp_out)).set_frame_rate(16000).set_channels(1)
                    tmp_out.unlink(missing_ok=True)
        base = base.overlay(seg_audio, position=start_ms)
    # export final
    base.export(str(out_wav), format="wav", parameters=["-ar", "16000", "-ac", "1"])
    return out_wav

# ----------------- Merge into final video -----------------
def merge_audio_into_video(original_video: Path, hindi_wav: Path, out_video: Path):
    print("🎬 Merging Hindi audio with video (muting original audio)...")
    cmd = f'ffmpeg -y -i {shlex.quote(str(original_video))} -i {shlex.quote(str(hindi_wav))} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {shlex.quote(str(out_video))}'
    run(cmd)

# ----------------- Playback -----------------
def play_video(video_path: Path):
    if shutil_which("mpv"):
        run(f'mpv --force-window=yes {shlex.quote(str(video_path))}', check=False)
    elif shutil_which("vlc"):
        run(f'vlc --play-and-exit {shlex.quote(str(video_path))}', check=False)
    else:
        print("Final video at:", video_path)

def shutil_which(cmd):
    return subprocess.call(f"which {shlex.quote(cmd)} >/dev/null 2>&1", shell=True) == 0

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="YouTube -> Hindi-dubbed video (per-segment TTS, keeps sync).")
    ap.add_argument("url", help="YouTube URL")
    ap.add_argument("--whisper", default="small", help="Whisper model: tiny|base|small|medium|large")
    ap.add_argument("--tts", choices=["gtts", "coqui"], default="gtts", help="TTS engine (gtts=online, coqui=offline)")
    ap.add_argument("--speaker-wav", default=None, help="(optional) WAV for XTTS voice cloning")
    ap.add_argument("--outdir", default="out_hindi_video", help="output directory")
    ap.add_argument("--play", action="store_true", help="auto-play final video if mpv/vlc present")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        video_path = outdir / "downloaded_video.mp4"
        audio_wav = td / "extracted.wav"
        final_hindi_wav = outdir / "final_hindi.wav"
        final_video = outdir / "video_hindi.mp4"

        # 1) download video
        video_file = download_video(args.url, video_path)

        # 2) extract audio for ASR
        extract_audio(video_file, audio_wav)

        # 3) transcribe segments
        segments, full_asr = transcribe_segments(audio_wav, model_size=args.whisper)

        if not segments:
            print("No segments found in transcription. Exiting.")
            return

        # Save ASR for debugging
        (outdir / "whisper_full.json").write_text(json.dumps(full_asr, ensure_ascii=False, indent=2))

        # 4) translate per segment
        trans_segments = translate_segments(segments)
        (outdir / "segments_translated.json").write_text(json.dumps(trans_segments, ensure_ascii=False, indent=2))

        # 5) generate TTS per segment
        tts_produced = []
        if args.tts == "gtts":
            tts_produced = generate_tts_segments_gtts(trans_segments, td)
        else:
            tts_produced = generate_tts_segments_coqui(trans_segments, td, speaker_wav=args.speaker_wav)

        if not tts_produced:
            raise RuntimeError("No TTS segments produced.")

        # 6) assemble TTS segments into one aligned Hindi wav
        assembled = assemble_segments_to_wav(tts_produced, final_hindi_wav, allow_time_stretch=True)

        # 7) merge back into video
        merge_audio_into_video(video_file, assembled, final_video)

        print("\n✅ Done! Final dubbed video:", final_video)
        if args.play:
            # try to play
            if shutil_which("mpv"):
                run(f'mpv --force-window=yes {shlex.quote(str(final_video))}', check=False)
            elif shutil_which("vlc"):
                run(f'vlc --play-and-exit {shlex.quote(str(final_video))}', check=False)
            else:
                print("mpv/vlc not found. open the file manually.")

if __name__ == "__main__":
    main()


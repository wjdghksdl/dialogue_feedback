import os
import tempfile
import numpy as np
import librosa
import logging
import traceback
import json

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# NLP
from transformers import pipeline
import spacy

# Whisper
try:
    import whisper as _whisper_module
except ImportError:
    _whisper_module = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dialogue_feedback")

# === 설정 ===
LANG = "ko"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# spaCy 로드
nlp = None
try:
    nlp = spacy.load("ko_core_news_sm")
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        pass

# Whisper 모델 로드
whisper_model = None
def ensure_whisper_model():
    global whisper_model
    if whisper_model is not None: return whisper_model
    if _whisper_module is None: return None
    try:
        whisper_model = _whisper_module.load_model(WHISPER_MODEL_NAME)
        return whisper_model
    except Exception:
        return None

app = FastAPI(title="Voice Feedback AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [유틸] 오디오 변환 ---
def save_upload_to_wav(upload_file: UploadFile, target_rate=16000):
    suffix = os.path.splitext(upload_file.filename)[1].lower() or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload_file.file.read())
        tmp_in = f.name
    
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        audio = AudioSegment.from_file(tmp_in)
        audio = audio.set_frame_rate(target_rate).set_channels(1)
        audio.export(tmp_out, format="wav")
        return tmp_out
    finally:
        if os.path.exists(tmp_in): os.unlink(tmp_in)

# --- [핵심] 분석 로직 (기획안 반영) ---

def analyze_prosody_details(y, sr):
    """
    6-2 Prosody 분석: 속도, 음높이, 억양, 리듬, 휴지기(습관) 분석
    """
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. 음량 (RMS)
    rms = librosa.feature.rms(y=y)[0]
    
    # 2. 속도 (Onset & Tempo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # 3. 음높이 (Pitch/F0)
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    
    # 4. 습관 분석 (묵음 구간 비율 - 주저함 감지)
    # RMS가 특정 임계값보다 낮은 구간의 비율 계산
    silence_threshold = 0.01
    silence_ratio = np.sum(rms < silence_threshold) / len(rms)

    return {
        "duration": float(duration),
        "bpm": float(tempo),
        "pitch_std": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0,
        "silence_ratio": float(silence_ratio),  # 0.0 ~ 1.0 (높으면 말을 자주 멈춤)
    }

def analyze_linguistic_structure(text):
    """
    6-3 NLP 분석: 발화 구조, 논리성(길이 기반 추정)
    """
    if not text: return {"word_count": 0, "avg_sent_len": 0}
    
    doc = nlp(text) if nlp else None
    words = text.split()
    sentences = list(doc.sents) if doc else [text]
    
    avg_len = len(words) / len(sentences) if sentences else 0
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sent_len": avg_len  # 문장당 평균 단어 수 (구조적 복잡도)
    }

def calculate_advanced_scores(prosody, nlp_data, role):
    """
    3-1, 3-2: 상황별(Role) 가중치 적용 및 5대 분야 점수 산출
    """
    scores = {}
    
    # === 기준값 설정 (상황별 다르게 적용) ===
    if role == "면접":
        target_bpm = 100    # 면접은 침착해야 함
        penalty_speed = 1.5 # 빠르면 감점 큼
        target_silence = 0.15 # 적당한 끊김은 생각하는 모습
    elif role == "발표":
        target_bpm = 110    # 발표는 에너지
        penalty_speed = 1.0
        target_silence = 0.10 # 발표는 끊기면 안됨
    else: # 일반대화
        target_bpm = 120    # 대화는 티키타카
        penalty_speed = 0.8
        target_silence = 0.20 # 대화는 좀 끊겨도 됨
        
    # 1. [속도] (Speed)
    bpm = prosody["bpm"]
    diff_bpm = abs(bpm - target_bpm)
    scores["speed"] = max(0, 100 - (diff_bpm * penalty_speed))
    
    # 2. [음정/억양] (Intonation) - 변화가 너무 없으면(단조로움) 감점
    pitch_var = prosody["pitch_std"]
    if pitch_var < 10: scores["pitch"] = 50 # 로봇 같음
    elif pitch_var > 50: scores["pitch"] = 90 # 생동감
    else: scores["pitch"] = 70 + (pitch_var / 2)
    scores["pitch"] = min(100, scores["pitch"])

    # 3. [대화 습관] (Habit) - 묵음(버벅임) 비율 역산
    silence = prosody["silence_ratio"]
    habit_score = 100 - (abs(silence - target_silence) * 200)
    scores["habit"] = max(0, min(100, habit_score))

    # 4. [구조적 안정성] (Structure) - 문장 길이 적절성
    avg_len = nlp_data["avg_sent_len"]
    # 너무 짧거나(단답) 너무 길면(장황) 감점
    if 5 <= avg_len <= 15: scores["structure"] = 95
    else: scores["structure"] = max(40, 100 - abs(avg_len - 10) * 5)

    # 5. [청자 편의성] (Listener Comfort) - 위 요소들의 조화
    # 속도가 적절하고 버벅임이 없을수록 편안함
    scores["comfort"] = (scores["speed"] * 0.4) + (scores["habit"] * 0.4) + (scores["pitch"] * 0.2)
    
    # [종합 점수]
    scores["overall"] = sum(scores.values()) / 5
    
    return {k: round(v) for k, v in scores.items()}

def generate_detailed_feedback(scores, role):
    """
    7-1: 종합 피드백 생성
    """
    txt = f"[{role} 모드 분석 결과]\n"
    
    # 강점
    high_score = max(scores, key=scores.get)
    if high_score == "speed": txt += "✅ 말하는 속도가 상황에 아주 적절합니다.\n"
    elif high_score == "pitch": txt += "✅ 목소리의 억양이 생동감 있어 지루하지 않습니다.\n"
    elif high_score == "structure": txt += "✅ 문장의 길이가 적절하여 논리적으로 들립니다.\n"
    elif high_score == "habit": txt += "✅ 불필요한 추임새나 공백 없이 깔끔하게 말합니다.\n"
    
    # 약점 및 개선
    low_score = min(scores, key=scores.get)
    if low_score == "speed": txt += "❗ 말하기 속도 조절이 필요합니다. 조금 더 호흡을 가다듬어 보세요.\n"
    elif low_score == "pitch": txt += "❗ 톤이 다소 단조롭습니다. 중요한 단어에 강세를 줘보세요.\n"
    elif low_score == "habit": txt += "❗ 중간중간 공백이 깁니다. 자신감 있게 이어 말하는 연습이 필요합니다.\n"
    elif low_score == "structure": txt += "❗ 문장이 너무 짧거나 너무 깁니다. 간결하고 명확하게 끝맺는 연습을 해보세요.\n"
    
    return txt

# --- [API 엔드포인트] ---
@app.post("/upload_audio")
async def upload_audio_endpoint(file: UploadFile = File(...), role: str = Form("일반대화")):
    tmp_wav = None
    try:
        # 1. 전처리 (WAV 변환)
        tmp_wav = save_upload_to_wav(file)
        y, sr = librosa.load(tmp_wav, sr=16000)
        
        # 2. 각 분야별 상세 분석
        transcript = ""
        model = ensure_whisper_model()
        if model:
            res = model.transcribe(tmp_wav, language=LANG)
            transcript = res.get("text", "").strip()
        
        prosody_data = analyze_prosody_details(y, sr)
        nlp_data = analyze_linguistic_structure(transcript)
        
        # 3. 상황별 점수 산출
        scores = calculate_advanced_scores(prosody_data, nlp_data, role)
        
        # 4. 피드백 생성
        feedback = generate_detailed_feedback(scores, role)
        
        return JSONResponse({
            "prosody": prosody_data,
            "nlp": nlp_data,
            "scores": scores, # 여기에 speed, pitch, habit, structure, comfort 다 들어감
            "feedback": feedback
        })
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp_wav and os.path.exists(tmp_wav): os.unlink(tmp_wav)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
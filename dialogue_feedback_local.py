import os
import tempfile
import uuid
import numpy as np
import librosa
import soundfile as sf
import json
import logging
import traceback
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# NLP/LLM (옵션)
from transformers import pipeline
import spacy

# Whisper는 지연로딩(사용시 로드)
try:
    import whisper as _whisper_module
except Exception:
    _whisper_module = None

# === 로깅 설정 ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dialogue_feedback_local")

# === 설정 ===
LANG = "ko"  # 전사 언어 설정
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# spaCy 로드
try:
    nlp = spacy.load("ko_core_news_sm")
    logger.info("spaCy ko_core_news_sm loaded")
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm loaded")
    except Exception:
        nlp = None
        logger.info("spaCy model not loaded; NLP features limited")

# Whisper 모델 핸들 (지연로딩)
whisper_model = None

def ensure_whisper_model():
    global whisper_model
    if whisper_model is not None:
        return whisper_model
    if _whisper_module is None:
        logger.error("whisper 패키지가 설치되어 있지 않습니다.")
        return None
    try:
        logger.info(f"Loading whisper model '{WHISPER_MODEL_NAME}' ...")
        whisper_model = _whisper_module.load_model(WHISPER_MODEL_NAME)
        logger.info("whisper model loaded")
        return whisper_model
    except Exception as e:
        logger.error(f"whisper model load failed: {str(e)}")
        whisper_model = None
        return None

app = FastAPI(title="Dialogue Correction AI - Local Whisper Prototype")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- [1] 오디오 처리 (들여쓰기 오류 수정됨) ----------------------
def save_upload_to_wav(upload_file: UploadFile, target_rate=16000):
    tmp_in = None
    tmp_out = None
    try:
        suffix = os.path.splitext(upload_file.filename)[1].lower() or ".tmp"
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        content = upload_file.file.read()
        tmp_in.write(content)
        tmp_in.flush()
        
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        # pydub를 사용해 변환
        audio = AudioSegment.from_file(tmp_in.name)
        audio = audio.set_frame_rate(target_rate).set_channels(1)
        audio.export(tmp_out.name, format="wav")
        
        logger.info(f"Saved uploaded audio to {tmp_out.name}")
        return tmp_out.name
    except Exception as e:
        logger.error(f"save_upload_to_wav error: {str(e)}")
        # 파일 정리 (안전한 문법으로 변경)
        if tmp_in:
            try:
                os.unlink(tmp_in.name)
            except Exception:
                pass
        if tmp_out:
            try:
                os.unlink(tmp_out.name)
            except Exception:
                pass
        raise
    finally:
        if tmp_in:
            try:
                tmp_in.close()
            except Exception:
                pass

def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

# ---------------------- [2] Whisper 전사 ----------------------
def transcribe(wav_path):
    model = ensure_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(wav_path, language=LANG)
        text = result.get("text", "").strip()
        return text
    except Exception as e:
        logger.error(f"Whisper transcription failed: {str(e)}")
        return ""

# ---------------------- [3] Prosody (음성 분석) ----------------------
def analyze_prosody(y, sr):
    metrics = {}
    try:
        duration = len(y) / sr
        metrics['duration_sec'] = duration

        # RMS (음량)
        rms = librosa.feature.rms(y=y)[0]
        metrics['rms_mean'] = float(np.mean(rms))
        metrics['rms_std'] = float(np.std(rms))

        # Tempo (속도)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            if hasattr(librosa.feature, 'rhythm'):
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
            else:
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            metrics['tempo_bpm'] = float(tempo[0])
        except Exception:
            metrics['tempo_bpm'] = None

        # Pitch (억양)
        try:
            f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            metrics['f0_std_hz'] = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
            metrics['f0_mean_hz'] = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        except Exception:
            metrics['f0_std_hz'] = 0.0
            metrics['f0_mean_hz'] = 0.0

        # 침묵/발화 구간
        intervals = librosa.effects.split(y, top_db=30)
        speech_durations = [(end - start) / sr for start, end in intervals]
        metrics['speech_total_sec'] = sum(speech_durations)
        metrics['silence_total_sec'] = duration - metrics['speech_total_sec']
        
    except Exception as e:
        logger.error(f"analyze_prosody error: {str(e)}")
    return metrics

# ---------------------- [4] NLP 분석 (CNN 오류 방지 적용) ----------------------
def analyze_nlp(text):
    out = {}
    
    # 텍스트가 비었거나 너무 짧으면 분석 중단 (CNN 오류 방지)
    if not text or len(text.strip()) < 2:
        return {
            "raw_text": "",
            "summary": "대화 내용이 감지되지 않았습니다. (목소리가 너무 작거나 잡음이 많을 수 있습니다)",
            "word_count": 0,
            "sentence_count": 0,
            "pos_counts": {}
        }

    try:
        out['raw_text'] = text
        words = text.split()
        out['word_count'] = len(words)
        out['char_count'] = len(text)

        if nlp:
            doc = nlp(text)
            out['sentence_count'] = len(list(doc.sents))
            pos_counts = {}
            for tok in doc:
                pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
            out['pos_counts'] = pos_counts
        else:
            out['sentence_count'] = 0
            out['pos_counts'] = {}

        # 요약 (텍스트가 50자 이상일 때만 수행)
        if len(text) > 50:
            try:
                summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")
                input_len = len(words)
                max_len = max(20, int(input_len * 0.6))
                summary_result = summarizer(text, max_length=max_len, min_length=10)
                out['summary'] = summary_result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                out['summary'] = "요약 실패"
        else:
            out['summary'] = text 

    except Exception as e:
        logger.error(f"analyze_nlp error: {str(e)}")
        out['raw_text'] = text
        out['summary'] = "분석 중 오류"
    
    return out

# ---------------------- [5] 점수 계산 ----------------------
def evaluate_all(prosody_metrics, nlp_metrics):
    scores = {}
    try:
        # 1. 속도 점수
        tempo = prosody_metrics.get('tempo_bpm', 0)
        if not tempo:
            scores['speed'] = 50
        else:
            if 80 <= tempo <= 160:
                scores['speed'] = 90
            else:
                diff = min(abs(tempo - 80), abs(tempo - 160))
                scores['speed'] = max(40, 90 - diff * 0.5)

        # 2. 명확성
        rms_std = prosody_metrics.get('rms_std', 0)
        scores['clarity'] = min(100, max(50, rms_std * 500))

        # 3. 억양
        f0_std = prosody_metrics.get('f0_std_hz', 0)
        scores['intonation'] = min(100, max(40, f0_std * 2))

        # 4. 구조
        sent_count = nlp_metrics.get('sentence_count', 0)
        if sent_count > 0:
            scores['structure'] = 80 
        else:
            scores['structure'] = 40

        # 5. 청중 친화성
        scores['audience_friendliness'] = (scores['clarity'] + scores['intonation']) / 2

        # 종합 점수
        total = (
            scores['speed'] * 0.2 + 
            scores['clarity'] * 0.2 + 
            scores['intonation'] * 0.2 + 
            scores['structure'] * 0.2 + 
            scores['audience_friendliness'] * 0.2
        )
        scores['overall'] = round(total, 1)

        # 소수점 정리
        for k, v in scores.items():
            scores[k] = round(v, 1)

    except Exception as e:
        logger.error(f"evaluate_all error: {str(e)}")
        scores = {'overall': 0}
    return scores

# ---------------------- [6] 사용자 피드백 생성 (화면 연결용) ----------------------
def generate_friendly_feedback(prosody, nlp, scores):
    """
    LLM 없이 로직 기반으로 친절한 피드백 메시지를 생성합니다.
    """
    # 텍스트 인식 실패 시
    if not nlp.get('raw_text'):
        return "⚠️ 음성이 명확하게 인식되지 않았습니다. 마이크를 조금 더 가까이 대고 말씀해 주세요."

    total = scores.get('overall', 0)
    
    # 점수대별 멘트
    if total >= 80:
        base_comment = "👏 와우! 전달력이 매우 뛰어난 스피치입니다."
    elif total >= 60:
        base_comment = "👍 좋은 편이에요! 조금만 더 자신감 있게 말해보세요."
    else:
        base_comment = "💪 목소리 톤이나 속도를 조절해서 전달력을 높여보세요."

    # 세부 조언 추가
    advice = []
    tempo = prosody.get('tempo_bpm', 0)
    if tempo and (tempo > 160):
        advice.append("말이 조금 빠릅니다. 천천히 말해보세요.")
    elif tempo and (tempo < 80):
        advice.append("말이 조금 느립니다. 리듬감을 살려보세요.")

    if prosody.get('f0_std_hz', 0) < 10:
        advice.append("목소리 톤이 다소 단조롭습니다. 억양을 넣어보세요.")

    if advice:
        return f"{base_comment} ({' '.join(advice)})"
    else:
        return base_comment

# ---------------------- [API] 엔드포인트 ----------------------
@app.post('/upload_audio')
async def upload_audio(file: UploadFile = File(...), role: str = '일반대화'):
    tmp_wav = None
    try:
        tmp_wav = save_upload_to_wav(file)
        y, sr = load_audio(tmp_wav)
        
        transcript = transcribe(tmp_wav)
        prosody = analyze_prosody(y, sr)
        nlp_metrics = analyze_nlp(transcript)
        scores = evaluate_all(prosody, nlp_metrics)
        
        # [수정] 프론트엔드가 'feedback' 필드를 화면에 뿌려준다고 가정하고
        # 여기에 친절한 멘트를 넣습니다.
        friendly_comment = generate_friendly_feedback(prosody, nlp_metrics, scores)

        # [중요] 프론트엔드 호환성을 위해 키 이름(transcript, prosody 등)을 원래대로 유지합니다.
        result = {
            "transcript": transcript,
            "prosody": prosody,
            "nlp": nlp_metrics,
            "scores": scores,
            "feedback": friendly_comment  # LLM 경고 메시지 대신 유용한 피드백 전달
        }

        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"upload_audio handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": "서버 오류 발생", "detail": str(e)}, status_code=500)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try: os.unlink(tmp_wav)
            except: pass

@app.get('/health')
async def health():
    return {"status": "ok"}
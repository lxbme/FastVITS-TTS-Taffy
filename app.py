import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import soundfile as sf
import io
from typing import Optional
import uvicorn
import base64
from text import text_to_sequence, _clean_text

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Language configurations
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}

class TTSRequest(BaseModel):
    text: str
    speaker: str
    language: Optional[str] = "日本語"
    speed: Optional[float] = 1.0

class TTSResponse(BaseModel):
    message: str
    audio_base64: str

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text: str, speaker: str, language: str, speed: float):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        
        if speaker not in speaker_ids:
            raise HTTPException(status_code=400, detail=f"Invalid speaker. Available speakers: {list(speaker_ids.keys())}")
            
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0 / speed,
            )[0][0, 0].data.cpu().float().numpy()
            
        # Convert audio to WAV format in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio, hps.data.sampling_rate, format='WAV')
        buffer.seek(0)
        
        # Encode audio data to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return audio_base64

    return tts_fn

def create_app(model_path: str, config_path: str):
    # Load model and configs
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()
    
    utils.load_checkpoint(model_path, net_g, None)
    speaker_ids = hps.speakers
    
    # Create FastAPI app
    app = FastAPI(title="TTS API Service")
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    
    @app.get("/speakers")
    async def get_speakers():
        return {"speakers": list(speaker_ids.keys())}
    
    @app.get("/languages")
    async def get_languages():
        return {"languages": list(language_marks.keys())}
    
    @app.post("/tts", response_model=TTSResponse)
    async def text_to_speech(request: TTSRequest):
        try:
            audio_base64 = tts_fn(
                request.text,
                request.speaker,
                request.language,
                request.speed
            )
            return TTSResponse(
                message="Success",
                audio_base64=audio_base64
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    return app

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./OUTPUT_MODEL/G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./OUTPUT_MODEL/config.json", help="directory to your model config file")
    parser.add_argument("--host", default="0.0.0.0", help="host to run the server on")
    parser.add_argument("--port", default=2333, type=int, help="port to run the server on")
    
    args = parser.parse_args()
    
    app = create_app(args.model_dir, args.config_dir)
    uvicorn.run(app, host=args.host, port=args.port)
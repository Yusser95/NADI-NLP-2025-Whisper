import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel, PeftConfig
from evaluate import load
from tqdm import tqdm
import librosa
import re
import os

## https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard/blob/main/models/whisper.py
def normalize_arabic_text(text):
    """
    Arabic text normalization:
    1. Remove punctuation
    2. Remove diacritics
    3. Eastern Arabic numerals to Western Arabic numerals

    Arguments
    ---------
    text: str
        text to normalize
    Output
    ---------
    normalized text
    """
    # Remove punctuation
    punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)

    # Remove diacritics
    diacritics = r'[\u064B-\u0652]'  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)
    
    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)   

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    return text.strip()



# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 # if torch.cuda.is_available() else torch.float32


processor_id = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(processor_id)


# Evaluation metric
wer_metric = load("wer")



dialects = ["Mauritania", "Algeria", "Morocco", "Yemen", "UAE", "Egypt", "Palestine", "Jordan"]

base_dir = "/pfss/mlde/workspaces/mlde_wsp_P_DFKI_Darmstadt/ya98xoke/tests/badr"



use_peft = False
run_base = False

save_path = "./results/test/full"
if use_peft:
    save_path = "./results/test/peft"
if run_base:
    save_path = "./results/test/base"


for d in dialects:

    if run_base:
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
    if use_peft:
        model_id = f"Yusser/whisper-dialect-Mauritania_Algeria_Morocco_Yemen_UAE_Egypt_Palestine_Jordan-ft-first-stage"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        
        peft_model_id = f"Yusser/whisper-dialect-{d}-ft-second-stage-tuning-lora-cer"
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        model = PeftModel.from_pretrained(model, peft_model_id)
    else:
        model_id = f"Yusser/whisper-dialect-{d}-ft-second-stage-tuning-cer"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)


    #dataset = load_dataset("UBC-NLP/Casablanca", d, split="test")
    dataset = load_dataset("UBC-NLP/NADI2025_subtask2_ASR_Test",d, split="test")
    print(dataset)

    #path = os.path.join(base_dir,"./nadi2025_datasets", d ) # "Yemen")
    #dataset = load_from_disk(path, keep_in_memory=True)["validation"]
    
    # Initialize lists to store results
    all_transcriptions = []
    all_references = []

    # Inference loop
    for batch in tqdm(dataset, desc="Evaluating..."):
        # Extract audio data (NumPy array) and sampling rate from batch
        audio_data = batch["audio"]["array"]
        sampling_rate = batch["audio"]["sampling_rate"]

        # Check if the audio data is empty
        if audio_data.size == 0:
            print(f"Skipping empty audio for {batch}")
            all_references.extend([""])
            all_transcriptions.extend([""])
            continue  # Skip this batch if audio is empty

        # Resample the audio to 16 kHz if needed
        if sampling_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=16000)

        # Prepare the input features
        input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device, dtype=torch_dtype)

        # Generate predictions
        pred_ids = model.generate(input_features, max_new_tokens=400) #,language="ar")
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)

        # Collect transcriptions and references
        all_transcriptions.extend(pred_text)
        if "transcription" in batch:
            all_references.extend([batch["transcription"]])
        else:
            all_references.extend(pred_text)
        
        
    # Normalize and compute WER
    all_transcriptions_norm = [normalize_arabic_text(text) for text in all_transcriptions]
    all_references_norm = [normalize_arabic_text(text) for text in all_references]
    wer = 100 * wer_metric.compute(predictions=all_transcriptions_norm, references=all_references_norm)
    print(f"{d} - Word Error Rate (WER): {wer:.2f}%")
    
    os.makedirs(res_dir,exist_ok=True)
    with open(os.path.join(res_dir,f"prediction_{d.lower()}.txt"), "w", encoding="utf-8") as f:
        for ref ,pred in tqdm(zip(all_references,all_transcriptions)):
            f.write(pred)
            f.write("\n")


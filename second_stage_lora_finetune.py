from datasets import load_dataset, DatasetDict, load_from_disk
from datasets import concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig


import os
import sys

import torch

model_id = "openai/whisper-large-v3"
model_id_ft = "Yusser/whisper-dialect-Mauritania_Algeria_Morocco_Yemen_UAE_Egypt_Palestine_Jordan-ft-first-stage"

language = "arabic"
use_peft = False
# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import re

import wandb
wandb.login()

def pre_process(text):
    # Step 1: Remove brackets and content inside
    text = re.sub(r"\[.*?\]", "", text)

    # Step 2: Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text



dialects = ["Mauritania", "Algeria", "Morocco", "Yemen", "UAE", "Egypt", "Palestine", "Jordan"]
base_dir = "./"

cwd = os.getcwd()
print("cwd",cwd)



from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_id)  
generation_config = model.generation_config  # ✅ this has `lang_to_id`, `task_to_id`, etc.


for i in range(0,len(dialects)):



    train_languages = [dialects[i]]

    wandb.init(project="my-whisper-fine-tuning", name=f"whisper-dialect-{'_'.join(train_languages)}-ft-second-stage-tuning-lora-cer")


    train_datasets = []

    for l in train_languages:
        print(l)
        path = os.path.join(base_dir,"./nadi2025_datasets", l ) # "Yemen")
        train_datasets.append(load_from_disk(path, keep_in_memory=True))


    # Example with multiple DatasetDicts
    combined_train = concatenate_datasets([ds["train"] for ds in train_datasets], axis=0).shuffle(seed=42)     #[:10]
    combined_eval = concatenate_datasets([ds["validation"] for ds in train_datasets], axis=0).shuffle(seed=42) #[:10]

    # Build a new DatasetDict
    dataset = DatasetDict({
        "train": combined_train,
        "validation": combined_eval
    })


    print(dataset)

    # Function to map
    def preprocess_transcription(batch):
        batch["transcription"] = pre_process(batch["transcription"])
        return batch

    # Assuming datasets is a DatasetDict with "train" and "validation"
    dataset["train"] = dataset["train"].map(preprocess_transcription)
    dataset["validation"] = dataset["validation"].map(preprocess_transcription)


   
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language, task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id) #, device=device)

    input_str = dataset["train"][0]["transcription"]
    labels = tokenizer(input_str).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")

    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id, language=language, task="transcribe")


    from datasets import Audio

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #.to(device)


    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"]) #, num_proc=4)

    




    import torch

    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt") # .to(device)

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt") # .to(device)

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    import evaluate

    metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)


        #pred_str = [normalize_arabic_text(t) for t in pred_str] # .cpu().detach().numpy()
        #label_str = [normalize_arabic_text(t) for t in label_str] # .cpu().detach().numpy()

        cer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}



    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model = WhisperForConditionalGeneration.from_pretrained(model_id_ft) 
    model.generation_config = generation_config 
    model.generation_config.language = language
    model.generation_config.task = "transcribe"

    model.generation_config.forced_decoder_ids = None
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # LoRA config
    peft_config = LoraConfig(
        #task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.5,
        bias="none",
        #inference_mode=False,
        target_modules=["q_proj", "v_proj"]  # <- required for Whisper
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    #äsys.exit()

    from transformers import Seq2SeqTrainingArguments



    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(base_dir, f"tests/badr/whisper-dialect-{'_'.join(train_languages)}-ft-second-stage-tuning-lora-cer"),  # change to a repo name of your choice
        per_device_train_batch_size=8, #16,
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        #num_train_epochs=6,
        max_steps=2000,
        #gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        generation_max_length=125, #225
        save_steps=1000,
        eval_steps=200,
        logging_steps=25,
        #report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=True,

        predict_with_generate=True, # fasle for peft
        remove_unused_columns=False,  # required for PeftModel
        label_names=["labels"]       # required for PeftModel

    )
    model.config.use_cache = False

    from transformers import Seq2SeqTrainer


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.model.config.model_input_names = ["input_features"]
    trainer.train()

    model = trainer.model

    wandb.finish()


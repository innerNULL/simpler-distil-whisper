# -*- coding: utf-8 -*-
# file: run_pseudo_labelling.py
# date: 2024-03-08
# 
# Usage:
# python ./run_pseudo_labelling.py ./run_pseudo_labelling.json
#
# Notes:
# The naming of pseudo labeled text field maybe be a little confusing here.
# In most case, we only do pseudo labeling on training data, and using 
# origin dev/test data for evaluation purpose to check if distil-whisper's
# performance continuouly improving.
# So we need keep pseudo labeled data have sime schema with original 
# datasets, and so we replace `target_text_col` with teacher model 
# generated text in pseudo labeled dataset, and put the value of original 
# `target_text_col` into a new fields called `origin_target_text_col`.


import pdb
import sys
import os
import json
import torch
import torchaudio
from tqdm import tqdm
from opencc import OpenCC
from torch import Tensor
from typing import Dict, List, Optional
from datasets import Audio
from transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torchmetrics.text import CharErrorRate

from lib import audio_file2model_inputs


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    model_name: str = configs["model"]
    processor_name: str = configs["processor"]
    data_path: str = configs["data_path"]
    lang: str = configs["lang"]
    output_path: str = configs["output_path"]
    print("Output at '%s'" % output_path)
    device: torch.device = torch.device(configs["device"])
    max_sample_size: int = configs["max_sample_size"]
    target_text_col: str = configs["target_text_col"]
    origin_text_col: str = "origin_%s" % configs["target_text_col"]
    metric_col: str = configs["metric_col"]
    metric_to_use: str = configs["metric_to_use"]

    dataset: List[Dict] = [
        json.loads(x) for x in open(data_path, "r").read().split("\n")
        if x not in {"", " "}
    ]

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        processor_name, language=lang, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang, task="transcribe"
    )
 
    results: List[Dict] = []
    target_sampling_rate: int = 16000
    for sample in tqdm(dataset):
        # Backup original target text
        sample[origin_text_col] = sample[target_text_col]

        inputs: Tensor = None
        inputs, _ = audio_file2model_inputs(
            sample["path"], processor, target_sampling_rate, configs["device"]
        )
        output_ids: List[int] = model.generate(inputs).to("cpu").tolist()[0]
        output_text: str = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        sample[target_text_col] = output_text

        if lang in {"mandarin", "zh-TW", "zh-CN", "zh"}:
            converter: OpenCC = OpenCC('tw2s.json')
            sample[origin_text_col] = converter.convert(sample[origin_text_col])
            sample[target_text_col] = converter.convert(sample[target_text_col])
        
        if metric_to_use == "cer":
            sample[metric_col] = CharErrorRate()(
                sample[target_text_col], sample[origin_text_col]
            ).to("cpu").tolist()
        else:
            raise Exception("Currently not support metrics '%s'" % metric_to_use)
        
        results.append(sample)

    out_file = open(output_path, "w")
    for sample in results:
        #print(sample)
        out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    out_file.close()
    print("Inference results are dumped at: %s" % output_path)



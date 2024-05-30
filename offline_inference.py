# -*- coding: utf-8 -*-
# file: inference.py
#
# python ./bin/model/whisper_and_distil_whisper/offline_inference.py ./demo_configs/model/whisper_and_distil_whisper/offline_inference.json


import pdb
import sys
import os
import json
import torch
import torchaudio
from transformers import pipeline
from tqdm import tqdm
from opencc import OpenCC
from torch import Tensor
from typing import Dict, List, Optional
from datasets import Audio
from transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torchmetrics.text import CharErrorRate

from lib import audio_file2model_inputs


def eval(
    dataset: List[Dict], outputs_col: str, targets_col: str, lang: str
) -> Dict:
    if lang not in {"mandarin"}:
        raise "So far not support evaluation for language '%s'" % lang
    metric_name: str = ""
    metric: Optional[CharErrorRate] = None
    converter: Optional[OpenCC] = None

    if lang == "mandarin":
        metric_name = "cer"
        metric = CharErrorRate()

    targets: List[str] = [x[targets_col] for x in dataset]
    outputs: List[str] = [x[outputs_col] for x in dataset]
    assert len(targets) == len(outputs)
    
    retults: Dict = {metric_name: float(metric(outputs, targets))}
    print(retults)
    return retults


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    model_name: str = configs["model"]
    processor_name: str = configs["processor"]
    data_path: str = configs["data_path"]
    lang: str = configs["lang"]
    audio_path_col: str = configs["audio_path_col"]
    output_text_col: str = configs["output_text_col"]
    output_path: str = configs["output_path"]
    print("Output at '%s'" % output_path)
    device: torch.device = torch.device(configs["device"])
    max_sample_size: int = configs["max_sample_size"]
    groundtruth_col: str = configs["groundtruth_col"]
    use_hf_pipeline: bool = configs["use_hf_pipeline"]

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
    inf_pipeline = None 
    if use_hf_pipeline:
        inf_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name, 
            tokenizer=processor.tokenizer, 
            feature_extractor=processor.feature_extractor, 
            chunk_length_s=30, return_timestamps=False
        )
 
    results: List[Dict] = []
    target_sampling_rate: int = 16000
    for sample in tqdm(dataset):
        output_text: str = ""
        if inf_pipeline:
            output_text = inf_pipeline(
                sample[audio_path_col], generate_kwargs={"language": lang}
            )["text"]
        else:
            inputs: Tensor = None
            inputs, _ = audio_file2model_inputs(
                sample[audio_path_col], processor, target_sampling_rate, configs["device"]
            ) 
            output_ids: List[int] = model.generate(inputs).to("cpu").tolist()[0]
            output_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
         
        output_text = output_text if lang != "mandarin" else OpenCC('tw2s.json').convert(output_text)
        sample[output_text_col] = output_text
        results.append(sample)
    
    if groundtruth_col != "":
        for x in results:
            x[groundtruth_col] = \
                x[groundtruth_col] if lang != "mandarin" \
                else OpenCC('tw2s.json').convert(x[groundtruth_col])
        eval(results, output_text_col, groundtruth_col, lang) 

    out_file = open(output_path, "w")
    for sample in results:
        out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    out_file.close()
    print("Inference results are dumped at: %s" % output_path)



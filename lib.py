# -*- coding: utf-8 -*-
# file: lib.py 
# date: 2024-03-09


import pdb
import sys
import os
import json
import torch
import torchaudio
import augly.audio as audaugs
from tqdm import tqdm
from datasets import load_dataset
from torch import nn
from typing import List, Dict, Callable, Optional, Union, Any, Tuple
from datasets import DatasetDict
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers.feature_extraction_utils import BatchFeature
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Module
from torchmetrics import Metric
from torchmetrics.text import CharErrorRate, WordErrorRate
from augly.audio import Compose, OneOf
from torchaudio.transforms import FrequencyMasking, TimeMasking
from opencc import OpenCC


AUGLY_TRANSFORMS: Compose = audaugs.Compose([
    #audaugs.Clip(duration_factor=0.25),
    audaugs.AddBackgroundNoise(p=0.5),
    audaugs.ChangeVolume(volume_db=10.0, p=0.5),
    audaugs.OneOf(
        [audaugs.Speed(factor=3.0), audaugs.TimeStretch(rate=3.0)], 
        p=0.5
    ),
])


def spec_argument(
    spec: Union[List, Tensor],
    freq_before_time_axis: bool=True, 
    freq_masking_prob: float=0.9, 
    freq_max_masking_ratio: int=0.15, 
    time_masking_prob: float=0.9, 
    time_max_masking_ratio: int=0.05
) -> List:
    """
    This default hyper-parameters choosing are refer to 
    SpecArgument paper's recommend
    """
    if isinstance(spec, list):
        spec = Tensor(spec)
    if len(spec.shape) not in {2, 3}:
        raise Exception("Dim error")
    if len(spec.shape) == 2: 
        spec = spec.reshape(-1, spec.shape[0], spec.shape[1])
    if not freq_before_time_axis:
        spec = spec.reshape(-1, spec.shape[1], spec.shape[2])

    freq_dim: int = spec.shape[1] 
    time_dim: int = spec.shape[2]
    freq_max_masking_len: int = int(freq_dim * freq_max_masking_ratio)
    time_max_masking_len: int = int(time_dim * time_max_masking_ratio)

    if rd.random() < freq_masking_prob:
        freq_masking: FrequencyMasking = FrequencyMasking(
            freq_mask_param=freq_max_masking_len
        )
        spec = freq_masking(spec)

    if rd.random() < time_masking_prob:
        time_masking: TimeMasking = TimeMasking(
            time_mask_param=time_max_masking_len
        )
        spec = time_masking(spec)

    return spec.tolist()


class DataCollatorSpeechSeq2SeqWithPaddingV1:
    def __init__(self, 
        processor: Any, 
        tokenizer: Any=None,
        lang: str="mandarin",
        path_col: str="path", 
        text_col: str="text",
        audio_duration_col: str="input_length",
        model_input_col: str="input_features", 
        model_label_col: str="labels", 
        sample_id_col: str="",
        target_sample_rate: int=16000, 
        spec_argument: bool=True,
        freq_masking_prob: float=0.7, 
        freq_max_masking_ratio: float=0.1,
        time_masking_prob: float=0.7, 
        time_max_masking_ratio: float=0.1
    ):
        self.processor: Any = processor
        self.tokenizer: Any = self.processor.tokenizer if tokenizer is None else tokenizer
        self.lang: str = lang
        self.path_col: str = path_col
        self.text_col: str = text_col
        self.audio_duration_col: str = audio_duration_col
        self.model_input_col: str = model_input_col
        self.model_label_col: str = model_label_col
        self.sample_id_col: str = sample_id_col
        self.target_sample_rate: int = target_sample_rate
        self.spec_argument: bool = spec_argument
        self.freq_masking_prob: float = freq_masking_prob
        self.freq_max_masking_ratio: float = freq_max_masking_ratio
        self.time_masking_prob: float = time_masking_prob
        self.time_max_masking_ratio: float = time_max_masking_ratio

    def __call__(self, jsonl_samples: List[Dict]) -> Dict[str, Tensor]:
        train_samples: List[Dict] = [
            josnl_record2train_sample(
                x, self.processor,
                lang=self.lang,
                path_col=self.path_col, text_col=self.text_col, 
                model_input_col=self.model_input_col, 
                model_target_col=self.model_label_col, 
                audio_duration_col=self.audio_duration_col, 
                target_sample_rate=self.target_sample_rate
            ) for x in jsonl_samples
        ]

        input_features: List[Dict[str, Union[List[float], Tensor]]] = [
            {self.model_input_col: sample[self.model_input_col].tolist()[0]} 
            for sample in train_samples
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [
            {"input_ids": sample[self.model_label_col]} for sample in train_samples
        ]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        if self.sample_id_col not in {""}:
            batch[self.sample_id_col] = torch.tensor(
                [x[self.sample_id_col] for x in jsonl_samples], dtype=torch.int32
            ).reshape(len(jsonl_samples), 1)

        return batch


def audio_file2model_inputs(
    path: str, fea_extractor: WhisperProcessor,
    target_sample_rate: int=16000, device: str="cpu"
) -> Tuple[Tensor, int]:
    waveform: Tensor = None
    sample_rate: int = -1
    waveform, sample_rate = torchaudio.load(path)
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    duration_sec: int = waveform.shape[-1] / target_sample_rate
    inputs: Tensor = fea_extractor(
        waveform.squeeze(), sampling_rate=target_sample_rate,
        return_tensors="pt"
    ).input_features.to(torch.device(device))
    return (inputs, duration_sec)


def text2token_ids(text: str, fea_extractor: WhisperProcessor) -> List[List[int]]:
    out: Tensor = fea_extractor(audio=None, text=text)["input_ids"]
    return out


def josnl_record2train_sample(
    jsonl_sample: Dict, 
    fea_extractor: WhisperProcessor,
    lang: str="mandarin",
    path_col: str="path", 
    text_col: str="text", 
    model_input_col: str="input_features", 
    model_target_col: str="labels", 
    audio_duration_col: str="input_length",
    target_sample_rate: int=16000, 
    device: str="cpu"
) -> Dict[str, Union[Tensor, int, str]]:
    output: Dict[str, Union[Tensor, int, str]] = {}

    output[text_col] = jsonl_sample[text_col] 
    
    output[model_input_col] = None
    output[audio_duration_col] = None
    output[model_input_col], output[audio_duration_col] = \
        audio_file2model_inputs(
            path=jsonl_sample[path_col], 
            fea_extractor=fea_extractor, 
            target_sample_rate=target_sample_rate,
            device=device
        ) 

    output[model_target_col] = text2token_ids(
        text=text_force_simplified_chinese(jsonl_sample[text_col], lang),
        fea_extractor=fea_extractor,
    )
    return output


def datasetdict_load_jsonl(
    train_data_path: str, dev_data_path: str, test_data_path: str, 
    sample_id_col: str=""
) -> DatasetDict:
    print("Running dataset dict JSONL loader")
    dataset: DatasetDict = DatasetDict()

    if train_data_path is not None:
        dataset["train"] = load_dataset("json", data_files=train_data_path)["train"]
    if dev_data_path is not None:
        dataset["validation"] = load_dataset("json", data_files=dev_data_path)["train"]
    if test_data_path is not None:
        dataset["test"] = load_dataset("json", data_files=test_data_path)["train"]
    
    if sample_id_col not in {""}:
        def _add_sample_id(sample: Dict, idx: int) -> Dict:
            sample[sample_id_col] = idx
            return sample

        for split in dataset:
            dataset[split] = dataset[split].map(_add_sample_id, with_indices=True)

    return dataset


def audio_get_meta(
    path: str, 
    audio_path_col: str="path",
    audio_duration_col: str="duration_sec"
) -> Dict[str, Union[str, int, float]]:
    metadata: Dict[str, Union[str, int, float]] = {}
    metadata[audio_path_col] = path

    waveform: Tensor = None
    sample_rate: int = -1
    waveform, sample_rate = torchaudio.load(path)
    
    duration_sec: int = waveform.shape[-1] / sample_rate
    metadata[audio_duration_col] = duration_sec

    return metadata


def hf_datasetdict_load_audio_jsonl(
    train_data_path: Optional[str]=None, 
    dev_data_path: Optional[str]=None, 
    test_data_path: Optional[str]=None,
    sample_id_col: str="",
    audio_duration_col: str="audio_duration", 
    audio_path_col: str="path"
) -> DatasetDict:
    out: DatasetDict = datasetdict_load_jsonl(
        train_data_path, dev_data_path, test_data_path, 
        sample_id_col
    )

    def _append_audio_meta(sample: Dict) -> Dict:
        audio_meta: Dict = audio_get_meta(
            sample[audio_path_col], audio_path_col, audio_duration_col
        )
        sample[audio_duration_col] = audio_meta[audio_duration_col]
        return sample

    for split in out:
        out[split] = out[split].map(_append_audio_meta, num_proc=4)
    return out


def fn_gen_hf_dataset_filter_by_asr_data(
    tokenizer: Any, 
    min_audio_duration: float=10.0,
    max_audio_duration: float=30.0,
    min_token_num: int=0,
    max_token_num: int=512,
    audio_path_col: Optional[str]=None, text_col: Optional[str]=None
) -> Callable:
    def _filter(sample: Dict) -> bool:
        
        if audio_path_col is not None:
            audio_duration: float = audio_get_meta(
                sample[audio_path_col], "path", "duration"
            )["duration"]
            if audio_duration <= min_audio_duration \
                or audio_duration >= max_audio_duration:
                return False

        if text_col is not None:
            tokens: List[int] = tokenizer.encode(sample[text_col])
            tokens_num: int = len(tokens)
            if tokens_num <= min_token_num or tokens_num >= max_token_num:
                return False

        return True

    return _filter


def text_force_simplified_chinese(text: str, lang: str="") -> str:
    if lang.lower() not in {"mandarin", "zh-cn", "zh-tw", "zh"}:
        return text
    return OpenCC("tw2s.json").convert(text)

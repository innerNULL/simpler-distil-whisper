# -*- coding: utf-8 -*-
# file: run_distillation.py
# date: 2024-03-09
#
# Usage:
# python ./bin/model/distil_whisper/run_distillation.py ./demo_configs/model/distil_whisper/run_distillation.json


import pdb
import sys
import os
import json
import torch
import torchaudio
import augly.audio as audaugs
from tqdm import tqdm
from torch import nn
from typing import List, Dict, Callable, Optional
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

from lib import DataCollatorSpeechSeq2SeqWithPaddingV1 
from lib import hf_datasetdict_load_audio_jsonl
from lib import fn_gen_hf_dataset_filter_by_asr_data
from lib import text_force_simplified_chinese


def cal_cer_or_wer(targets: List[str], outputs: List[str], lang: str) -> float:
    metric: Optional[Metric] = None

    if lang.lower() in {"zh", "chinese", "mandarin", "zh-tw", "zh-cn"}:
        metric = CharErrorRate()
    else:
        metric = WordErrorRate()
    return metric(outputs, targets)


def get_parameter_names(
    model: Module, 
    forbidden_layer_types: List[Module], forbidden_module=None
) -> List[str]:
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    pdb.set_trace()
    return result


def kl_divergence(
    target_dist: Tensor, predicted_log_dist: Tensor, labels: Tensor
) -> Tensor:
    kl_loss: nn.KLDivLoss = nn.KLDivLoss(reduction="none")
    divergence: Tensor = kl_loss(predicted_log_dist, target_dist)
    # ignore padded tokens from divergence, i.e. where labels are not set to -100
    padding_mask: Tensor = (labels >= 0).unsqueeze(-1)
    divergence = divergence * padding_mask
    # take the average over the mini-batch
    divergence = divergence.sum() / padding_mask.sum()
    return divergence


def train_step(
    batch: BatchFeature, 
    teacher_model: WhisperForConditionalGeneration, 
    student_model: WhisperForConditionalGeneration,
    teacher_model_device: torch.device,
    student_model_device: torch.device,
    temperature: int=2.0,
) -> Dict[str, Tensor]:
    teacher_model.eval()
    student_model.train()
    
    labels: Tensor = batch["labels"]
    inputs: Tensor = batch["input_features"]
    
    labels = labels.to(teacher_model_device)
    inputs = inputs.to(teacher_model_device)
    with torch.no_grad():
        teacher_model_outputs: Dict[str, Tensor] = teacher_model(
            input_features=inputs, labels=labels
        )

    labels = labels.to(student_model_device)
    inputs = inputs.to(student_model_device)
    student_model_outputs: Duct[str, Tensor] = student_model(
        input_features=inputs, labels=labels
    )

    # Dimension: batch-size * padded-token-number * token-vocab-size
    teacher_logits: Tensor = teacher_model_outputs["logits"]
    student_logits: Tensor = student_model_outputs["logits"]
    teacher_outputs_dist: Tensor = nn.functional.softmax(
        teacher_logits / temperature, dim=-1
    ).to(student_model_device)
    student_outputs_log_dist: Tensor = nn.functional.log_softmax(
        student_logits / temperature, dim=-1
    )
    loss_pl: Tensor = student_model_outputs.loss
    loss_kl: Tensor = kl_divergence(
        teacher_outputs_dist, student_outputs_log_dist, labels
    ) * pow(temperature, 2)
    loss: Tensor = 0.8 * loss_pl + 1.0 * loss_kl
    
    all_loss: Dict = {
        "loss": loss, "loss_pl": loss_pl, "loss_kl": loss_kl
    }
    return all_loss


def eval_step(
    batch: BatchFeature,
    tokenizer: WhisperTokenizer,
    teacher_model: WhisperForConditionalGeneration, 
    student_model: WhisperForConditionalGeneration,
    teacher_model_device: torch.device,
    student_model_device: torch.device,
    temperature: int=2.0,
) -> Dict[str, Tensor]:
    teacher_model.eval()
    student_model.eval()
    
    labels: Tensor = batch["labels"]
    inputs: Tensor = batch["input_features"]

    labels = labels.to(teacher_model_device)
    inputs = inputs.to(teacher_model_device)
    with torch.no_grad():
        teacher_model_outputs: Dict[str, Tensor] = teacher_model(
            input_features=inputs, labels=labels
        )
        
    labels = labels.to(student_model_device)
    inputs = inputs.to(student_model_device)
    with torch.no_grad():
        student_model_outputs: Duct[str, Tensor] = student_model(
            input_features=inputs, labels=labels
        )
    
    # Dimension: batch-size * padded-token-number * token-vocab-size
    teacher_logits: Tensor = teacher_model_outputs["logits"]
    student_logits: Tensor = student_model_outputs["logits"]
    teacher_outputs_dist: Tensor = nn.functional.softmax(
        teacher_logits / temperature, dim=-1
    ).to(student_model_device)
    student_outputs_log_dist: Tensor = nn.functional.log_softmax(
        student_logits / temperature, dim=-1
    )
    loss_pl: Tensor = student_model_outputs.loss
    loss_kl: Tensor = kl_divergence(
        teacher_outputs_dist, student_outputs_log_dist, labels
    ) * pow(temperature, 2)
    loss: Tensor = 0.8 * loss_pl + 1.0 * loss_kl

    all_metrics: Dict = {
        "loss": loss.cpu().tolist(), 
        "loss_pl": loss_pl.cpu().tolist(), "loss_kl": loss_kl.cpu().tolist()
    }

    target_texts: List[str] = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    inputs = inputs.to(teacher_model_device)
    teacher_output_texts: List[str] = tokenizer.batch_decode(
        teacher_model.generate(inputs=inputs), skip_special_tokens=True
    )
    inputs = inputs.to(student_model_device)
    student_output_texts: List[str] = tokenizer.batch_decode(
        student_model.generate(inputs=inputs), skip_special_tokens=True
    )

    teacher_output_texts = [
        text_force_simplified_chinese(x, tokenizer.language) 
        for x in teacher_output_texts
    ]
    student_output_texts = [
        text_force_simplified_chinese(x, tokenizer.language) 
        for x in student_output_texts
    ]
    teacher_metric: float = cal_cer_or_wer(
        target_texts, teacher_output_texts, tokenizer.language
    ).cpu().tolist()
    student_metric: float = cal_cer_or_wer(
        target_texts, student_output_texts, tokenizer.language
    ).cpu().tolist()
    all_metrics["teacher_cer/wer"] = teacher_metric
    all_metrics["student_cer/wer"] = student_metric
    return all_metrics


def train_loop(
    dataloader: DataLoader,
    teacher_model: WhisperForConditionalGeneration, 
    student_model: WhisperForConditionalGeneration,
    optimizer: AdamW,
    teacher_model_device: torch.device, 
    student_model_device: torch.device,
    temperature: int=2.0,
) -> None:
    for i, batch in enumerate(tqdm(dataloader)):
        all_loss: Dict[str, Tensor] = train_step(
            batch, teacher_model, student_model,
            teacher_model_device, student_model_device,
            temperature
        )
        loss: Tensor = all_loss["loss"] 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 500 == 0:
          print("training log: ", {k: v.cpu().tolist() for k, v in all_loss.items()})


def eval_loop(
    dataloader: DataLoader,
    tokenizer: WhisperTokenizer,
    teacher_model: WhisperForConditionalGeneration,
    student_model: WhisperForConditionalGeneration,
    teacher_model_device: torch.device,
    student_model_device: torch.device,
    temperature: int=2.0
) -> None:
    metrics_recorder: Dict[str, List] = {
        "loss": [], "loss_pl": [], "loss_kl": [], 
        "teacher_cer/wer": [], "student_cer/wer": []
    }
    for batch in tqdm(dataloader):
        all_metrics: Dict[str, Tensor] = eval_step(
            batch, tokenizer, teacher_model, student_model,
            teacher_model_device, student_model_device,
            temperature
        )
        for k in metrics_recorder:
            metrics_recorder[k].append(all_metrics[k])
    
    all_metrics: Dict[str, float] = {}
    for k in metrics_recorder:
        all_metrics[k] = sum(metrics_recorder[k]) / len(metrics_recorder[k])
    print("eval log: ", all_metrics)


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    model_configs: Dict = configs["model"]
    data_configs: Dict = configs["data"]
    train_configs: Dict = configs["train"]
    common_configs: Dict = configs["common"]

    os.system("mkdir -p %s" % model_configs["distil_model_path"])
    os.system("cp %s %s" % (sys.argv[1], model_configs["distil_model_path"]))

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        model_configs["processor_path_or_name"], 
        language=common_configs["lang"], task="transcribe"
    )
    teacher_model: WhisperForConditionalGeneration = \
        WhisperForConditionalGeneration.from_pretrained(
            model_configs["teacher_model_path_or_name"]
        )
    student_model: WhisperForConditionalGeneration = \
        WhisperForConditionalGeneration.from_pretrained(
            model_configs["student_model_path"]
        )

    datasets_dict: DatasetDict = hf_datasetdict_load_audio_jsonl(
        data_configs["train_jsonl_path"], 
        data_configs["dev_jsonl_path"], 
        data_configs["test_jsonl_path"],
        sample_id_col="", 
        audio_duration_col=data_configs["audio_duration_col"],
        audio_path_col=data_configs["audio_path_col"]
    )
    dataset_filter: Callable = fn_gen_hf_dataset_filter_by_asr_data(
        processor.tokenizer,
        min_audio_duration=data_configs["min_audio_duration"],
        max_audio_duration=data_configs["max_audio_duration"],
        min_token_num=data_configs["min_token_num"], 
        max_token_num=data_configs["max_token_num"],
        audio_path_col=data_configs["audio_path_col"],
        text_col=data_configs["text_col"]
    )
    datasets_dict = datasets_dict.filter(dataset_filter, num_proc=4)

    # Filter sample's which `metric_col` value smaller than `min_pseudo_label_metric`
    datasets_dict["train"] = datasets_dict["train"].filter(
        lambda x: x[data_configs["metric_col"]] <= data_configs["min_pseudo_label_metric"], 
        num_proc=4
    )

    collator: DataCollatorSpeechSeq2SeqWithPaddingV1 = \
        DataCollatorSpeechSeq2SeqWithPaddingV1(
            processor, 
            lang=common_configs["lang"],
            path_col=data_configs["audio_path_col"],
            text_col=data_configs["text_col"],
            audio_duration_col="input_length",
            model_input_col="input_features",
            model_label_col="labels",
            sample_id_col="",
            target_sample_rate=common_configs["sampling_rate"]
        )

    """TODO
    a = get_parameter_names(student_model, [nn.LayerNorm], 
        forbidden_module=[student_model.model.encoder]
    )
    """
    default_train_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(output_dir="./fake")
    optimizer: AdamW = AdamW(
        params=[param for name, param in student_model.named_parameters()],
        lr=default_train_args.learning_rate,
        betas=(default_train_args.adam_beta1, default_train_args.adam_beta2),
        eps=default_train_args.adam_epsilon,
    )
    print(optimizer)

    lr_scheduler: ExponentialLR = ExponentialLR(\
        optimizer, gamma=train_configs["lr_decay_gamma"]
    )

    teacher_model_device: torch.device = torch.device(train_configs["teacher_model_device"])
    student_model_device: torch.device = torch.device(train_configs["student_model_device"])
    
    teacher_model = teacher_model.to(teacher_model_device)
    student_model = student_model.to(student_model_device)
    
    teacher_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=common_configs["lang"], task="transcribe"
    )
    student_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=common_configs["lang"], task="transcribe"
    )

    train_dataloader: DataLoader = DataLoader(
        datasets_dict["train"],
        collate_fn=collator, 
        batch_size=train_configs["batch_size"], 
        num_workers=4,
        shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(
        datasets_dict["validation"],
        collate_fn=collator, 
        batch_size=train_configs["batch_size"], 
        num_workers=4
    )

    for epoch in range(train_configs["epochs"]):
        print("training log: epoch=%i" % epoch)
        print("lr=%f" % lr_scheduler.get_lr()[0])
        train_loop(
            train_dataloader, teacher_model, student_model, optimizer, 
            teacher_model_device, student_model_device, 
            2.0
        )
        eval_loop(
            dev_dataloader, processor.tokenizer, teacher_model, student_model, 
            teacher_model_device, student_model_device,
            2.0
        )
        ckpt_dir: str = os.path.join(model_configs["distil_model_path"], "ckpt_%i" % epoch)
        student_model.save_pretrained(ckpt_dir)
        print("Saved CKPT to %s" % ckpt_dir)

        lr_scheduler.step()

    final_ckpt_dir: str = os.path.join(model_configs["distil_model_path"], "final") 
    student_model.save_pretrained(final_ckpt_dir)
    print("Saved final CKPT to %s" % final_ckpt_dir)


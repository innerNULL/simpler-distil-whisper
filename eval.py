# -*- coding: utf-8 -*-
# file: eval.py
#
# Usage:
# python bin/eval.py demo_configs/eval.json


import pdb
import sys
import json
import unicodedata
import re
import string
import pandas as pd
from pandas import DataFrame
from typing import Dict, List
from torchmetrics.text import CharErrorRate
from opencc import OpenCC


def run_text_norm(input_string):
    # Remove Chinese and English punctuations
    chinese_punctuations = '，。！？【】（）《》“”‘’：；“”'
    english_punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    punctuations_pattern = f"[{re.escape(chinese_punctuations)}{re.escape(english_punctuations)}]"

    # Remove alphabets and numbers
    alphanum_pattern = r'[A-Za-z0-9]'

    # Combine patterns
    combined_pattern = f'{punctuations_pattern}|{alphanum_pattern}'

    # Remove specified characters using regex
    result = re.sub(combined_pattern, '', input_string)
    
    return result


def eval(
    asr_results_path: str, target_col: str, output_col: str,  
    lang: str="", text_norm: bool=False
) -> None:
    asr_results: List[str] = [
        json.loads(x) for x in open(asr_results_path, "r").read().split("\n")
        if x not in {""}
    ]
    targets: List[str] = [x[target_col] for x in asr_results]
    outputs: List[str] = [x[output_col] for x in asr_results]

    if lang in {"mandarin", "zh-TW", "zh-tw"}:
        converter: OpenCC = OpenCC('tw2s.json')
        targets = [converter.convert(x) for x in targets]
        outputs = [converter.convert(x) for x in outputs]

    if text_norm:
        targets = [run_text_norm(x) for x in targets]
        outputs = [run_text_norm(x) for x in outputs]

    assert(len(targets) == len(outputs))

    cer = CharErrorRate()
    results: Dict = {
        "sample_size": len(outputs),
        "cer": float(cer(outputs, targets))
    }
    print(results)


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    eval(
        configs["asr_results_path"], configs["target_col"], configs["output_col"],
        configs["lang"], configs["text_norm"]
    )

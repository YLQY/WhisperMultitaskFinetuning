import random
from tqdm import tqdm
from random import shuffle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
import torchaudio
from torch.utils.data import IterableDataset
from whisper.utils.common_utils import (
    log_info
)

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def normalization_text(text):
    """
        文本格式化
    """

    return text

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# 准备数据
class IterWhisperDataset(IterableDataset):

    def __init__(self,wav_scp,text,whisper_feature_extractor,whisper_tokenizer):
        # 处理为字典
        self.data_list = {}
        # 音频路径
        with open(wav_scp,'r',encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                idx = line.split(" ")[0]
                wav_path = " ".join(line.split(" ")[1:])
                self.data_list[idx] = []
                self.data_list[idx].append(wav_path)
                pass
            pass
        # 音频文本
        with open(text,'r',encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                idx = line.split(" ")[0]
                text = " ".join(line.split(" ")[1:])
                self.data_list[idx].append(text)
                pass
            pass

        self.whisper_feature_extractor = whisper_feature_extractor
        self.whisper_tokenizer = whisper_tokenizer
        log_info("打乱文本，全部个数为："+str(len(self.data_list)))
        # 打乱
        # self.data_list = random_dic(self.data_list)
        pass
    
    def __len__(self):
        return len(self.data_list)

    # 传入模型遍历的东西
    def __iter__(self):
        
        # 遍历我们的所有数据
        for idx in self.data_list:
            
            # 拿到一些信息 - 语种
            language = idx.split("|")[1]
            # 拿到一些信息 - 任务类型
            task = idx.split("|")[2]

            # 音频的路径
            wav_path = self.data_list[idx][0]
            # 音频的文本
            text = self.data_list[idx][1]
            
            text = normalization_text(text)

            example = {}
            example['idx'] = idx
            # 提取特征
            data_audio = torchaudio.load(wav_path)
            example['input_features'] = self.whisper_feature_extractor(data_audio[0].numpy(),sampling_rate=16000).input_features[0]

            # 任务1 - 中文转写
            if language == "chinese" and task == "transcribe":
                # token
                self.whisper_tokenizer.set_prefix_tokens(language='zh', task=task)
            elif language == "chinese" and task == "translate":
                self.whisper_tokenizer.set_prefix_tokens(language='zh', task=task)
            else:
                #log_info("---------error-----------：language或者task有问题")
                self.whisper_tokenizer.set_prefix_tokens(language=language, task=task)
                
            
            example['labels'] = self.whisper_tokenizer(text).input_ids[:]
            res_jie = self.whisper_tokenizer.decode(example['labels'],skip_special_tokens=False)
            # print("---解码--->",res_jie)
            # print(example['labels'])
            # print(self.whisper_tokenizer.batch_decode(example['labels']))
            yield example

        

        pass

    pass

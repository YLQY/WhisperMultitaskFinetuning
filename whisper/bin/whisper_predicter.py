from peft import PeftConfig,PeftModel
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
)
from tqdm import tqdm
import torch
import numpy as np
import torchaudio
from whisper.tokenizer.toknizer_whisper import WhisperTokenizer
from whisper.utils.common_utils import (
    log_info
)
from whisper.utils.data_utils import (
    IterWhisperDataset
)


class WhisperPredicter:
    """
        关于whisper训练的所有东西
    """
    def __init__(self,config):
        self.config = config

        # 加载模型
        self.load_whisper_model()
        # 加载数据
        self.load_data_wav_and_text()

        pass

    def predict_start(self):
        """
            模型的预测
        """
        out_file = open(self.config['predict']['result_file'],'w',encoding="utf-8")
        for step, batch in enumerate(tqdm(self.eval_data_list)):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    generated_tokens = (
                        self.model.generate(
                            input_features=torch.from_numpy(batch["input_features"][np.newaxis,:,:]),
                            decoder_input_ids=torch.from_numpy(np.array([batch["labels"][:4]])),
                            max_new_tokens=255,
                        )
                        .cpu()
                        .numpy()
                    )
                    labels = batch["labels"]
                    labels = np.where(labels != -100, labels, self.whisper_tokenizer.pad_token_id)

                    decoded_preds = self.whisper_tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    decoded_labels = self.whisper_tokenizer.batch_decode(labels, skip_special_tokens=True)
                    out_file.write(batch['idx']+" "+decoded_preds[0]+"\n")
                    pass
                pass
            del generated_tokens, labels, batch
        pass

    def load_data_wav_and_text(self):
        """
            准备数据
        """
        log_info("加载eval集")
        self.eval_data_list = IterWhisperDataset(
            self.config['predict']['eval']['wav_scp'],
            self.config['predict']['eval']['text'],
            self.whisper_feature_extractor,
            self.whisper_tokenizer
        )
        pass

    def load_whisper_model(self):
        """
            加载模型
        """
        whisper_model = self.config['model']['model_path']

        # 特征提取
        log_info("加载 whisper_feature_extractor")
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

        # token
        log_info("加载 whisper_tokenizer")
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model)

        # 加载大模型
        if self.config['model']['is_large_model']:
            log_info("加载 whisper模型 - 经过peft微调")
            peft_model_id = self.config['predict']['model_path']
            peft_config = PeftConfig.from_pretrained(peft_model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                peft_config.base_model_name_or_path
            )
            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            pass
        else:
            log_info("加载 whisper模型")
            self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

        log_info("模型结构：")
        log_info(self.model)
        self.model.eval()
        pass

    pass










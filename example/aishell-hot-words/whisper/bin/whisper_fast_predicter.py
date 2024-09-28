from peft import PeftConfig,PeftModel
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    WhisperTokenizer
)
import ctranslate2
from tqdm import tqdm
import torch
import numpy as np
import torchaudio
from whisper.utils.common_utils import (
    log_info
)
from whisper.utils.data_utils import (
    IterWhisperDataset
)

class WhisperFastPredicter:
    """
        关于whisper训练的所有东西
    """
    def __init__(self,config):
        self.config = config

        # 加载模型
        self.load_whisper_model_ctranslate()
        # 加载数据
        self.load_data_wav_and_text()

        pass

    def predict_start(self):
        """
            模型的预测
        """
        out_file = open(self.config['dev_env']['result_file'],'w',encoding="utf-8")
        for step, batch in enumerate(tqdm(self.eval_data_list)):
            # 模型的输入
            input_features = ctranslate2.StorageView.from_array(
                batch["input_features"][np.newaxis,:,:]
            )
            results = self.model.generate(
                input_features, 
                [batch["labels"][:4]]
            )
            # 预测结果
            decoded_preds = self.whisper_tokenizer.batch_decode([results[0].sequences_ids[0]], skip_special_tokens=False)
            out_file.write(batch['idx']+" "+decoded_preds[0]+"\n")
            print(decoded_preds[0])
            pass

    def load_data_wav_and_text(self):
        """
            准备数据
        """
        log_info("加载eval集")
        self.eval_data_list = IterWhisperDataset(
            self.config['dev_env']['dev']['wav_scp'],
            self.config['dev_env']['dev']['text'],
            self.whisper_feature_extractor,
            self.whisper_tokenizer
        )
        pass

    def load_whisper_model_ctranslate(self):
        """
            使用ctranslate进行模型加速
        """
        whisper_model = self.config['dev_env']['ori_model_path']
        # 特征提取
        log_info("加载 whisper_feature_extractor")
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

        # token
        log_info("加载 whisper_tokenizer")
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model)

        # 加载模型
        self.model = ctranslate2.models.Whisper(
            self.config['dev_env']['ctranslate_model_path'],
            device=self.config['dev_env']['conf']['device'],
            compute_type=self.config['dev_env']['conf']['compute_type']
        )
        pass

    @staticmethod
    def predict_one_wav(wav_path,prompt,model_path,ctranslate_model_path):
        """
            可以预测一个音频
        """
        import librosa
        # 读取数据
        audio, _ = librosa.load(
            wav_path, sr=16000, mono=True
        )

        # 提取特征
        processor = WhisperProcessor.from_pretrained(model_path)
        inputs = processor(audio, return_tensors="np", sampling_rate=16000)
        features = ctranslate2.StorageView.from_array(inputs.input_features)

        # Load the model on CPU.
        model = ctranslate2.models.Whisper(ctranslate_model_path)
        log_info("---------开始识别----------")
        prompt = processor.tokenizer.convert_tokens_to_ids(
            prompt
        )
        # 识别
        results = model.generate(features, [prompt])
        # 解码
        transcription = processor.decode(results[0].sequences_ids[0])
        log_info(transcription)
        log_info("---------结束识别----------")
        pass

    pass










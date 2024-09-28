from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperTokenizer
)

from whisper.utils.common_utils import (
    log_info
)
from whisper.utils.data_utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    IterWhisperDataset
)


class WhisperTrainer:
    """
        关于whisper训练的所有东西
    """
    def __init__(self,config):
        self.config = config

        # 加载模型
        self.load_whisper_model()
        # 加载数据
        self.load_data_wav_and_text()
        # 模型参数
        self.load_whisper_model_argv()
        pass

    def train_start(self):
        """
            模型训练
        """
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data_list,
            eval_dataset=self.test_data_list,
            tokenizer=self.whisper_feature_extractor,
            data_collator=self.data_collator
        )

        # 训练
        check_point = self.config['model']['model_train_argv']['resume_from_checkpoint']
        if check_point != "":
            log_info("模型恢复中....."+check_point)
            train_result = trainer.train(resume_from_checkpoint=check_point)
        else:
            train_result = trainer.train()
        trainer.save_model()
        pass

    def load_data_wav_and_text(self):
        """
            准备数据
        """
        log_info("加载训练集")
        self.train_data_list = IterWhisperDataset(
            self.config['data']['train']['wav_scp'],
            self.config['data']['train']['text'],
            self.whisper_feature_extractor,
            self.whisper_tokenizer
        )
        log_info("加载测试集")
        self.test_data_list = IterWhisperDataset(
            self.config['data']['test']['wav_scp'],
            self.config['data']['test']['text'],
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

        # whisper模型
        log_info("加载 whisper模型")
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

        # 数据的process
        log_info("加载 processor")
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        log_info("模型结构：")
        log_info(self.model)

        # 加载大模型
        if self.config['model']['is_large_model']:
            log_info("大模型由peft加载，Lora减少训练参数")
            self.model = self._load_large_model_peft(self.model)
            pass
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            forward_attention_mask=self.config['model']['data_collator']['forward_attention_mask'],
        )

        pass

    def load_whisper_model_argv(self):
        log_info("加载模型训练的参数")
        self.training_args = Seq2SeqTrainingArguments(
            output_dir = self.config['model']['model_train_argv']["out_model_path"],  # change to a repo name of your choice
            per_device_train_batch_size = self.config['model']['model_train_argv']["per_device_train_batch_size"],
            gradient_accumulation_steps = self.config['model']['model_train_argv']["gradient_accumulation_steps"],  # increase by 2x for every 2x decrease in batch size
            learning_rate = self.config['model']['model_train_argv']["learning_rate"],
            warmup_steps = self.config['model']['model_train_argv']["warmup_steps"],
            num_train_epochs = self.config['model']['model_train_argv']["num_train_epochs"],
            evaluation_strategy = self.config['model']['model_train_argv']["evaluation_strategy"],
            fp16 = self.config['model']['model_train_argv']["fp16"],
            per_device_eval_batch_size = self.config['model']['model_train_argv']["per_device_eval_batch_size"],
            generation_max_length = self.config['model']['model_train_argv']["generation_max_length"],
            logging_steps = self.config['model']['model_train_argv']["logging_steps"],
            remove_unused_columns = self.config['model']['model_train_argv']["remove_unused_columns"],  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
            label_names = self.config['model']['model_train_argv']["label_names"],  # same reason as above
        )
        pass

    def _load_large_model_peft(self,model):
        """
            加载大模型
        """
        # 训练大模型
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)

        config = LoraConfig(
            r=32, 
            lora_alpha=64, 
            target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",#["q_proj", "v_proj"],
            lora_dropout=0.05, 
            bias="none"
        )
        model = get_peft_model(model, config)

        log_info("查看训练参数和模型原始参数量")
        model.print_trainable_parameters()

        return model

    pass










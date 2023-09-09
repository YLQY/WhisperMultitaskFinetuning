
import yaml
import logging
from peft import PeftModel
from whisper.utils.ctranlate2_convert import TransformersConverter

from transformers import (
    WhisperTokenizer,
    WhisperForConditionalGeneration,
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def convert_finetuning_peft_model_into_whisper(
    peft_model_path,
    whisper_model_path,
    out_ctranslate_path
    ):
    """
        微调后的模型和whisper合并参数，为一个，并且转化
    """
    # 加载whisper模型
    whisper_base_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path)
    # 加载peft模型
    model = PeftModel.from_pretrained(whisper_base_model, peft_model_path)
    merged_model = model.merge_and_unload(progressbar=True)

    whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_path)

    # 获得转换器
    transformers_converter = TransformersConverter(whisper_model_path,merged_model,whisper_tokenizer)

    # output_dir: Output directory where the CTranslate2 model is saved.
    # vmap: Optional path to a vocabulary mapping file that will be included
    #         in the converted model directory.
    # quantization: Weight quantization scheme (possible values are: int8, int8_float32,
    #         int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
    # force: Override the output directory if it already exists.
    # 模型转化
    transformers_converter.convert(out_ctranslate_path,force=True)
    pass

def load_whisper_config(config_path):
    """
        读取配置
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        return result
    pass

def log_info(str_log):
    logging.info(str_log)
    pass








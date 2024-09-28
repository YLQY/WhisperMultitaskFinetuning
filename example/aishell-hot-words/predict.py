
from whisper.bin.whisper_predicter import WhisperPredicter
from whisper.utils.common_utils import (
    load_whisper_config,
    log_info
)


log_info("加载配置文件")
config = load_whisper_config(
    "./config/whisper_multitask.yaml"
)

log_info("加载模型测试器")
whisper_predicter = WhisperPredicter(config)

log_info("测试")
whisper_predicter.predict_start_hot_words()

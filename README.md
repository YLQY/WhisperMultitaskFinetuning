# WhisperMultitaskFinetuning
Whisper语音大模型的多任务微调，目前支持如下任务
 1. 多语种的**转录**（ASR）
 2. 多语种到英文的**翻译**任务（AST）
## 安装
 - 克隆此代码
```markdown
git clone https://github.com/YLQY/WhisperMultitaskFinetuning.git
```
 - 安装Conda请参考 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
 - 创建Conda环境
```markdown
conda create -n whisper python=3.8
conda activate whisper
```
 - 配置环境 - 有一个命令会运行很久，中途可能会断掉，请多尝试
```markdown
bash set_env.sh
```
 - 下载Whisper_large_v2模型（大概需要一晚上）
```markdown
git clone https://huggingface.co/openai/whisper-large-v2
cd whisper-large-v2
git lfs fetch
git lfs checkout
```
### PS
 - 也可以用Whisper_Base模型做尝试
```markdown
git clone https://huggingface.co/openai/whisper-base
cd whisper-base
git lfs fetch
git lfs checkout
```
## 案例1：在Whisper上同时微调转录与翻译任务
```markdown
cp -r whisper example/aishell-sample/
cd example/aishell-sample
```
### 1.1 数据准备
 - 转录数据，参考data/transcribe/{wav.scp,text}
 - 翻译数据，参考data/translate/{wav.scp,text}
   
    > **wav.scp**  <br>
    > id | language | task -> /absolute/path/to/wav <br>
    
    > **text** <br>
    > id | language | task -> label

| id   | 语种 |     任务类型 |
| :-----: | :--: | :-------: |
| BAC009S0150W0001 |  chinese  | translate |
| BAC009S0150W0001 |  chinese  | transcribe |
```markdown
# 模型总共训练数据 - 将翻译和转录的数据合并为一个整体
cat data/transcribe/wav.scp data/translate/wav.scp > data/wav.scp
cat data/transcribe/text data/translate/text > data/text
```

### 1.2 更改配置
修改 config/whisper_multitask.yaml
```yaml
data:
  train:
    wav_scp: "训练数据wav.scp的绝对路径"
    text: "训练数据text的绝对路径"
  test:
    wav_scp: "测试数据wav.scp的绝对路径"
    text: 测试数据text的绝对路径"

predict:
  model_path: "自己微调之后的模型位置"
  result_file: "结果保存的绝对路径"
  eval: 
    wav_scp: "验证数据wav.scp的绝对路径"
    text: "验证数据text的绝对路径"

model:
  model_path: "自己的whisper_large_v2或者base模型路径"
  is_large_model: "如果是whisper_large_v2模型则设置为True否则为False"
  data_collator:
    forward_attention_mask: False
  model_train_argv:
    out_model_path: "模型保存的路径"
    resume_from_checkpoint : ""
    per_device_train_batch_size: 1
    per_device_eval_batch_size: 1
    gradient_accumulation_steps: 1
    num_train_epochs: 1
    learning_rate: 0.0001
    logging_steps: 2
    fp16: False
    warmup_steps: 50
    evaluation_strategy: "epoch"
    generation_max_length: 128
    remove_unused_columns: False
    label_names:
      - labels
```
### 1.3 训练模型
更改代码中配置文件路径
```markdown
python3 train.py
```
### 1.4 测试模型
更改代码中配置文件路径
```markdown
python3 predict.py
```
### 1.5 使用CTranslate2进行模型加速
```markdown
from whisper.utils.common_utils import convert_finetuning_peft_model_into_whisper

log_info("Lora参数融入Whsiper")
convert_finetuning_peft_model_into_whisper(
  # 微调后的peft模型路径
  peft_model_path=config['predict']['model_path'],
  # 原始whisper模型路径
  whisper_model_path=config['dev_env']['ori_model_path'],
  # 可以使用ctranslate加速后的模型位置
  out_ctranslate_path = config['dev_env']['ctranslate_model_path']
)
```

## 讨论交流群
<img src=https://github.com/YLQY/WhisperMultitaskFinetuning/blob/main/res/qq.jpg width=40% />


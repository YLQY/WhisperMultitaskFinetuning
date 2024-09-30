## 案例1：在Transformers上实现热词功能-Whisper为案例
```markdown
conda create -n whipser-hot-words python=3.10
conda activate whipser-hot-word
pip install transformers==4.44.0
pip install torch==1.13.0
pip install peft==0.13.0
pip install torchaudio==0.13.0
pip install ctranslate2==4.4.0
```
### 1 安装环境&代码更新
```markdown
pip show transformers

复制代码到transformers的安装路径
cp whisper/utils/tree_node.py /path/to/transformers/generation/
cp code/utils.py /path/to/transformers/generation/utils.py
```
### 2 准备数据
```markdown
数据集1：热词aishell（一共235条）
https://www.modelscope.cn/datasets/speech_asr/speech_asr_aishell1_hotwords_testsets/files
数据集2：其他集合（cv-corpus-19.0-2024-09-13-zh-CN.tar.gz）
https://commonvoice.mozilla.org/zh-CN/datasets

eg:
热词词库：data/hotword.txt
热词测试集aishell：data/{wav.scp,text}
吉他集合测试集：data/{commonvoice.wav.scp,commonvoice.text}
```
### 3 下载模型
```markdown
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/BELLE-2/Belle-whisper-large-v2-zh
git lfs fetch
git lfs checkout
```
### 4 更改配置文件
```markdown
目录：config/whisper_multitask.yaml

# 模型推理
predict:
  # 模型的路径，对应上面步骤3，下载模型的位置
  model_path: "/mnt/f/WSL/lainspeech/whisper/hot_words/model"
  # 结果输出文件
  result_file: "./data/result"
  # 对应上面步骤2，准备的wav.scp和text路径
  eval: 
    wav_scp: "/mnt/f/WSL/lainspeech/whisper/hot_words/WhisperMultitaskFinetuning/example/aishell-hot-words/data/wav.scp"
    text: "/mnt/f/WSL/lainspeech/whisper/hot_words/WhisperMultitaskFinetuning/example/aishell-hot-words/data/text"
model:
  # 模型的路径，对应上面步骤3，下载模型的位置
  model_path: "/mnt/f/WSL/lainspeech/whisper/hot_words/model"

```
### 5 运行代码
```markdown
python3 predict.py
```
### 6 计算WER
```markdown
# 传入步骤4中的结果输出文件路径
bash wer.sh ./data/result
```


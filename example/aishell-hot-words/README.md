## 案例1：在Transformers上实现热词功能-Whisper为案例		
### 0 效果
| 参数   | Aishell1-235 | Aishell1-235-fix1 | Aishell1-235-fix2 | Aishell1-235-fix3 | CommonVoice-19-test-fix1 |
| :-----------------: | :----: | :-------: | :-------: | :-------: | :-------: |
| B=3、S=0  | 9.00/78.72  | 9.03/78.72	|7.90/67.23  |	7.90/67.23  |	16.15/64.59 |
| B=3、S=1  | 6.85/62.55  |	6.95/62.97	|6.27/56.17	 |6.09/55.31	  |14.91/65.14|
| B=3、S=2  |	6.01/53.19	|6.32/55.31	  |5.98/51.06  |	5.67/48.51  |	14.49/67.17
| B=3、S=3	| 10.65/51.48	|6.17/50.63	  |5.93/47.65  |	5.69/48.08  |	16.95/71.54
| B=3、S=4	| 22.96/52.76	|6.93/49.78	  |6.77/48.08  |	6.61/48.93  |	
| B=3、S=5	| 44.24/62.12	|12.07/60.0	  |12.07/60.0  |	10.97/54.04 |	
| B=5、S=0	| 9.00/78.29  |	8.97/78.29	|7.37/62.12	 |7.37/62.12    |	
| B=5、S=1  |	6.69/61.27	|6.80/61.70	  |5.83/50.63  |	5.62/50.21  |	
| B=5、S=2  |	14.48/52.76	|6.17/53.61	  |5.54/45.95  | 	5.25/45.95  |	
| B=5、S=3  |	23.59/50.21	|6.43/49.36	  |5.90/43.82  |	5.30/43.82  |	
| B=5、S=4  |	45.05/50.21	|7.24/46.38	  |7.01/44.68  |	9.60/44.68  |	
| B=5、S=5  |	125.48/65.10|	14.88/57.87 |	15.09/57.44|	16.77/51.48	|
| B=10、S=0	| 9.00/78.29	|8.97/78.29	  |6.80/56.17	 |6.80/56.17    |	
| B=10、S=1 |	6.61/60.42	|6.72/60.85	  |5.30/46.38  |	5.09/45.53	|
| B=10、S=2 |	18.68/51.48	|5.93/52.34	  |4.59/38.29  |	4.46/38.72  |	
| B=10、S=3 |	39.88/48.08	|6.53/45.95	  |5.56/37.44  |	5.43/40.42  |	
```markdown
conda create -n whisper-hot-words python=3.8
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


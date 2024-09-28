## 案例1：在Transformers上实现热词功能-Whisper为案例
```markdown
pip install transformers==4.44.0
pip install torch==1.13.0
```
### 1.1 安装环境&代码更新
```markdown
pip show transformers

复制代码到transformers的安装路径
cp whisper/utils/tree_node.py /path/to/transformers/generation/
cp code/utils.py /path/to/transformers/generation/utils.py
```
### 1.2 准备数据
```markdown
数据集1：热词aishell（一共235条）
https://www.modelscope.cn/datasets/speech_asr/speech_asr_aishell1_hotwords_testsets/files
数据集2：其他集合
https://commonvoice.mozilla.org/zh-CN/datasets

eg:
热词词库：data/hotword.txt
热词aishell：data/{wav.scp,text}
```
### 1.3 运行代码




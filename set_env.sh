apt install git-lfs
pip install transformers==4.32.0
pip install datasets
pip install transformers[torch]
pip install tensorboardX
pip install torch
pip install torchaudio
pip install librosa
pip install jiwer
pip install evaluate
pip install -U requests[security]

# 这个命令可能一次运行不成功，尝试多运行几次
pip install -q git+https://github.com/huggingface/peft.git@main

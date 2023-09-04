apt install git-lfs
pip install transformers==4.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers[torch] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install librosa -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install jiwer -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install evaluate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U requests[security] -i https://pypi.tuna.tsinghua.edu.cn/simple

# 这个命令可能一次运行不成功，尝试多运行几次
pip install -q git+https://github.com/huggingface/peft.git@main

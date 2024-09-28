res=$1
python whisper/utils/compute-wer.py --char=1 --v=1 \
      data/text ${res}

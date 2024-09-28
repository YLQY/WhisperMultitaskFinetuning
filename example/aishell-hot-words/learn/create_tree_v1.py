from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    WhisperTokenizer
)

text = "邓郁松"

whisper_model="/mnt/f/WSL/lainspeech/whisper/hot_words/model"
whisper_tokenizer=WhisperTokenizer.from_pretrained(whisper_model)
tokens = whisper_tokenizer(text,skip_special_tokens=True).input_ids[2:-1]

hot_tokens = [-1]+tokens+[99999]

print(hot_tokens)

class TreeNode:
    def __init__(self,value):
        self.value = value
        # 这个里面需要的数据结构是TreeNode
        self.child_nodes=[]
        pass

    # 添加节点
    def add_node(self,node):
        # 判断这个节点是否有value，如果有就不用添加，如果没有才需要添加
        for item in self.child_nodes:
            # 找到了
            if item.value == node.value:
                pass
            pass

        # 没有找到
        self.child_nodes.append(node)
        pass
    pass

# 打印树
def print_tree(root,level=0):
    print(" |-"*level,root.value)
    for item in root.child_nodes:
        print_tree(item,level+1)
        pass
    pass

root = TreeNode(-1)
root_ori = root
# 每个热词，按照token一个一个添加
for item in hot_tokens:
    # 根节点已经存在，不用添加
    if item == -1:
        continue
    temp_node = TreeNode(item)
    root.add_node(temp_node)
    root=temp_node
    pass


print_tree(root_ori)



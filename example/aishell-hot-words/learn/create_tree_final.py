from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    WhisperTokenizer
)


whisper_model="/mnt/f/WSL/lainspeech/whisper/hot_words/model"
whisper_tokenizer=WhisperTokenizer.from_pretrained(whisper_model)

all_hot_words_token = []

with open("/mnt/f/WSL/lainspeech/whisper/hot_words/WhisperMultitaskFinetuning/example/aishell-hot-words/data/hotword.txt",'r',encoding="utf-8") as file:
    for line in file.readlines():
        line = line.strip()
        tokens = whisper_tokenizer(line,skip_special_tokens=True).input_ids[2:-1]
        all_hot_words_token.append([-1]+tokens+[99999])
    pass

print(max_len)
print(all_hot_words_token)

class TreeNode:
    def __init__(self,value):
        self.value = value
        # 这个里面需要的数据结构是TreeNode
        self.child_nodes=[]
        pass
    # 判断是否存在这个节点
    def has_node(self,value):
        # 判断这个节点是否有value，如果有就不用添加，如果没有才需要添加
        for item in self.child_nodes:
            if item.value == value:
                return item
            pass
        return None

    # 添加节点
    def add_node(self,node):
        # 判断是否存在这个节点
        temp_node = self.has_node(node.value)
        # 不存在，添加
        if temp_node is None:
            self.child_nodes.append(node)
            return node
        # 存在这个节点
        return temp_node

    # 添加分数
    def _add_score_idx(self,decoder_path):

        # 热词完全匹配
        if decoder_path == []:
            return self.child_nodes
        
        temp = self.has_node(decoder_path[0])
        if temp is not None:
            return temp._add_score_idx(decoder_path[1:])
        else:
            return None
        pass

    # 添加分数
    def find_path_add_score_return_idx(self,find_path):
        total_length = max(20,len(find_path))

        for i in range(total_length):
            target_path = find_path[i:]
            print("查找路径",target_path)
            return_nodes = self._add_score_idx(target_path)
            if return_nodes is not None:
                return return_nodes
            pass
        return self._add_score_idx([])

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

for hot_tokens in all_hot_words_token:
    root=root_ori
    # 每个热词，按照token一个一个添加
    for item in hot_tokens:
        # 根节点已经存在，不用添加
        if item == -1:
            continue
        temp = TreeNode(item)
        # 添加节点
        temp = root.add_node(temp)
        root=temp
        pass


print_tree(root_ori)

lists = root_ori.find_path_add_score_return_idx([35,68,97,5,6,6,5363,66,66666])
if lists is not None:
    for item in lists:
        print(item.value)
        pass
else:
    for item in root_ori.child_nodes:  
        print(item.value)
        pass
print(max_len)

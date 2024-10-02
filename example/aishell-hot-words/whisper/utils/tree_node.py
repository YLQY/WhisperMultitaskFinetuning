class TreeNode:
    def __init__(self,value,deep=0):
        self.value = value
        self.deep = deep
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

        # 热词开始匹配
        if decoder_path == [-1]:
            return self.child_nodes
        # 热词匹配的过程中、热词的结尾
        if len(decoder_path) == 0:
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
            #print("查找路径",target_path)
            return_nodes = self._add_score_idx(target_path)
            if return_nodes is not None:
                return return_nodes
            pass
        return self._add_score_idx([-1])

        pass

    # 查找最后的热词是否在树里面
    def final_add_score(self,find_path):
        # 代表热词匹配了
        if self.has_node(99999):
            return True
        # 路径没有传入东西
        if len(find_path) == 0:
            return False
        temp = self.has_node(find_path[0])
        if temp is not None:
            return temp.final_add_score(find_path[1:])
        else:
            return False
        pass
    pass

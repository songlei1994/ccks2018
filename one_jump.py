from unitils import jaccard,hint,get_hint_from_seq,local_les
from config import placeholder
import Levenshtein
import networkx as nx
import numpy as np
import re

def one_jump_solver(one_jump_solver_ans,one_jump_solver_ss):
    
    """
    返回的 
            x[0] 定义为 key: 答案节点
                        value:候选答案特征
                        具体来说 (prop_max_value,jaccard,node_num,hint)
            
            x[1] 定义为 key: 答案节点
                        value:与答案节点相连的 hint > 0 的root节点
                        如不是hint>0 应该会引入噪声
                        数据类型为 dict
                            path 字段 每个元素对应一个相连的root
                            节点
                            node 字段为清洗去重的节点
                        
    """
    
    ans_for_select = {}
    ans_raw = {}
    # 无意义停用词 需要
    pattern = re.compile("(" + "|".join(placeholder) + "|现在|些地"+ ")")
    
    one_jump_solver_seq = pattern.sub("",one_jump_solver_ss)
    #print(one_jump_solver_seq)
    root_node = set([y for x,y in one_jump_solver_ans.keys()])
    root_path = {value:key for key,value in one_jump_solver_ans.keys()}
    for roots in one_jump_solver_ans.keys():
        G = one_jump_solver_ans[roots]
        if not len(G):
            continue 
        neighbor = set([ y for x,y in G.out_edges(roots[1])] + [x for x,y in G.in_edges(roots[1])]) - set([roots[1]])
        #print(roots)
        for row in neighbor:
            try:
                edge = G.edges[roots[1],row]["prop"]
            except KeyError:
                edge = G.edges[row,roots[1]]["prop"]
            
            infor_1 = jaccard(one_jump_solver_ss,roots[0] + edge)
            #infor_2 = hint(one_jump_solver_seq,roots[0] + roots[1] + edge + row)
            #infor_3 = Levenshtein.ratio(one_jump_solver_ss,roots[0] + edge)
            try:
                target,num = local_les(one_jump_solver_ss,edge.split("_")[0])
            except ValueError:
                num = 0
                
            if row not in ans_for_select:
                
                ans_for_select[row] = [num,infor_1]
                
                ans_raw[row] = {}
                ans_raw[row]["node"] = set()
                ans_raw[row]["path"] = set()
                ans_raw[row]["path"].add((roots[0],roots[1],edge))
                node_char = roots[1].split("_")[0].strip(' <>"“”').replace("（","(").replace("）",")")
                ans_raw[row]["node"].add(node_char)
                
            else :                
                aa = ans_for_select[row]
                
                #从其他root获得
                if hint(one_jump_solver_ss,roots[0] + roots[1] + edge) > 0: # 如果能够提供新信息
                    num = max(num,aa[0])
                    infor_1 = max(infor_1,aa[1])
                    ans_for_select[row] = [num,infor_1]
                    ans_raw[row]["path"].add((roots[0],roots[1],edge))
                    # 这里的node 要清洗
                    node_char = roots[1].split("_")[0].strip(' <>"“”').replace("（","(").replace("）",")")
                    ans_raw[row]["node"].add(node_char)
                """
                bb = ans_for_select[row]
                if aa[0]*aa[1]*aa[2] > bb[0]*bb[1]*bb[2]:
                    ans_for_select[row] = aa
                    ans_raw[row] = (roots,edge,row)
                """
            #从子节点获得
            second = (set(G[row]) - neighbor).intersection(root_node)
            for line in second:
                edge =  G.edges[row,line]["prop"]
                if hint(one_jump_solver_seq,root_path[line] + line + edge) > 0: # 如果能够提供新信息
                    node_char = line.split("_")[0].strip(' <>"“”').replace("（","(").replace("）",")")
                    ans_raw[row]["node"].add(node_char)
                    ans_raw[row]["path"].add((root_path[line],line,edge))
    
    for key in ans_for_select:
        link_node = [x + y + z for x,y,z in ans_raw[key]["path"]]
        infor_2 = hint(one_jump_solver_seq,"".join(link_node))
        node_num = len(ans_raw[key]["node"])
        ans_for_select[key].append(node_num)
        ans_for_select[key].append(infor_2)
                
    #return ans_for_select,ans_raw            
    values = []
    keys = []
    for key,value in ans_for_select.items():
        keys.append(key)
        values.append(value)
    #return ans_for_select
    keys = np.array(keys)
    hh = np.array(values).astype("float32")     

    #return hh
    # 搜索能返回 合理规模 的 候选 答案集
    for sup in range(1,30,1):
        index = hh[:,3] > 0.1*sup*hh[:,3].mean()
        if sum(index)< 80 and sum(index)>0:
            tmp = hh[index]
            ttt = {}
            ppp = {}
            #print("here")
            for row in keys[index]:
                ttt[row] = ans_for_select[row]
                ppp[row] = ans_raw[row]
            return ttt,ppp
    return ans_for_select,ans_raw
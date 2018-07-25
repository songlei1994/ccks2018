from unitils import search_prop,jaccard,hint,get_hint_from_seq,local_les
from config import placeholder
import Levenshtein
import networkx as nx
import numpy as np

"""
 
                        # 统计 边 的命中率（由于边是模糊命中的，这样似乎不太好）
                        ans_raw[line]= (roots,tmp_first_edge,row,edge,line) 
                        
                        if tmp_first_edge in jump_hint.keys():
                            prop_count = 2
                        else:
                            prop_count = 1
                        # jacard 相似度    
                        infor_1 = jaccard(two_jump_solver_ss,roots[0] + tmp_first_edge + edge)
                        # 字符 重合数     
                        # root[0] 可能也要加进来
                        infor_2 = hint(two_jump_solver_ss,roots[1] + tmp_first_edge + edge + row + line)
                        # 编辑 距离 的ratio 越大越好
                        infor_3 = Levenshtein.ratio(two_jump_solver_ss,roots[0] + tmp_first_edge + edge)
                        
                        if line not in ans_for_select:
                            #,np.sign(jump_hint[edge]) 这个特征好像比想象中重要
                            ans_for_select[line] = (num,abs(jump_hint[edge]),prop_count,infor_1,infor_2,infor_3,np.sign(jump_hint[edge]))
                        else:
                            aa = (num,abs(jump_hint[edge]),prop_count,infor_1,infor_2,infor_3,np.sign(jump_hint[edge]))
                            bb = ans_for_select[line]
                            if aa[0]*aa[3]*aa[4]*aa[5]/np.sqrt(aa[1]+1) > bb[0]*bb[3]*bb[4]*bb[5]/np.sqrt(bb[1]+1):
                                ans_for_select[line] = aa
"""

def two_jump_solver(two_jump_solver_ans,two_jump_solver_ss):
    
    """
    返回可能 的 答案 以及相关的特征
    """
    
    jump_hint = get_hint_from_seq(two_jump_solver_ss)
    #jump_hint = np.array(jump_hint)
    ans_for_select = {}
    ans_raw = {}
    for roots in two_jump_solver_ans.keys():

        G = two_jump_solver_ans[roots]
        if not len(G):
            continue

        first = set([ y for x,y in G.out_edges(roots[1])] + [x for x,y in G.in_edges(roots[1])]) - set([roots[1]])
        # 一级子节点
        for row in first:

            second = set(G[row]) - set([roots[1]])
            for line in second:

                edge = G.edges[row,line]["prop"]
                if  edge in jump_hint.keys():  
                    
                    target,num = local_les(two_jump_solver_ss,edge.split("_")[0])
                    try:
                        tmp_first_edge = G.edges[roots[1],row]["prop"]
                    except KeyError:
                        # 到这里说明边是反向的
                        tmp_first_edge = G.edges[row,roots[1]]["prop"]                        
                        
                    infor_1 = jaccard(two_jump_solver_ss,roots[0] + tmp_first_edge + edge)
                    
                    infor_2 = hint(two_jump_solver_ss,roots[1] + tmp_first_edge + edge + row + line)
                    
                    infor_3 = Levenshtein.ratio(two_jump_solver_ss,roots[0] + tmp_first_edge + edge)
                    
                    if tmp_first_edge in jump_hint.keys():
                        prop_count = 2
                    else:
                        prop_count = 1
                        
                    if line  not in ans_for_select:
                        ans_for_select[line] = (num,abs(jump_hint[edge]),prop_count,infor_1,infor_2,infor_3,np.sign(jump_hint[edge]))
                        ans_raw[line]= (roots,tmp_first_edge,row,edge,line)
                    else:
                        # 这里要合并具有相同桥接节点的
                        tmp_path = ans_raw.get(line)
                        tmp_f = ans_for_select.get(line)
                        #print(tmp_path)
                        if  tmp_path and tmp_path[2] == row:
                        
                            infor_1 = jaccard(two_jump_solver_ss,tmp_path[0][0] + roots[0] + tmp_first_edge + edge)
                            infor_1 = max(infor_1,tmp_f[3])
                            
                            infor_2 = hint(two_jump_solver_ss,tmp_path[0][1] + roots[1] + tmp_first_edge + edge + row + line)
                            
                            infor_3 = Levenshtein.ratio(two_jump_solver_ss,tmp_path[0][0] + roots[0] + tmp_first_edge + edge)
                            infor_3 = max(infor_3,tmp_f[5])
                            
                            ans_for_select[line] = (num,abs(jump_hint[edge]),prop_count,infor_1,infor_2,infor_3,np.sign(jump_hint[edge]))
                            
                            #tmp_roots = roots + tmp_path
                            
                            ans_raw[line] = (roots,tmp_first_edge,row,edge,line) + tmp_path[0]
                            
                            #print(row)
                        else:
                            aa = (num,abs(jump_hint[edge]),prop_count,infor_1,infor_2,infor_3,np.sign(jump_hint[edge]))
                            bb = ans_for_select[line]
                            ans_raw[line]= (roots,tmp_first_edge,row,edge,line)
                            if aa[0]*aa[3]*aa[4]*aa[5]/np.sqrt(aa[1]+1) > bb[0]*bb[3]*bb[4]*bb[5]/np.sqrt(bb[1]+1):
                                ans_for_select[line] = aa
                                
                                
    return ans_for_select,ans_raw

def down_to_low(down_to_low_ans):
    
    """
    通过启发式 的 参数 调整 限制 待选答案的数目,一定 程度上解决了样本严重不平衡的问题
    """
    
    values = []
    keys = []
    for key,value in down_to_low_ans.items():
        keys.append(key)
        values.append(value)
    aaa = []
    hh = np.array(values).astype("float32") 
    # 搜索能返回 合理候选答案规模的参数
    try:
        for sup in range(5,20,1):
            for inf in range(20,1,-1):
                index = np.logical_and(hh[:,1] < 0.1*sup*hh[:,1].mean(),hh[:,4] > 0.1*inf*hh[:,4].mean())
                #print(index)
                tmp = hh[index]

                if len(tmp)< 100:
                    #print(len(tmp))
                    keys = np.array(keys)
                    #print(keys
                    aaa = keys[index]
    except IndexError:
        pass
    if len(aaa):
        return aaa
    return keys
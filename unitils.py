import os
import re
import time
import queue
import urllib
import pickle
import pymysql
import requests
import threading 
import Levenshtein



import numpy as np
import pandas as pd
import networkx as nx
import jieba_fast as jieba
import multiprocessing as mp


from lxml import etree
from copy import deepcopy
from html.parser import unescape
from multiprocessing import Manager

from config import stop_words,entity_stop,placeholder

from pymysqlpool import ConnectionPool
config = {
 'pool_name': 'test',
 'host': '192.168.1.13',
 'port': 3306,
 'user': 'root',
 'password': 'root',
 'database': 'ccks',
 'use_dict_cursor':False   
}
def connection_pool():
 # Return a connection pool instance
    """
    with connection_pool().connection() as conn:
        cur = conn.cursor()
    #print(data)
    """
    pool = ConnectionPool(**config)
    pool.connect()
    return pool
# 这里的type必须是数据库里的标准表达方式
match_type_conn = pymysql.connect(host="192.168.1.13",user='root', password='root', database='ccks',charset='utf8')
match_type_cur = match_type_conn.cursor()

def get_entry(word,cur,top = 100):
    """
    接受字符形式的word
    """
    sql = "select * from `pkuorder` where `entry`='%s'" % word
    cur.execute(sql)
    data = cur.fetchall()
    return data[:min(top,len(data))]

def get_type(entry,cur):
    """
    entity 两边有尖括号,如果输入没有,自动补齐
    """
    if entry[0]=="<":
        sql = "select * from `pkutype` where `entry`='%s'" % entry
    else:
        sql = "select * from `pkutype` where `entry`='<%s>'" % entry
    cur.execute(sql)
    return cur.fetchall()
    

    
def from_entry(entry,cur):
    """
    entity 两边有尖括号,如果输入没有,自动补齐
    """
    if entry[0] in ['"',"<"]:
        sql = "select * from `pkubase` where `entry`='%s' " % entry
    else:
        sql = "select * from `pkubase` where `entry`='<%s>'" % entry
    cur.execute(sql)
    
    return cur.fetchall()

def from_prop(prop,cur):
    sql = "select * from `pkubase` where `prop`='<%s>'" % prop
    cur.execute(sql)
    return cur.fetchall()

def from_value(value,cur,word=False):
    """
    entity,value 两边分别有尖括号,双引号
    如果想指定为value,请设置 word = True
    如果想指定为entity,请设置 word = False
    """
    if value[0] in ['"',"<"]:
        sql = "select * from `pkubase` where `value`='%s .'" % value
    elif not word:
        sql = "select * from `pkubase` where `value`='<%s> .' " % value
    else:
        sql = "select * from `pkubase` where `value`='\"%s\" .' " % value
    cur.execute(sql)

    return cur.fetchall()
    
def get_father(node,graph,get_father_cur,get_father_seq = "",word=False):
    """
    为图中节点添加父节点,并返回父节点
    """
    father = []
    for row in from_value(node,get_father_cur,word = word):
        __,entry,prop,value = row
        value = value.strip(" .")
        #print(entry,local_jaccard(seq,entry))
        # 贪心算法
        #if local_jaccard(seq,entry) > 0 or local_jaccard(seq,prop) > 0:
        graph.add_edges_from([(entry,value,{"prop":prop})])
        father.append(entry)
    return graph,father

def get_child(node,graph,get_child_cur,get_child_seq = "",word=False):
    """
    为图中节点添加子节点,并返回子节点
    """
    childs = []
    for row in from_entry(node,get_child_cur):
        __,entry,prop,value = row
        value = value.strip(" .")
        #if local_jaccard(seq,value) > 0 or local_jaccard(seq,prop) > 0:
        graph.add_edges_from([(entry,value,{"prop":prop})])
        childs.append(value)
    return graph,childs

def jaccard(seqa,seqb):

    """
    返回两个句子的 jaccard 相似度 并没有 计算 字出现的次数
    """
    seqa = set(list(seqa.upper()))
    seqb = set(list(seqb.upper()))
    aa = seqa.intersection(seqb)
    bb = seqa.union(seqb)
    #return (len(aa)-1)/len(bb)
    return len(aa)/len(bb)


def search_entity(seq,search_entity_cur,max_str=True,top = 100):
    
    """
    暴力搜索实体,有包含关系时应选取最大子串
    设置 max_str = False 可以关闭返回最大字串的功能
    """
    
    entitys={}
    for i in range(len(seq)):
        for j in range(i+2,len(seq)+1):
            if seq[i:j] in entity_stop:
                continue
            out = get_entry(seq[i:j],search_entity_cur,top)
            if out:
                entitys[seq[i:j]] = [x for x in out if x[3]<1000]
    if max_str:
        keys = entitys.keys()
        remove = set()
        for i in keys:
            for j in keys:
                if i in j and i!=j:
                    remove.add(i)
                    break
        for key in remove:
            entitys.pop(key)
    return entitys

def search_word(seq ,search_word_cur,max_str = True):
    """
    暴力搜索实体,有包含关系时应选取最大子串
    设置 max_str = False 可以关闭返回最大字串的功能
    """
    entitys = []
    for i in range(len(seq)):
        for j in range(i+2,len(seq)+1):
            if seq[i:j] in entity_stop:
                continue
            out = from_value(seq[i:j],search_word_cur,word = True)
            if out:
                #entitys.append('"'+seq[i:j]+'"')
                entitys.append(seq[i:j])
    if max_str:
        remove = set()
        for i in entitys:
            for j in entitys:
                if i in j and i!=j:
                    remove.add(i)
                    break
        for key in remove:
            entitys.remove(key)
    return entitys


class baike:
    
    def __init__(self,word,msg = False):
        self._header = {
                "Host": "baike.baidu.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://baike.baidu.com"}
        self.word = word
        
        self._session = requests.Session()
        self._response = self._session.get(url = "https://baike.baidu.com/search/word?word=%s" % self.word,headers = self._header)
        self._response.close()
        self._session.close()
        
        if "none" not in self._response.url:
            try:
                self.entry = urllib.parse.unquote(re.findall("https://baike.baidu.com/item/(.*)\?fromtitle",\
                                                             self._response.url)[0]).split('/')[0]
            except:
                try:
                    self.entry = urllib.parse.unquote(re.findall("https://baike.baidu.com/item/(.*)",\
                                                             self._response.url)[0]).split('/')[0]
                except:
                    self.entry = ""
        else :
            self.entry = ""
    
    def get_msg(self):
                
        if "none" not in self._response.url:
            try:
                self._selector = etree.HTML(self._response.content)
                self._lis = self._selector.xpath("//div[@class = 'basic-info cmn-clearfix']")
                self._name =  [unescape(x.xpath('string(.)')).strip().replace("\xa0","") \
                              for x in self._lis[0].xpath("//dt[@class = 'basicInfo-item name']")]
                self._value = [unescape(x.xpath('string(.)')).strip().replace("\xa0","") \
                              for x in self._lis[0].xpath("//dd[@class = 'basicInfo-item value']")]
                self.msg = dict(zip(self._name,self._value))
            except:
                self.msg = {}
                
        else:
            self.msg = {}

    def get_mul(self):
        
        self._mul = self._selector.xpath("//ul[@class ='polysemantList-wrapper cmn-clearfix']")
        if self._mul:
            self.other_type = [ unescape(x.xpath('string(.)')).strip().replace("\xa0","")\
                                for x in self._mul[0].xpath("//li[@ class='item']//a")]
            self._type = self._mul[0].xpath("//li[@ class='item']//span")[0].text
        else:
            self._mul = []
            self._type = ""


def search_baike(seq,search_baike_timeout = 10):

    """
    利用百度百科实现常用实体的链接
    """
    
    class myThread (threading.Thread):
    
        def __init__(self, threadID):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.thread_stop = False 

        def run(self):
            retry = 0
            while not self.thread_stop:   
                try:  
                # 一定要用get_nowait 不然会阻塞
                    task=q.get_nowait()#接收消息  
                    try:
                        aa = baike(task)
                        #time.sleep(0.5)
                        if aa.entry:
                            #print(aa.entry)
                            search_baike_entitys[task] = aa.entry
                    except requests.exceptions.ConnectionError:
                        #time.sleep(2)
                        if retry <= 1:
                            time.sleep(1)
                            q.put_nowait(task)
                            retry += 1
                        #不做错误处理 容易 死循环
                        else:
                            break
                except queue.Empty:  
                    self.thread_stop=True  
                    break  
    search_baike_entitys = {}
    q = queue.Queue()
    seq = jieba.lcut(seq)
    for i in range(len(seq)):
        for j in range(i+1,min(len(seq)+1,i+4)):
            target = "".join(seq[i:j])
            #print(target)
            q.put(target)
    # 创建新线程
    start = time.time()
    thread1 = myThread(1)
    thread2 = myThread(2)
    thread3 = myThread(3)
    thread4 = myThread(4)
    #thread5 = myThread(5)

    # 开启新线程
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    #thread5.start()

    thread1.join(timeout = search_baike_timeout)
    thread2.join(timeout = search_baike_timeout)
    thread3.join(timeout = search_baike_timeout)
    thread4.join(timeout = search_baike_timeout)
    #thread5.join(timeout = search_baike_timeout)
    #print ("退出主线程")
    #print("用时:",time.time()-start)
    #return entitys
    ans = {}
    with connection_pool().connection() as conn:
    #conn = pymysql.connect(host="192.168.1.13",user='root', password='root', database='ccks',charset='utf8')
        cur = conn.cursor() 
        ssss = list(search_baike_entitys.items())
        try:
            for key,value in ssss:
            
                if key in entity_stop:
                    continue
                out = get_entry(value,cur)
                if out:
                    ans[key] = [x for x in out if x[3]<1000]
        except RuntimeError:
            pass
    #conn.close()
    
    return ans#,entitys
    
    
def count(word,count_cur):
    """
    word自动识别尖括号和引号的词
    """
    #word = word.replace("'","\\'").replace('"',"")
    try:
        num = 0
        if word[0] == "<" or word[0] == '"':
            sql = "select `count(0)` from pkuvalue where `value` = '%s'" % word
        else:
            sql = "select `count(0)` from pkuvalue where `value` = '<%s>' or `value` = '\"%s\"'" % (word,word)
        count_cur.execute(sql)
        data = count_cur.fetchall()
        
        # else 产生的sql语句可能返回多个结果
        for row in  data:
            num += row[0]
        
        #搜索 实体值
        if word[0] == "<":
            sql = "select `count(0)` from pkuentity where `entry` = '%s' " % word
        else:
            sql = "select `count(0)` from pkuentity where `entry` = '<%s>' " % word
        count_cur.execute(sql)
        data = count_cur.fetchall()
        if data:
            num +=data[0][0]
        return (word,num)
    except:
        return (word,100000)

def entry_two_jump(entry_two_jump_q):
    results = []
    
    with connection_pool().connection() as conn:
        #conn = pymysql.connect(host="192.168.1.13",user='root', password='root', database='ccks',charset='utf8')
        cur = conn.cursor()
        
        for raw,entry_two_jump_entry in entry_two_jump_q:
            
            node = "<" + entry_two_jump_entry + ">"
            tmp_graph = nx.DiGraph()
            

            #控制father的生长
            if count(entry_two_jump_entry.replace("'","\\'"),cur)[1] <= 5000:
                #print(row[2])
                tmp_graph,father = get_father(node.replace("'","\\'"),tmp_graph,cur)
                for line in father:
                #get_father(line,tmp_graph,seq)
                    try:
                        get_child(line,tmp_graph,cur)
                    except:
                        pass
            try:
                tmp_graph,child = get_child(node,tmp_graph,cur)
                for line in child:
                    try:
                        get_child(line,tmp_graph,cur)
                    except:
                        pass
            except:
                pass
            results.append((raw,node,tmp_graph))
    #conn.close()
            
    return results

def word_two_jump(word_two_jump_p):
    results = []
    
    with connection_pool().connection() as conn:
        #conn = pymysql.connect(host="192.168.1.13",user='root', password='root', database='ccks',charset='utf8')
        cur = conn.cursor()
        for raw,node in word_two_jump_p:
            tmp_graph = nx.DiGraph()
            #莫名的root节点丢失
            tmp_graph.add_node(node)
            #控制father的生长
            if count(row.strip('"'),cur)[1] <= 5000:
                tmp_graph,father = get_father(node,tmp_graph,cur)
                for line in father:
                #get_father(line,tmp_graph,seq)
                    try:
                        get_child(line,tmp_graph,cur)
                    except:
                        pass
                        
            results.append((raw,node,tmp_graph))
    
    return results
    


def test_mul(pq):
    while  not pq.empty(): 
        node = pq.get_nowait()
        print(node)
        
def get_ans(seq,get_ans_timeout = 20,get_ans_max_str = True):

    job_p = []
    job_q = []
    with connection_pool().connection() as conn:
        #conn = pymysql.connect(host="192.168.1.13",user='root', password='root', database='ccks',charset='utf8')
        cur = conn.cursor()
        tmp_entities = search_entity(seq,cur,get_ans_max_str)
    #print(tmp_entities)
    entities = []
    for key,value in tmp_entities.items():
        if key not in entity_stop and len(key)>=2:
            for line in value:
                #print(line)
                #if "汉语词汇" not in line:
                entities.append((key,line[2]))
                    
    
    baike = search_baike(seq,get_ans_timeout)
    
    if get_ans_max_str:
        keys = baike.keys()
        remove = set()
        for i in keys:
            for j in keys:
                if i in j and i!=j:
                    remove.add(i)
                    break
        for key in remove:
            baike.pop(key)
    
    for key,value in baike.items():
        if key not in entity_stop and len(key)>=2:
            for line in value:
                #print(line)
                #if "汉语词汇" not in line:
                entities.append((key,line[2]))
    
    
    

    
    tmp_value = search_word(seq,cur,get_ans_max_str)
    
    
    
    conn.close()        
    
    entities_graph = {}
    
    job_q = set(entities)
    pool_results = entry_two_jump(job_q)
    for raw,node,graph in pool_results:
        entities_graph[(raw,node)] = graph
    
    for row in tmp_value:
        node = "\"" + row + "\""
        node = node.replace('""','"')
        job_p.append((node,node))
    pool_results = word_two_jump(job_p)
    for raw,node,graph in pool_results:
        entities_graph[(raw,node)] = graph
    
    return entities_graph

def hint(seqa,seqb):
    seqa = set(list(seqa.upper()))
    seqb = set(list(seqb.upper()))
    aa = seqa.intersection(seqb)
    return len(aa)
    


def search_prop(seq ,search_word_cur,reverse = False):
    """
    暴力搜索谓词,有包含关系时应选取最大子串
    设置 max_str = False 可以关闭返回最大字串的功能
    """
    entitys = {}
    for i in range(len(seq)):
        for j in range(i+1,len(seq)+3):
            #print(seq[i:j])
            # 正序搜索
            sql = "select * from pkuprop where prop like '<%s>' or prop like '<%s_>' or prop like '<%s__>' " \
               % (seq[i:j],seq[i:j],seq[i:j])    
            search_word_cur.execute(sql)
            out = search_word_cur.fetchall()
                
            if out:
                #entitys.append('"'+seq[i:j]+'"')
                if seq[i:j] not in entitys:
                    entitys[seq[i:j]] = list(out)
                else:
                    entitys[seq[i:j]].extend(out)
            if reverse :
                sql = "select * from pkuprop where rprop like '>%s<' or rprop like '>%s_<' or rprop like '>%s__<' " \
                % (seq[i:j][::-1],seq[i:j][::-1],seq[i:j][::-1])    
                search_word_cur.execute(sql)
                out = search_word_cur.fetchall()
                #print(out)
                if out:
                    if seq[i:j] not in entitys:
                        entitys[seq[i:j]] = list(out)
                    else:
                        entitys[seq[i:j]].extend(out)
    return entitys
#insert into pkuprop select prop,reverse(prop) as rprop,`count(0)` from pkueprop;
# search_prop(ss,match_type_cur,True)

with open("task4coqa_train.txt","r",encoding="utf8") as f:
    qa = f.read()
qa_pair = re.split("\n\n",qa)
anses = []
for row in qa_pair[:-1]:
    question,sparql,answer=row.split("\n")
    
    #删除两侧的多余空格
    answer = answer.strip()
    
    #答案有时会少了引号,导致和数据库对不上,我们手动加
    
    if answer[0]!="<" and answer[0]!='"':
        answer = '"'+answer+'"'
    
    question = re.sub("(q\d{1,4}:|？)","",question)
    
    anses.append((question,sparql,answer))
dd = pd.DataFrame(anses,columns=["question","sparql","answer"])

def pre_clean(raw):
    out = []
    if isinstance(raw,list):
        tmp = raw
    else:
        tmp = raw.split("\t")
    for row in tmp:
        out.append(row.lstrip(' "“”').rstrip(' "“”').replace("（","(").replace("）",")"))
    return out

def ans_clean(raw):
    out = []
    if isinstance(raw,list):
        tmp = raw
    else:
        tmp = raw.split("\t")
    for row in tmp:
        out.append(row.lstrip(' "“”').rstrip(' "“”').replace("（","(").replace("）",")"))
    return out

def p_value(p_value_pre,p_value_ans):
    return len(set(p_value_pre).intersection(set(p_value_ans)))/len(set(p_value_pre))

def r_value(r_value_pre,r_value_ans):
    return len(set(r_value_pre).intersection(set(r_value_ans)))/len(set(r_value_ans))

def f1(f1_pre,f1_ans):
    rr = r_value(f1_pre,f1_ans)
    pp = p_value(f1_pre,f1_ans)
    return 2*rr*pp/max(0.00000001,rr + pp)

def score(dd,file_path = "ans.txt"):
    ans = open(file_path,"rb").read().decode("utf8").split("\n")[:-1]
    fff = 0
    for line in ans:
        # 答案前4位作为题号
        num = line[:4]
        pre_line = line[4:]
        pre = pre_clean(pre_line)
        ans = ans_clean(dd[num])
        fff += f1(pre,ans)
    return fff/len(ans)
with open("task4coqa_validation.questions.txt","r",encoding="utf8") as f:
    ques = f.read()
ques = re.split("\n",ques)

# 提取 图中 答案的提示 
# 主要是边
# 这个只选取了 双跳节点 连着的边
def get_hint(get_hint_ss,graph,get_hint_root):
    
    # 取图中谓词 和 句子中谓词的交
    ans_prop = []
    for row in graph.edges:
        ans_prop.append(graph.edges[row]["prop"])
    #  
    
    ans_hint = []
    for row in placeholder:
        try:
            index = get_hint_ss.index(row)
            #print(index)
            seq_hint = search_prop(get_hint_ss[index+len(row):],match_type_cur)
            
            for key,value in seq_hint.items():
                for line,__ in value:
                    #print(row)
                    if line in ans_prop:
                        # 编码位置信息
                        ans_hint.append((line,get_hint_ss[index+len(row):].index(line[1])))
            
            seq_hint = search_prop(get_hint_ss[:index],match_type_cur)
            
            for key,value in seq_hint.items():
                for line,__ in value:
                    #print(row)
                    if line in ans_prop:
                        # 编码位置信息
                        att_index = get_hint_ss[:index].rindex(line[1]) - index
                        ans_hint.append((line,att_index))
                        #ans_hint.append((row,index - get_hint_ss.index(row[1])))
            ans_hint = list(set(ans_hint))
            ans_hint.sort(key = lambda x:x[1])
            break
        except ValueError:
            continue
            
    return ans_hint

# index 有左右 之分 取min？
# 答案的提示
# 默认打开反向索引
def get_hint_from_seq(get_hint_from_seq_ss,reverse = True):
    
    """
    找出 占位符附近 可能关于答案相连的边的信息，需要注意 在 占位符 的左边 和右边 是不一样的
    用 正负 表示
    """
    
    ans_hint = {}
    for row in placeholder:
        #print(row)
        try:
            # index or rindex is a problem
            index = get_hint_from_seq_ss.index(row)
            #print(index)
            #print(get_hint_from_seq_ss[index+len(row):])
        except ValueError:
            continue
            
        seq_hint = search_prop(get_hint_from_seq_ss[index+len(row):],match_type_cur,reverse)

        for key,value in seq_hint.items():
            for line,__,___ in value:
                #print(row)
                try:
                    #ans_hint.setdefault(line,100)
                    #min(get_hint_from_seq_ss[index+len(row):].index(line[1]),ans_hint[line])
                    ans_hint[line] =  get_hint_from_seq_ss[index+len(row):].index(key)
                except:
                    continue

        seq_hint = search_prop(get_hint_from_seq_ss[:index],match_type_cur,reverse)
        #print(get_hint_from_seq_ss[:index])
        for key,value in seq_hint.items():
            for line,__,___ in value:
                #print(row)
                try:
                    #ans_hint.setdefault(line,-100)
                    # 不计较词的长度 用line[-2]
                    att_index = get_hint_from_seq_ss[:index].rindex(key) - index
                    ans_hint[line] = att_index
                except ValueError:
                    continue
        #ans_hint.sort(key = lambda x:x[1])
        return ans_hint
        
    return ans_hint
# get_hint_from_seq(ss)
def local_les(local_les_seq,local_les_word):
    step = len(local_les_word.strip("<>"))
    # 无法应对缩写 尝试 + 2 解决 官网 官方网站的问题 local_les_seq[x:x + step + 2]
    #Levenshtein.ratio("官方网站","官网")*hint("官方网站","官网")
    # 越大越好
    tmp_score = list(map(lambda x: Levenshtein.ratio(local_les_seq[x:x + step],local_les_word)\
                         *hint(local_les_seq[x:x + step],local_les_word), range(len(local_les_seq) - step + 1)))
    score_min = max(tmp_score)
    index = tmp_score.index(score_min)
    return local_les_seq[index:index + step + 2],score_min

def common_node(common_node_G,common_node_left,common_node_right):
    
    """
    返回有向图中 两个 节点 的 一步 相连 的 公共 节点
    """
    
    aa = set([x for x,y in common_node_G.in_edges(common_node_left)] + [y for x,y in common_node_G.out_edges(common_node_left)])
    bb = set([x for x,y in common_node_G.in_edges(common_node_right)] + [y for x,y in common_node_G.out_edges(common_node_right)])
    return aa.intersection(bb)

    


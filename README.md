# ccks2018
CCKS 2018 开放领域的中文问答任务 1st 解决方案
Keenpower队采用了相似匹配求解、神经网络、ensemble等技术解决开放域基于pkubase知识图谱的问答问题，取得了一定效果。由于所使用的训练数据和软件较大，我们的相关材料放在百度网盘中，可按照文中给出的链接访问。

# 模型构造

```flow
graph TD;
seq:(原始问句) --> entity:(预解空间);
entity:-->one:(单跳求解器);
entity: -->two:(双跳求解器);
seq:--> pares:(问句解析);
pares:-->main_entry:(主实体发现);
main_entry:--> one_1:(1-度问题求解);
one_1:-->two_2:(2-度问题求解);
two_2: -->other:(文本相似匹配问答模型:0.6)
one:--> ans:(两类预测答案);
two:--> ans:;
ans: --> CNN;
ans: --> num_feature:(数量特征);
num_feature: -->final_model:;
CNN --> char_feature:(相似度特征);
char_feature:-->final_model:(感知机分类:0.48);
other: --> merge:(答案融合);
final_model: -->merge:;
merge: -->final_ans:(最终答案:0.66);

```


# 结果融合

所需文件</br>

链接: https://pan.baidu.com/s/1zspZx5BxMFPwN_xjYFD0mg 密码: 8ub7

只需要构建pkutype数据库(方法在pkubase.ipynb中),修改untils中的数据库路径,然后运行merge.ipynb即可(会读取之前感知机分类生成的中间结果)</br>

result_ss.txt(感知机分类)和result.test.txt(文本相似匹配问答模型模型)的F1大概为0.48,0.6.result_new(文本相似匹配问答模型的老版本)的F1没有测试.但是观察发现,虽然这个模型在有些简单问题上出了错,但在某些问题上,当前两个文件犯错时它却给出了正确答案,所以这里也作为融合的一部分

```
result_ss.txt
result.test.txt
result_new.txt
path.test.txt
```

### 强制类型修正

根据问题的提示词("是谁,什么时间,哪个国家"等等),确定答案的类型,优先选择符合类型的.</br>
用人物来举例,像董卿有个属型为"<类型><人物>",从而可以判断这个实体为人物.事实上,属性是需要重建的,pkutype里面的类型有错误信息,我们可以用迭代的方式重建类型,清洗噪声.具体来说,从"<人物>"这个属性出发,我们可以得到一堆实体,然后这一堆实体又会产生一堆谓词P,我们对谓词进行排序,并使用一些类似tf-idf的特征筛选掉常见特征(如<中文名>),这样我们就建立了"<人物>"这个类型到谓词的映射,我们反过来可以通过谓词集合来重建pkutype,达到消除噪声的目的.而这些计算,利用Hive on Spark可以很快地实现.

### 比较查询路径

产生答案的一系列三元组与问题符合的很好(信息不多也不少),那么优先选择该问题

### 投票

如果有两个文件选择了同一个答案,那么最终答案就选择她.

### 融合效果
之前将(0.48,0.51)融合到了0.6,然后在不改代码的前提下将(0.48,0.6)融合到了0.66,这说明融合有提升.




# 文本相似匹配问答模型 


# 感知机分类

<font color ="red"> 如果使用缓存的预解空间文件,可以不导入pkubase,pkuvalue.构建pkuprop,pkutype表单即可.(方法在pkubase.ipynb中)</br></font>
运行  感知机分类.ipynb  即可获得答案.运行train_model.ipynb可以训练文件(至少需要pkuprop表单)
具体细节请查看"感知机分类.pdf"

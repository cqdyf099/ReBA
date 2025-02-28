# !/usr/bin/env python
# coding: utf-8
# version：2024.08.12
# yongqiang cai

import re
import torch
import numpy as np

import sys
# sys.path.append("/home/yqcai/lightmann") # ~/lightmann does not work?
sys.path.append(".")


import pickle
def data_save(data,filename):
    f = open(filename,"wb") # .dat
    pickle.dump(data,f)
    f.close()
    
def data_load(filename):
    return pickle.load(open(filename,"rb")) 

# 找出子串的所有位置
def find_all_substring_positions(text, substring):
    '''
    Example:
    text = "Python is a powerful programming language. Python is also easy to learn."
    substring = "Python"
    positions = find_all_substring_positions(text, substring)
    print(positions) # [0, 43]
    '''
    positions = [match.start() for match in re.finditer(substring, text)]
    return positions
 
# 找出词在 token 序列中的位置
# 本来可以用 find_all_substring_positions 来把代码写得觉得一些，先用 DYF 之前的版本 word_in_sentence_tk_all
# '综合性医疗网站，提供妇产科小儿科内科以及外科资讯和讨论。'.find('大')
# ''.join(sentence_list).find('大')
def get_index_of_word_in_tokens(target_word, tokens):
    '''
    Example:
    word = '小儿科'
    tokens = ['提供','妇','大小','儿'wa,'科','小儿科','科内','科以及外科资讯和讨论。']
    word_in_sentence_tk(word, tokens) # [[2, 3, 4], [5]]
    # 如果没有出现目标词，则返回的是 []
    '''
    
    k = 0
    cur_word = ''
    indexes = []
    index = []

    break_flag = False
    for i, words in enumerate(tokens):
        if break_flag:
            k = 0
            cur_word = ''
            index = []
        for word in words:
            if k < len(target_word) and word == target_word[k]:
                break_flag = False
                cur_word += word
                k += 1
                index.append(i)
                if k == len(target_word):
                    indexes.append(list(set(index)))
                    break_flag = True
                    break
            elif k < len(target_word) and word == target_word[0]:
                break_flag = False
                cur_word = word
                k = 1
                index = [i]
                if k == len(target_word):
                    indexes.append(list(set(index)))
                    break_flag = True
                    break
            else:
                break_flag = False
                cur_word = ''
                k = 0
                index = []
    return indexes


##

# 计算余弦相似度矩阵
def similarity_matrix_cos(vs):
    '''e.g. input: vs.shape = torch.Size([4, 4096])
    out.shape = torch.Size([4, 4])
    '''
    vs_unit = vs / torch.norm(vs, dim=-1, keepdim=True) # 归一化
    return vs_unit @ vs_unit.T # 得到的是相似矩阵

def distance_matrix(vs):
    '''e.g. input: vs.shape = torch.Size([4, 4096])
    out.shape = torch.Size([4, 4])
    '''
    n = vs.shape[0]
    M = [[torch.linalg.norm(vs[i] - vs[j]) for i in range(n)] for j in range(n)]
    return torch.from_numpy(np.array(M))

# def distance_matrix(vs):
#     '''
#     e.g. input: vs.shape = torch.Size([4, 4096])
#     out.shape = torch.Size([4, 4])
#     '''
#     # 计算两两向量之间的欧氏距离，利用广播机制
#     # vs.unsqueeze(0) 使得 vs 的维度变为 (1, n, d)
#     # vs.unsqueeze(1) 使得 vs 的维度变为 (n, 1, d)
#     # 然后利用广播，直接计算 (n, n, d) 的张量
#     diff = vs.unsqueeze(0) - vs.unsqueeze(1)
    
#     # 计算每对向量的欧氏距离，沿着最后一个维度求范数
#     dist_matrix = torch.linalg.norm(diff, dim=-1)
    
#     return dist_matrix

 




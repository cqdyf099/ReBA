# !/usr/bin/env python
# coding: utf-8
# version：2024.08.12

import re
import torch
import numpy as np

import sys
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

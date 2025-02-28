import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import sys



def get_model_token(model_name):
    if model_name == 'gpt2':
        from transformers import BertTokenizerFast, AutoModel
        model_dir1 = '../model/bert'
        model_dir = '../model/gpt'
        tokenizer = BertTokenizerFast.from_pretrained(model_dir1)
        model = AutoModel.from_pretrained(model_dir, device_map="auto")
        return model, tokenizer
    if model_name == 'bert':
        
        from transformers import BertTokenizerFast, AutoModel
        
        model_dir = '../model/bert'
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_dir).to(device)
        
        return model, tokenizer
    if model_name == 'bert-wwm':
        from transformers import BertTokenizerFast, AutoModel
        model_dir = 'models/hfl-chinese-bert-wwm'
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_dir).to(device)
        return model, tokenizer
    if model_name == 'llama2':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_path = "../model/llama2"
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",load_in_8bit=True)
        return model, tokenizer
    if model_name == 'llama3':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_path = "models/Llama3-chinese"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", load_in_8bit=True)
        return model, tokenizer
    if model_name == 'baichuan':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        root = 'models/baichuan-inc/Baichuan-7B'
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(root, device_map="auto", trust_remote_code=True, load_in_8bit=True)
        return model, tokenizer
    if model_name == 'chatglm3':
        from transformers import AutoTokenizer, AutoModel
        model_path = "models/THUDM/chatglm3-6b-base-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        return model, tokenizer
    if model_name == 'chatglm4':
        from transformers import AutoTokenizer, AutoModel
        model_path = "models/THUDM/glm4"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        return model, tokenizer
    if model_name == 'qwen':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = 'models/Qwen-cai'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, load_in_8bit=True)#.eval()
        return model, tokenizer

## 在token意义下返回某个词在句子中的所有位置(索引)
def word_in_sentence_tk_all(target_word, sentence_list):
    k = 0
    cur_word = ''
    indexs = []
    index = []

    break_flag = False
    for i,words in enumerate(sentence_list):
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
                    indexs.append(list(set(index)))
                    break_flag = True
                    break
            elif k < len(target_word) and word == target_word[0]:
                break_flag = False
                cur_word = word
                k = 1
                index = [i]
                if k == len(target_word):
                    indexs.append(list(set(index)))
                    break_flag = True
                    break
            else:
                break_flag = False
                cur_word = ''
                k = 0
                index = []
    #indexs = list(set(indexs))
    return indexs

## 找出目标词在句子中的索引，即二维数组里找出包含目标数组的所有索引
def word_in_sentence_tk(target_word, sentence_list):
    # 示例：
    # target_word = '小儿科'
    # sentence_list = ['提供','妇','小儿科','科内','科以及外科资讯和讨论。']
    # return [2]
    # 注意：sentence_list 依赖tokenizer分词器的结果，该过程不可控，该函数考虑了这一点，如果 sentence_list = ['提供','妇小', '儿科','科内','科以及外科资讯和讨论。']，返回的索引将是[1,2]，可以自己做做实验
    k = 0
    cur_word = ''
    indexs = []

    break_flag = False
    for i,words in enumerate(sentence_list):
        if break_flag:
            break
        #print(i)
        for word in words:
            if k == len(target_word):
                break_flag = True
                break
            if k < len(target_word) and word == target_word[k]:
                cur_word += word
                # print(k,cur_word)
                k += 1
                # print(k,target_word[k])
                indexs.append(i)
            elif k < len(target_word) and word == target_word[0]:
                cur_word = word
                k = 1
                indexs=[i]
            else:
                cur_word = ''
                k = 0
                indexs = []
    indexs = list(set(indexs))
    return indexs

# target_word = '小儿科'
# sentence_list = ['提供','妇小','儿科','科内','科以及外科资讯和讨论。']
# print(word_in_sentence_tk(target_word,sentence_list))


def get_word_embedding(word, sentence, model, tokenizer, model_name):
    if model_name == 'gpt2' or model_name == 'bert' or model_name == 'bert-wwm':
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        input_ids = input_ids.to(model.device)

        out = model(input_ids)
        #print(out)
        hidden_states = out.last_hidden_state
        #print(hidden_states.shape)
        # word_ids = tokenizer.encode(word, return_tensors='pt')[0,1:-1]
        
        # indexs = list_in_list(word_ids,input_ids[0])
        
        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        indexs = word_in_sentence_tk(word, sentence_list)
        # print(indexs,sentence_list)
        if len(indexs)==0:
            print('没有找到词:', word,sentence)
            return []
        word_embedding = torch.mean(hidden_states[0][indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    elif model_name == 'llama': #由于llama词汇量太少，会有多个token对应一个字的情况，所以在定位词向量时需要挑出它对应的所有token
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        out = model(input_ids ,output_hidden_states=True)
        hidden_states = out.hidden_states
        word_ids = tokenizer.encode(word, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')

        true_ids = torch.tensor([id for id in word_ids[0] if tokenizer.decode(id) != ''])

        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]

        word_list = [tokenizer.decode(id) for id in true_ids]
        word_list = ''.join(word_list)

        indexs = word_in_sentence_tk(word_list, sentence_list)

        word_embedding = torch.mean(hidden_states[-1][0][indexs],0)
        return word_embedding
    elif model_name == 'llama2' or model_name == 'llama2-chat' or model_name == 'llama3':
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        input_ids = input_ids.to(model.device)
        out = model(input_ids ,output_hidden_states=True)
        hidden_states = out.hidden_states
        
        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        # print(sentence_list)
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            print('没有找到词:')
            print(word,sentence_list)
            return []
        word_embedding = torch.mean(hidden_states[-1][0][indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    elif model_name == 'chatglm2' or model_name == 'chatglm3' or model_name == 'chatglm4':
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        input_ids = input_ids.to(model.device)
        out = model(input_ids ,output_hidden_states=True)
        
        hidden_states = out.hidden_states
        if model_name == 'chatglm4':
            sentence_list = [tokenizer.decode([id]) for id in input_ids[0]]
            squeeze_index = 0
        else:
            sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
            squeeze_index = 1
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            print(word, sentence_list)
            return []
        word_embedding = torch.mean(hidden_states[-1].squeeze(squeeze_index)[indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    elif model_name == 'baichuan' or model_name == 'qwen':
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        input_ids = input_ids.to(model.device)
        out = model(input_ids ,output_hidden_states=True)
        hidden_states = out.hidden_states

        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        #print(sentence_list)
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            print('没有找到词:')
            print(word,sentence_list)
            return []
        word_embedding = torch.mean(hidden_states[-1][0][indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    
def get_word_embedding_multilayer(word, sentence, layer, model, tokenizer, model_name):
    if model_name == 'gpt2' or model_name == 'bert':
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        input_ids = input_ids.to(model.device)

        out = model(input_ids,output_hidden_states = True)
        #print(out)
        # hidden_states = out.last_hidden_state
        hidden_states = out.hidden_states
        # print(len(hidden_states),hidden_states[0].shape)
        if layer > len(hidden_states) - 1:
            print('错误, layer应该小于:', len(hidden_states))
            return []
        # print(hidden_states.shape)
        # word_ids = tokenizer.encode(word, return_tensors='pt')[0,1:-1]
        
        # indexs = list_in_list(word_ids,input_ids[0])
        
        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            print('没有找到词:', word,sentence)
            return []
        word_embedding = torch.mean(hidden_states[layer][0][indexs],0).cpu().detach().to(torch.float)
        # word_embedding = torch.mean(hidden_states[0][indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    elif model_name == 'llama2':
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        out = model(input_ids ,output_hidden_states=True)
        hidden_states = out.hidden_states
        if layer > len(hidden_states) - 1:
            print('错误，layer应该小于:', len(hidden_states))
            return []
        
        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        # print(sentence_list)
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            print('没有找到词:')
            print(word,sentence_list)
            return []
        word_embedding = torch.mean(hidden_states[layer][0][indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    elif model_name == 'chatglm2' or model_name == 'chatglm3' or model_name == 'chatglm4':
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        input_ids = input_ids.to(model.device)
        out = model(input_ids ,output_hidden_states=True)
        hidden_states = out.hidden_states

        if layer > len(hidden_states) - 1:
            print('错误, layer应该小于:', len(hidden_states))
            return []

        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        #print(sentence_list)
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            return []
        word_embedding = torch.mean(hidden_states[layer].squeeze(1)[indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    elif model_name == 'baichuan' or model_name == 'qwen':
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_attention_mask=False, return_tensors='pt')#
        input_ids = input_ids.to(model.device)
        out = model(input_ids ,output_hidden_states=True)
        hidden_states = out.hidden_states

        sentence_list = [tokenizer.decode(id) for id in input_ids[0]]
        #print(sentence_list)
        indexs = word_in_sentence_tk(word, sentence_list)
        if len(indexs)==0:
            print('没有找到词:')
            print(word,sentence_list)
            return []
        word_embedding = torch.mean(hidden_states[layer][0][indexs],0).cpu().detach().to(torch.float)
        return word_embedding
    


if __name__ == '__main__':
    # 检查是否有足够的命令行参数
    if len(sys.argv) < 2:
        print("请提供至少一个参数。输入格式为python xxx.py model_name")
        print('本代码仅作为加载模型使用，支持bert,gpt2,llama2,baichuan,qwen,chatglm3')
        sys.exit(1)
    
    # 获取第一个参数
    model_name = sys.argv[1]
    
    print(f"正在调用模型: {model_name}")
    model, tokenizer = get_model_token(model_name)
    print(f'{model_name}调用成功,下面尝试找出词向量')
    word = 'hello'
    sentence = 'hello world'
    result = get_word_embedding(word, sentence, model, tokenizer, model_name)
    print('词向量计算成功，可以正常使用',result[0:5],result.shape)
    # print('本代码仅作为加载模型使用，支持bert,gpt2,llama2,baichuan,qianwen,chatglm2,chatglm3')
from mteb import MTEB
from C_MTEB import ChineseTaskList ,load_retrieval_data
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import os
from prettytable import PrettyTable
import torch
from Code.models import LLM
from requests.exceptions import ConnectionError
import pandas as pd
from tqdm import tqdm
import numpy as np

os.environ['HF_ENDPOINT'] = ''  # 禁用网络连接

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
    
# ReBA编码 mean pool
class LLM_ours_mean_pool(LLM):
    def get_attention_matrix(self, outputs):
        A = 0
        for layer in range(len(outputs.attentions)):
            X = outputs.attentions[layer][0].to(torch.float32).cpu().detach().numpy()
            for i in range(len(X)):
                Ai = (X[i] + X[i].T)/2
                A = (A+Ai)/2 + abs(A-Ai)/2 # max
            # plt.matshow(A, cmap=plt.cm.Reds);
        A = np.triu(A) # **** 修改
        # 对每一列求和，最多求到第n行，2n是矩阵A的行数
        A = A[:len(A)//2].sum(axis=0)
        return A # 2n维的向量，n是句子长度
    
    def get_sentence_embedding(self, sentence, repeat=1):
        self.repeat = repeat
        last_index = min(len(self.tokenize(sentence)[0]) - 1, self.control_num - 1)
        if repeat > 1:
            sentence = sentence * repeat # 重复
        input_ids = self.tokenize(sentence)
        if len(input_ids[0]) > self.control_num:
            input_ids = input_ids[:,:self.control_num]
        with torch.no_grad():
            out = self.model(input_ids, 
                             output_hidden_states=True,
                             output_attentions=True
                            )
            hidden_states = out.hidden_states[-1]

        em = hidden_states[0].cpu().detach().to(torch.float)
        weights = self.get_attention_matrix(out)
        # 归一化
        weights = weights / sum(weights)
        # 返回隐藏层状态的加权平均值
        result = (1 / (len(weights)//2)) * np.average(em, axis=0, weights=weights)
        return torch.tensor(result).to(self.device)

# baseline mean pool
class LLM_mean(LLM):
    def get_sentence_embedding(self, sentence, repeat=1):
        self.repeat = repeat
        # 对输出求平均
        input_ids = self.tokenize(sentence)
        if len(input_ids[0]) > self.control_num:
            input_ids = input_ids[:,:self.control_num]
        with torch.no_grad():
            out = self.model(input_ids, 
                             output_hidden_states=True
                            )
            hidden_states = out.hidden_states[-1]

        em = hidden_states[0].cpu().detach().to(torch.float)
        result = em.mean(dim=0)
        return result
    
# echo编码 mean pool
class LLM_mean_echo(LLM): #
    def get_sentence_embedding(self, sentence, repeat=1):
        self.repeat = repeat
        if repeat > 1:
            sentence = sentence * repeat # 重复
        last_index = min(len(self.tokenize(sentence)[0]) - 1, self.control_num - 1) # 原句子最后一个token的索引
        input_ids = self.tokenize(sentence)
        if len(input_ids[0]) > self.control_num:
            input_ids = input_ids[:,:self.control_num]
        with torch.no_grad():
            out = self.model(input_ids, 
                             output_hidden_states=True
                            )
            hidden_states = out.hidden_states[-1]

        em = hidden_states[0].cpu().detach().to(torch.float)
        result = em[last_index : , : ].mean(dim=0)
        return result

class EmbeddingModelWrapper:
    def __init__(self, tokenizer, model, llm, repeat, last_token, attention):
        self.llm = llm
        self.tokenizer = tokenizer
        self.model = model

        self.repeat = repeat
        self.last_token = last_token
        self.attention = attention
        if 'gpt' in llm.model_name:
            self.llm.control_num = 512
        else:
            self.llm.control_num = 1024
        self.output_folder = None

    def tokenize(self,sentence):
        '''获得单个句子的 token 数字序列'''
        input_ids = self.tokenizer.encode(sentence, 
                                          return_tensors='pt', # bert
                                          add_special_tokens=True, 
                                          return_attention_mask=False, # llama2
                                         )  # [101,...,102]
        return input_ids.to(self.model.device)

    def encode(self, texts, batch_size=8, convert_to_tensor=False, show_progress_bar=True):
        repeat = self.attention # 句子重复次数
        last_token = self.last_token # 是否使用最后一个词的编码作为词编码，若为True，则使用
        attention = self.attention # attention=True表示使用ReBA编码

        if attention:
            self.output_folder = f"results/{model_name}_ourslast_{repeat}_model({self.llm.control_num})"
        elif repeat > 1:
            self.output_folder = f"results/{model_name}_echolast_{repeat}_model({self.llm.control_num})"
        elif last_token:  # repeat == 1 and last_token is True
            self.output_folder = f"results/{model_name}_last{repeat}_model({self.llm.control_num})"
        else: #mean
            self.output_folder = f"results/{model_name}_mean{repeat}_model({self.llm.control_num})"
        # print('output_folder:', self.output_folder)


        embeddings = []
        iterator = tqdm(texts) if show_progress_bar else texts
        for text in iterator:
            # embedding = self.get_sentence_embedding(text, self.control_num, repeat=repeat, last_token=last_token, attention = attention)
            embedding = self.llm.get_sentence_embedding(text, repeat=repeat)
            contains_nan = torch.isnan(embedding).any() # 异常值处理，如果有NaN，将该句子的embedding设置为0
            if contains_nan:
                print(f"包含NaN的句子: {text}")
                embedding = torch.zeros(embedding.shape).to(embedding.device)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings).squeeze(1)
        # 必须要转换到cpu上
        embeddings = embeddings.cpu().detach().to(torch.float)
        # Error while evaluating QBQTC: Input contains NaN. 遇到错误，评估被跳过: Input contains NaN.
        # print(embeddings.shape)
        # 如果有NaN，返回0

        return embeddings

if __name__ == '__main__':
    # data_path = '/home/yfduan/word2fun/C_MTEB_data'
    data_path = '/home/yfduan/word2fun/C_MTEB_data'
    model_name = 'llama2' ##### 可修改参数：待选模型，llama2, qwen, baichuan, falcon
    
    llm = LLM(model_name=model_name) #可修改，当
    model = llm.model
    tokenizer = llm.tokenizer
    model.eval()

    embedding_model = EmbeddingModelWrapper(tokenizer, model, llm)

    tasks = ChineseTaskList
    # retrieval_tasks = ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval']
    evaluation = MTEB(tasks=tasks)
    skipped_tasks = []

    for task in evaluation.tasks:
        try:
            if task.description.get('type') == 'Retrieval':
                task.corpus, task.queries, task.relevant_docs = load_retrieval_data(
                    os.path.join(data_path, task.description["hf_hub_name"]),
                    task.description['eval_splits']
                )
            else:
                dataset = load_dataset(
                    path=os.path.join(data_path, task.description["hf_hub_name"]),
                    revision=task.description.get("revision", None),
                )
                task.dataset = dataset
            task.data_loaded = True
        except (FileNotFoundError, ConnectionError):
            print(f"数据集未找到或连接错误，跳过任务：{task.description.get('name')}")
            skipped_tasks.append(task.description.get('name'))
            continue


    evaluation.tasks = [task for task in evaluation.tasks if task.description.get('name') not in skipped_tasks]
    print('任务长度:', len(evaluation.tasks))

    output_f = embedding_model.output_folder
    print("输出文件夹:", output_f)
    try:
        results = evaluation.run(embedding_model, output_folder=f"results/{model_name}_ours_mean{2}_model(not divide by k)")
        # results = evaluation.run(embedding_model, output_folder=output_f)
        print("评估结果:", results)
    except Exception as e:
        print(f"遇到错误，评估被跳过: {e}")

    if skipped_tasks:
        print("以下任务因数据集文件缺失或连接错误而被跳过:\n", "\n".join(skipped_tasks))

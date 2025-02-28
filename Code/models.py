import os
import pandas as pd
import transformers
import torch
import numpy as np
from tqdm.contrib import tzip
import sys
for path in ['./','../','../../',]:
    sys.path.append(path+"Code")
from utinity import *
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Setting eos_token is not supported")
warnings.filterwarnings("ignore", message="Setting pad_token is not supported")
warnings.filterwarnings("ignore", message="Setting unk_token is not supported")

class Problem:
    def __init__(self,data_csv="Data/SLPWC_v1.csv",name='SLPWC'):
        
        df = pd.read_csv(data_csv)

        import re
        try:
            self.words = [re.findall(r'(?<=“)(.*?)(?=”)', sentence)[0] for sentence in df['问题'].values]

            df['目标词'] = self.words
        except:
            self.words = df['目标词'].values
        self.question = df[['选项1','选项2','选项3','选项4']].values
        self.answer = df['答案'].values
        self.n_question = len(self.question)
        self.df = df
        self.name = name

class Problem_3: 
    def __init__(self,data_csv="Data/SLPWC_v1.csv",name='SLPWC'):
        
        df = pd.read_csv(data_csv)

        df = df.drop(df[df['答案']=='D'].index)

        import re
        self.words = [re.findall(r'(?<=“)(.*?)(?=”)', sentence)[0] for sentence in df['问题'].values]
        self.question = df[['选项1','选项2','选项3']].values

        self.answer = df['答案'].values
        self.n_question = len(self.question)
        self.df = df
        self.name = name

        df['目标词'] = self.words
        self.questionW = df[['选项1','选项2','选项3','目标词']].values

ROOTPATH_Models = '/home/yfduan/word2fun/model/'
PATH_Models = {
    'bert': 'bert',
    'llama2': 'llama2',
    'gpt2': 'gpt',
    'baichuan': 'baichuan',
    'qwen': 'qwen',
    'falcon': 'Chinese-Falcon-7B'
}

class LLM:
    def __init__(self, model_name,load_model=True):
        ''' 支持 bert，llama2
        还可以修改之后支持 gpt2,baichuan,qwen,chatglm3,glm4,baichuan2 等
        '''
        self.model_name = model_name
        self.model_dir = ROOTPATH_Models + PATH_Models[model_name]
        self.repeat = None
        self.control_num = None
        if load_model:
            self.load_model()
            print(f'{model_name} 调用成功!')

    def load_model(self):
        

        self.control_num = 1024

        if self.model_name == 'bert':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained(self.model_dir)
            self.model = transformers.AutoModel.from_pretrained(self.model_dir).to(self.device)
            self.layers = 12
            self.control_num = 512

        elif self.model_name == 'gpt2':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained(ROOTPATH_Models + PATH_Models['bert'])
            self.model = transformers.AutoModel.from_pretrained(self.model_dir).to(self.device)
            self.layers = 12
            self.control_num = 512

        elif self.model_name == 'llama2':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_dir,use_fast=False)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_dir,device_map="auto",load_in_8bit=True)
            self.layers = 32


        elif self.model_name == 'baichuan':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_dir,use_fast=True,trust_remote_code=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_dir,device_map="auto",quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),trust_remote_code=True)
            self.layers = 32

        elif self.model_name == 'falcon':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_dir,use_fast = True,trust_remote_code='True') 
            self.model = transformers.FalconForCausalLM.from_pretrained(self.model_dir, trust_remote_code=True,load_in_8bit=True)
            self.layers = 32

        elif self.model_name == 'qwen':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_dir,use_fast=True,trust_remote_code=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_dir,device_map="auto",use_cache_quantization=True,quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),trust_remote_code=True)
            self.layers = 32
        self.device = self.model.device

    def tokenize(self,sentence):
        
        input_ids = self.tokenizer.encode(sentence, 
                                          return_tensors='pt', 
                                          add_special_tokens=True, 
                                          return_attention_mask=False, 
                                         )  
        return input_ids.to(self.model.device)


    def cut_sentence(self,sentence):
        input_ids = self.tokenize(sentence)
        tokens = [self.tokenizer.decode(i) for i in input_ids[0]] 
        return '/'.join(tokens)

    def get_word_embedding_echo(self,word, sentence, repeat=1):
        self.repeat = repeat
        if repeat > 1:
            sentence = sentence * repeat 

        input_ids = self.tokenize(sentence)
        if len(input_ids[0]) > 512 and 'gpt' in self.model_name:
            input_ids = input_ids[:,:512]

        tokens = [self.tokenizer.decode(i) for i in input_ids[0]] 

        indexes = self.get_index_of_word_in_tokens(word,tokens) 
        if len(indexes)==0:
            print('没有找到词:', word, sentence)
            indexes = [[0]]
        with torch.no_grad():
            out = self.model(input_ids, 
                             output_hidden_states=True,
                             output_attentions=False
                            )
            hidden_states = out.hidden_states
        return [torch.mean(hidden_state[0][indexes[-1]],0).cpu().detach().to(torch.float) for hidden_state in hidden_states] 


    def get_word_embedding(self, word, sentence,full=False,shift=False,repeat=1,attention=False): 
        
        self.repeat = repeat
        if repeat > 1:
            sentence = sentence * repeat 

        input_ids = self.tokenize(sentence)
        if len(input_ids[0]) > 512 and 'gpt' in self.model_name:
            input_ids = input_ids[:,:512]

        tokens = [self.tokenizer.decode(i) for i in input_ids[0]] 

        indexes = self.get_index_of_word_in_tokens(word,tokens) 
        if len(indexes)==0:
            print('没有找到词:', word, sentence)
            indexes = [[0]]
        with torch.no_grad():
            out = self.model(input_ids, 
                             output_hidden_states=True,
                             output_attentions=attention
                            )
            hidden_states = out.hidden_states
        if full:
            return tokens,hidden_states,indexes
        elif attention:
            attention_matrix = self.get_attention_matrix(out)
            self.attention_matrix = attention_matrix 
            return [self.merge_token_embedding_by_attention(h,indexes,attention_matrix) for h in hidden_states] 
        else:
            return [self.merge_token_embedding(h,indexes,shift) for h in hidden_states] 

    def get_index_of_word_in_tokens(self,word,tokens):
        '''找出哪些 token 涉及到目标词 word
        解决找不到 emo 的问题：先将 emo 单独做 tokenize 一下'''
        return get_index_of_word_in_tokens(word,tokens)

    def get_attention_matrix(self,outputs):
        A = 0
        for layer in range(len(outputs.attentions)):
            X = outputs.attentions[layer][0].cpu().detach().numpy()
            for i in range(len(X)):
                Ai = (X[i] + X[i].T)/2
                A = (A+Ai)/2 + abs(A-Ai)/2 
        return A

    def merge_token_embedding_by_attention(self,hidden_state,indexes,attention_matrix):
        indexes = indexes[0] 
        em = hidden_state[0][:].cpu().detach().to(torch.float)
        weights = sum([attention_matrix[index] for index in indexes])
        weights = weights / sum(weights)
        return np.average(em, axis=0, weights=weights)


    def merge_token_embedding(self,hidden_state,indexes,shift=False):
        
        indexes = indexes[0] 
        embedding = torch.mean(hidden_state[0][indexes],0).cpu().detach().to(torch.float)
        if shift:
            embedding_shift = torch.mean(hidden_state[0][:],0).cpu().detach().to(torch.float)
            return embedding_shift
        else:
            return embedding

    def get_sentence_embedding(self,sentence,repeat=1, last_token=True):
        self.repeat = repeat
        if repeat > 1:
            sentence = sentence * repeat 
        input_ids = self.tokenize(sentence)
        last_index = len(input_ids)//repeat
        with torch.no_grad():
            out = self.model(input_ids, 
                             output_hidden_states=True,
                             output_attentions=True
                            )
            hidden_states = out.hidden_states[-1]

        if last_token:

            attention_matrix = self.get_attention_matrix(out)
            self.attention_matrix = attention_matrix
            em = hidden_states[0][:].cpu().detach().to(torch.float) 
            weights = attention_matrix[-1]
            weights = weights / sum(weights)

            return np.average(em, axis=0, weights=weights)

        else:
            return hidden_states.mean(dim=1) 

    def demo(self):
        word = '世界'
        sentence = '我的世界是什么样的'
        result1 = self.get_sentence_embedding(sentence)
        print('句向量计算成功，可以正常使用',result1[0:5],len(result1),result1.shape)

class Solver:
    def __init__(self):
        self.name = 'Solution'
    def embedding_model(self,word, sentence):
        return None
    def predict_from(self,embeddings):
        return None


class Solver_Baseline(Solver):

    def __init__(self,llm=None,layer=-1):

        self.llm = llm
        self.layer = layer
        self.name = llm.model_name + '_baseline'

    def embedding_model(self,word, sentence):
        embedding_org = self.llm.get_word_embedding(word, sentence) 
        return embedding_org


    def predict_from(self,embeddings):
        

        n = len(embeddings)
        pred_list = [self._predict_from(self._embeddings_to_embedding(embeddings,i)) for i in range(n)]
        all_method = pred_list[0].keys()
        prediction = {}
        for method in all_method:
            prediction[method] = [chr(65 + pred[method]) for pred in pred_list] 
        return prediction

    def _embeddings_to_embedding(self,embeddings,item):
        embedding = embeddings[item] 
        embedding = torch.from_numpy(np.array(embedding)) 
        embedding = embedding[:,self.layer,:] 
        return embedding


    def _predict_from(self,embedding):
        embedding_shift = embedding - torch.mean(embedding,dim=0)

        dist = torch.norm(embedding,dim=1)
        dist_shift = torch.norm(embedding_shift,dim=1)
        distM = torch.mean(distance_matrix(embedding),dim=0)
        distM_shift = torch.mean(distance_matrix(embedding_shift),dim=0)
        sim = torch.mean(similarity_matrix_cos(embedding),dim=0)
        sim_shift = torch.mean(similarity_matrix_cos(embedding_shift),dim=0)

        pred = {

            'distance_shift': torch.argmax(dist_shift).item(),



            'cos_similarity_shift': torch.argmin(sim_shift).item()
        }
        return pred

class Solver_Echo(Solver_Baseline):

    def __init__(self,llm=None,layer=-1,shift=True,repeat=1):

        self.llm = llm
        self.layer = layer
        self.shift = shift
        self.repeat = repeat
        self.name = llm.model_name + '_echo' + 'v'*repeat

    def embedding_model(self,word, sentence): 
        embedding_org = self.llm.get_word_embedding_echo(word, sentence,
                                                    repeat=self.repeat)
        return embedding_org

class Solver_Ours(Solver_Baseline):

    def __init__(self,llm=None,layer=-1,shift=True,repeat=1,attention=False):

        self.llm = llm
        self.layer = layer
        self.shift = shift
        self.repeat = repeat
        self.attention = attention
        self.name = llm.model_name + '_ours_' + 'v'*repeat

    def embedding_model(self,word, sentence): 

        embedding_org = self.llm.get_word_embedding(word, sentence,
                                                    shift=self.shift,
                                                    repeat=self.repeat,
                                                    attention=self.attention)
        return embedding_org




class Evaluation():
    def __init__(self,problem,solver,later=False,use_temp=False):

        self.problem = problem
        self.solver = solver
        self.use_temp = use_temp
        self.embeddings = self.get_embeddings(problem, solver) 
        if not later:
            self.cal_accuracy()

    def get_embeddings(self,problem, solver):
        if not os.path.exists('Temp'):
            os.makedirs('Temp')

        embeddings = None

        tmp_filename = 'Temp/temp_embeddings_%s_%s.dat' %(problem.name, solver.name)
        if self.use_temp:
            try:
                embeddings = data_load(tmp_filename)
                print('读取文件：', tmp_filename)
            except:
                embeddings = None

        if embeddings is None:

            print('利用',solver.name,'计算目标词的向量嵌入：')
            embeddings = [[solver.embedding_model(w,s) for s in q] for w,q in tzip(problem.words, problem.question)] 
            data_save(embeddings,tmp_filename)
            print('保存文件：', tmp_filename)

        return embeddings

    def cal_accuracy(self):
        prediction = self.solver.predict_from(self.embeddings)

        accuracy = {}
        for method in prediction.keys():
            pred = prediction[method]
            accuracy[method] =  sum([ _ans == _pred  for _ans, _pred in zip(self.problem.answer,pred)]) / len(pred) 

        self.prediction = prediction
        self.accuracy = accuracy


    def save(self,df=None,filename='prediction_tmp.xlsx'):
        

        if df is None: df = self.problem.df
        for key in self.prediction.keys():
            df[key] = self.prediction[key] 

        df2 = pd.DataFrame.from_dict(self.accuracy,orient='index',columns=[self.solver.name])

        if filename is not None:
            xlsx = pd.ExcelWriter(filename)
            df.to_excel(xlsx,index=False,sheet_name='prediction')
            df2.to_excel(xlsx,index=True,sheet_name='accuaracy')
            xlsx.close()

        return df,df2


if __name__ == '__main__':

    llm = LLM('gpt2')
    llm.demo()
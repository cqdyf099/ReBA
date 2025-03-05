English 

This folder contains a Python file: **download.py**

### **Purpose**

Download pre-trained large models.

### **Usage**

The models mentioned in this document include: **bert-base-chinese, gpt2-base-chinese, Chinese-LLaMA-2-7B, Qwen-7B, Baichuan-7B, Chinese-Falcon-7B**.

To download these models, run the following command in the terminal:

```bash
python download.py <url> <file_name>
```

| Model              | URL                                                  | file_name         |
| ------------------ | ---------------------------------------------------- | ----------------- |
| bert-base-chinese  | https://huggingface.co/google-bert/bert-base-chinese | bert              |
| gpt2-base-chinese  | https://huggingface.co/ckiplab/gpt2-base-chinese     | gpt               |
| Chinese-LLaMA-2-7B | https://huggingface.co/LinkSoul/Chinese-Llama-2-7b   | llama2            |
| Qwen-7B            | https://huggingface.co/Qwen/Qwen-7B                  | qwen              |
| Baichuan-7B        | https://huggingface.co/baichuan-inc/Baichuan-7B      | baichuan          |
| Chinese-Falcon-7B  | https://huggingface.co/Linly-AI/Chinese-Falcon-7B    | Chinese-Falcon-7B |

For example, to download the Baichuan model, run the following command in the terminal:

```bash
python download.py https://huggingface.co/baichuan-inc/Baichuan-7B baichuan
```

----

中文版

本文件夹下有一个python文件：download.py

**用途**

下载预训练大模型

用法：

文中涉及到的模型有: bert-base-chinese, gpt2-base-chinese, Chinese-LLaMA-2-7B, Qwen-7B, Baichuan-7B, Chinese-Falcon-7B，为了下载这些模型，请依次在终端运行如下格式的代码：

python download.py \<url\> \<file_name\>

| Model              | url                                                  | file_name         |
| ------------------ | ---------------------------------------------------- | ----------------- |
| bert-base-chinese  | https://huggingface.co/google-bert/bert-base-chinese | bert              |
| gpt2-base-chinese  | https://huggingface.co/ckiplab/gpt2-base-chinese     | gpt               |
| Chinese-LLaMA-2-7B | https://huggingface.co/LinkSoul/Chinese-Llama-2-7b   | llama2            |
| Qwen-7B            | https://huggingface.co/Qwen/Qwen-7B                  | qwen              |
| Baichuan-7B        | https://huggingface.co/baichuan-inc/Baichuan-7B      | baichuan          |
| Chinese-Falcon-7B  | https://huggingface.co/Linly-AI/Chinese-Falcon-7B    | Chinese-Falcon-7B |

举例，如果要下载百川大模型，可以终端运行如下代码：

python download.py https://huggingface.co/baichuan-inc/Baichuan-7B baichuan


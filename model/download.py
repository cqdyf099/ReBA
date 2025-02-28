## 用于批量下载hf-mirror.com网站上的大模型
## 例: python download.py https://hf-mirror.com/baichuan-inc/Baichuan-7B/tree/main baichuan

import os
import sys
import requests
from bs4 import BeautifulSoup
import urllib.parse

# 检查是否输入了正确的参数
if len(sys.argv) != 3:
    print("用法: python download.py <url> <download_path>")
    sys.exit(1)

# 从命令行获取 URL 和下载文件夹路径
url = sys.argv[1]
download_path = sys.argv[2]

file_name1 = "url.txt"  # 文件名
os.makedirs(download_path, exist_ok=True)
with open(os.path.join(download_path, file_name1), "w") as file:
    file.write(url)
print(f"URL 已成功保存到 {file_name1}")

# 检查下载路径是否存在，如果不存在则创建文件夹
if not os.path.exists(download_path):
    os.makedirs(download_path)

# 发送请求获取页面内容
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 查找所有链接
links = soup.find_all('a')

# 过滤出包含 'download=True' 的链接(即模型文件)
download_links = []
for link in links:
    href = link.get('href')
    if href and 'download=true' in href:
        # 将相对链接转换为绝对链接
        download_links.append(f"https://hf-mirror.com{href}")

# 检查是否获取到链接
if not download_links:
    print("未找到任何包含 'download=true' 的链接。")
else:
    # 使用 wget 下载每个链接的文件到指定文件夹
    for download_link in download_links:
        print(f"正在下载: {download_link}")
        # 获取文件名部分
        parsed_url = urllib.parse.urlparse(download_link)
        file_name = os.path.basename(parsed_url.path)
        
        

        # 查看文件是否已存在
        old_file_path = os.path.join(download_path, file_name)
        # print(old_file_path)
        mid_file_path = old_file_path + "?download=true"
        
        if os.path.exists(mid_file_path) or os.path.exists(old_file_path):
            print(f"文件已存在，跳过下载: {old_file_path}")
        else:
            # 下载文件
            os.system(f"wget -P {download_path} {download_link}")

        # 重命名文件
        old_file_path = old_file_path + "?download=true"
        new_file_name = file_name.replace("?download=true", "")
        new_file_path = os.path.join(download_path, new_file_name)
        
        # 重命名文件
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
            print(f"文件已重命名为: {new_file_name}")
    
    print("所有文件下载并重命名完成！")

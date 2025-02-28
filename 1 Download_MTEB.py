from mteb import MTEB
import os
import subprocess

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
data_path = './C_MTEB_data'

def show_dataset():
    evaluation = MTEB(task_langs=["zh", "zh-CN"])
    dataset_list = []
    for task in evaluation.tasks:
        if task.description.get('name') not in dataset_list:
            dataset_list.append(task.description.get('name'))
            desc = 'name: {}\t\thf_name: {}\t\ttype: {}\t\tcategory: {}'.format(
                task.description.get('name'), task.description.get('hf_hub_name'),
                task.description.get('type'), task.description.get('category'),
            )
            print(desc)
    print(len(dataset_list))

def download_dataset():
    evaluation = MTEB(task_langs=["zh", "zh-CN"])
    err_list = []

    for task in evaluation.tasks:
        task_name = task.description.get('hf_hub_name')
        task_paths = [task_name, f"{task_name}-qrels"]

        for path in task_paths:
            print('Downloading:', path)
            cmd = [
                'huggingface-cli', 'download', '--repo-type', 'dataset', '--resume-download',
                '--local-dir-use-symlinks', 'False', path, '--local-dir', os.path.join(data_path, path)
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Downloaded: {path}")
            except subprocess.CalledProcessError as e:
                err_list.append(path)
                print(f"Error downloading {path}")

    if err_list:
        print('Download failed for the following datasets: \n', '\n'.join(err_list))
    else:
        print('All datasets downloaded successfully.')

if __name__ == '__main__':
    download_dataset()
    show_dataset()

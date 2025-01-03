import os

from datasets import load_dataset


def get_mnbvc_dataset(name: str, path: str):
    if not os.path.isdir(f"{path}/{name}"):
        os.makedirs(f"{path}/{name}")
    dataset_train = load_dataset("liwu/MNBVC", name=name, split='train', streaming=True)
    # dataset_test = load_dataset("liwu/MNBVC", name=name, split='test', streaming=True)
    cnt = 0
    file_cnt = 0
    content = []
    for data in dataset_train:
        if cnt % 1000 == 0:
            print(f"{cnt} lines have been processed in file {name}")
        if name in ["law_judgement"]:
            content.append(data["text"])
        elif name in ["news_peoples_daily", "wikipedia"]:
            content.append(parse_data(data))
        cnt += 1
        if cnt % 100000 == 0:
            with open(f"{path}/{name}/{name}_train_{file_cnt}.txt", "w") as f:
                f.write(' '.join(content))
            f.close()
            file_cnt += 1
            content = []


def parse_data(data: dict) -> str:
    res = []
    for paragraph in data["段落"]:
        res.append(paragraph["内容"])
    return " ".join(res)


# names = ["law_judgement", "gov_xuexiqiangguo", "gov_report", "co_ann_report", "news_peoples_daily", "wikipedia"]
names = ["wikipedia"]
for name in names:
    get_mnbvc_dataset(name=name, path="./data")
    print(f"{name} dataset has been downloaded")

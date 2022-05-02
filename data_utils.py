# Survey News https://huggingface.co/datasets/Fraser/news-category-dataset keep top5 most popular
# Survey DBPEDIA https://huggingface.co/datasets/dbpedia_14 first 4 labels

# Contra TREC https://huggingface.co/datasets/trec use coarse labels
# Contra 20NG https://huggingface.co/datasets/newsgroup use contra ood

from datasets import load_dataset
import torch
import constants
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler, random_split)

DBPEDIA = 'dbpedia_14'
DBPEDIA_TOP_K = 4
DBPEDIA_SPLIT = 0.5

NEWS = 'Fraser/news-category-dataset'
NEWS_TOP_K = 5

TREC = 'trec'
NG = 'newsgroup'

def get_dbpedia():
    hf_dataset = load_dataset(DBPEDIA,  download_mode="force_redownload")
    label_strings = list(hf_dataset['train'].features['label'].names)
    
    train, test = hf_dataset['train'], hf_dataset['test']

    return train, test, label_strings

def get_news():
    # The labels are sorted in decreasing popularity
    hf_dataset = load_dataset(NEWS)
    label_strings = list(hf_dataset['train'].features['category_num'].names)
    train, test, val = hf_dataset['train'], hf_dataset['test'], hf_dataset['validation']
    return train, val, test, label_strings

def get_dataloader(x, labels, tokenizer):

    torch.manual_seed(0)

    tokenized = tokenizer(x, padding=constants.PADDING, max_length=constants.MAX_LENGTH, truncation = constants.TRUNCATION)
    x_ids = torch.tensor([f for f in tokenized.input_ids], dtype=torch.long)
    x_mask = torch.tensor([f for f in tokenized.attention_mask], dtype=torch.long)
    label_tensor = torch.tensor(labels)
    tensor_dataset = TensorDataset(x_ids, x_mask, label_tensor)
    loader = DataLoader(tensor_dataset, batch_size = constants.BATCH_SIZE)
    return loader

def get_data(dataset_name, tokenizer):

    torch.manual_seed(0)

    if dataset_name == DBPEDIA:
        train, test, label_strings = get_dbpedia()
        train = train.sort('label')
        test = test.sort('label')

        train_filtered = train.filter(lambda example : example['label'] < DBPEDIA_TOP_K)
        train_filtered = train_filtered.shuffle(seed=0)

        # test_filtered = test.filter(lambda example : example['label'] >= DBPEDIA_TOP_K)
        test_split = test.train_test_split(test_size = 0.5) # shuffles for you, no need to shuffle again
        val_final = test_split['train'].shuffle(seed=0)
        test_final = test_split['test'].shuffle(seed=0)

        train_loader = get_dataloader(train_filtered['content'], train_filtered['label'], tokenizer)
        val_loader = get_dataloader(val_final['content'], val_final['label'], tokenizer)
        test_loader = get_dataloader(test_final['content'], test_final['label'], tokenizer)
        label_indices = torch.tensor(tokenizer(label_strings[:DBPEDIA_TOP_K], padding='longest')['input_ids'])
        return train_loader, val_loader, test_loader, label_indices, DBPEDIA_TOP_K

    elif dataset_name == NEWS:
        train, val, test, label_strings = get_news()
        train = train.sort('category_num')
        val = val.sort('category_num')
        test = test.sort('category_num')

        train_filtered = train.filter(lambda example : example['category_num'] < NEWS_TOP_K)
        train_content = [x['headline'] + ' ' + x['short_description'] for x in train_filtered]
        train_filtered = train_filtered.add_column('content', train_content)
        train = train_filtered.shuffle(seed=0)
        print(train)

        # val_filtered = val.filter(lambda example : example['category_num'] >= NEWS_TOP_K)
        val_content = [x['headline'] + ' ' + x['short_description'] for x in val]
        val = val.add_column('content', val_content)
        val = val.shuffle(seed=0)

        # test_filtered = test.filter(lambda example : example['category_num'] >= NEWS_TOP_K)
        test_content = [x['headline'] + ' ' + x['short_description'] for x in test]
        test = test.add_column('content', test_content)
        test = test.shuffle(seed=0)

        train_loader = get_dataloader(train['content'], train['category_num'], tokenizer)
        val_loader = get_dataloader(val['content'], val['category_num'], tokenizer)
        test_loader = get_dataloader(test['content'], test['category_num'], tokenizer)
        label_indices = torch.tensor(tokenizer(label_strings[:NEWS_TOP_K], padding='longest')['input_ids'])
        return train_loader, val_loader, test_loader, label_indices, NEWS_TOP_K
    else:
        raise ValueError("Unsupported Dataset")

    # return train, test

    







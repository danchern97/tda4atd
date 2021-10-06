import os
import re
import numpy as np

from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertModel

def grab_attention_weights(model, tokenizer, sentences, MAX_LEN, device='cuda:0'):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                       return_tensors='pt',
                                       add_special_tokens=True,
                                       max_length=MAX_LEN,             # Max length to truncate/pad
                                       pad_to_max_length=True,         # Pad sentence to max length)
                                       truncation=True
                                      )
    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    attention = model(input_ids, attention_mask, token_type_ids)['attentions']
    # layer X sample X head X n_token X n_token
    attention = np.asarray([layer.cpu().detach().numpy() for layer in attention], dtype=np.float16)
    
    return attention

def grab_weights_for_all(reviews,
                         model_name='bert-base-multilingual-cased',
                         layer_of_interest=-1,
                         head_of_interest=0,
                         recalculate=True,
                         output_file='adj_matricies.npy'
                         ):
    """
    Returns attention weights (matricies) for each sentence from reviews, for
    chosen layer and head. If recalculate==False, loads such weights from .npy
    file. Otherwise, calculates them and saves into .npy file.

    Args:
        reviews (list[str])
        model_name (str)
        layer_of_interest (int)
        head_of_interest (int),
        recalculate (bool),
        output_file (str).

    Returns:
        np.array[int,int,int]
    """
    
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    

    r_file = Path(output_file)

    adj_matricies = []

    if r_file.is_file() and not recalculate:
        adj_matricies = np.load(r_file, allow_pickle=True)
        print("Загружены ранее вычисленные веса.")

    else:
        for i in tqdm(range(len(reviews)),
                            desc="Вычисление весов"):
            attention_w = grab_attention_weights(model, tokenizer, reviews[i])
            adj_matrix  = attention_w[layer_of_interest].detach().numpy()[0][head_of_interest]
            adj_matricies.append(adj_matrix)

        adj_matricies = np.asarray(adj_matricies)
        np.save(r_file, adj_matricies)

        print("Результаты вычисления сохранены в файл", r_file, ".")
        
    return adj_matricies
  
def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

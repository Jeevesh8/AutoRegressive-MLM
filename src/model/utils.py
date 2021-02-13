import haiku as hk
from haiku.data_structures import to_immutable_dict, to_mutable_dict
import re, jax
from typing import List, Tuple

def print_keys(params, n=0):
    for key in params:
        print('\t'*n+key)
        try :
            print_keys(params[key], n+1)
        except:
            pass

def change_keys(dic, key, replace_with):
    dic1 = {}
    for k in dic.keys():
        dic1[k.replace(key, replace_with, 1)] = dic[k]
    return dic1

def copy_available_keys(dic1, dic2, special_pairs: List[Tuple[str, Tuple[str,str]]]=None):
    """
    Updates the value of any key of dic2 to its corresponding value in dic1, 
    if the key exists in dic1. Else dic2[key] remains unchanged.
    
    special_pairs: Are the pairs of (dic1_key, dic2_key) where the values 
    have to be copied b/w those keys
    
    Returns : updated dic2
    """
    special_keys = [elem2[0] for (elem1, elem2) in special_pairs]
    for k in dic2.keys():
        if k in dic1.keys():
            print("Loading Pre-Trained weights for :", k)
            dic2[k] = dic1[k]
        else:
            print("Can't find weights for  : ", k, " in pretrained wts provided(dic1).") 
    
    if special_pairs is not None:
        
        for (dic1_key, dic2_key) in special_pairs:
            print("Loading Pre-Trained weights for : ", dic2_key, " to ", dic1_key)
            if len(dic2_key)==1:
                dic2[dic2_key[0]] = dic1[dic1_key]
            elif len(dic2_key)==2:
                dic2[dic2_key[0]][dic2_key[1]] = dic1[dic1_key]

    return dic2

def logits_to_ar_classifier_params(pretrained_params, classifier_params):
    """
    This function adds extra parameters to pretrained ones, for the downstream task.
    pretrained_params:  are the params obtained from AutoRegressive-MLM pre-training.
    classifier_params:  params obtained from initializing the model for downstream task.
    """    
    pretrained_params = to_mutable_dict(pretrained_params)
    pretrained_params['ar_classifier'] = pretrained_params['mlm_predictor']
    pretrained_params.pop('mlm_predictor')
    pretrained_params['ar_classifier'] = change_keys(pretrained_params['ar_classifier'], 'extended_encoder', 'auto_regressive_classifier')
    pretrained_params['ar_classifier'] = change_keys(pretrained_params['ar_classifier'], 'auto_regressive_classifier/~/', 'auto_regressive_classifier/')
    #pretrained_params['ar_classifier']['extended_encoder/linear'] = classifier_params['ar_classifier']['auto_regressive_classifier/linear']
    pretrained_params['ar_classifier']['auto_regressive_classifier/~/gru/~/gru'] = classifier_params['ar_classifier']['auto_regressive_classifier/~/gru/~/gru']
    return to_immutable_dict(pretrained_params)

#############################################################################################
#                        HuggingFace Specific Weight Loading Code                           #
#############################################################################################
from io import BytesIO
from functools import lru_cache

import joblib
import requests

def postprocess_key(key):
    """
    Changing keys of RoBERTa huggingface model to match our model.
    """
    key = key.replace('self.', '')
    key = re.sub(r'layer\.(\d+)\.', r'layer_\1.', key)
    key = key.replace('output.dense', 'output_dense')
    key = key.replace('attention.output.LayerNorm', 'attention_output_LayerNorm')
    key = key.replace('output.LayerNorm', 'output_LayerNorm')
    key = key.replace('intermediate.dense', 'transformer_mlp.intermediate_dense')
    key = re.sub(r'layer_(\d+)\.output_dense', r'layer_\1.transformer_mlp.output_dense', key)
    key = key.replace('.', '/')
    key = key.replace('LayerNorm/weight', 'LayerNorm/scale')
    key = key.replace('LayerNorm/bias', 'LayerNorm/offset')
    return key

def change_wts_structure(pt_wts, dont_touch=[]):
    """
    Utiility for converting the structure of pre-trained weights of 
    HuggingFace RoBERTa into that required by our model.

    dont_touch: These elements are left unchanged.
    """
    new_wts_struct = {} 
    
    for k in pt_wts.keys():
        
        if k in dont_touch:
            new_wts_struct[k] = pt_wts[k]
            continue
        
        nested_modules = k.split('/')
        second_last_module = '/'.join(nested_modules[:-1]) 
        last_module = nested_modules[-1]

        if last_module=='weight':
            last_module = 'w'
        elif last_module =='bias':
             last_module = 'b'

        if second_last_module not in new_wts_struct:
            new_wts_struct[second_last_module] = {} 
        
        new_wts_struct[second_last_module][last_module] = pt_wts[k]

    return new_wts_struct

def add_extra_word_embeddings(w, config):
    """
    Adds word embeddings for extra tokens, than were in the model 
    from which pre-trained weights are loaded.
    """
    stddev = 1. / np.sqrt(config['d_model'])
        
    n_extra = len(self.config['extra_tokens'])
    
    key, subkey = jax.random.split( jax.random.PRNGKey(22) )
    extra_w = stddev*jax.random.truncated_normal(subkey, -2., 2., 
                                                 shape=[n_extra, self.config['d_model']])        
    return jax.numpy.concatenate([w, extra_w], axis=0)
    
@lru_cache()
def get_pretrained_weights(config):
    # We'll use the weight dictionary from the RoBERTa encoder at HuggingFace
    from transformers import RobertaModel
    huggingface_model = RobertaModel.from_pretrained(config['initialize_pretrained'], output_hidden_states=True)

    weights = huggingface_model.state_dict()

    weights = {
        postprocess_key(key): value.numpy()
        for key, value in weights.items()
    }
    
    weights['embeddings/word_embeddings/weight'] = add_extra_word_embeddings(weights['embeddings/word_wmbeddings/weight'],
                                                                      config)
    
    weights = change_wts_structure(weights, dont_touch=['embeddings/word_embeddings/weight', 
                                                        'embeddings/position_embeddings/weight'])

    return weights
############################################################################################





##########################  Probably useless utilies  #####################################

class Scope(object):
    """
    A tiny utility to help make looking up into our dictionary cleaner.
    There's no haiku magic here.
    """
    def __init__(self, weights, prefix):
        self.weights = weights
        self.prefix = prefix

    def __getitem__(self, key):
        return hk.initializers.Constant(self.weights[self.prefix + key]) if self.weights is not None else None

## For Fine-Tuning on downstream task of Argument Classification

import copy
import numpy as np
from functools import partial
from copy import deepcopy
import pickle

import jax
import jax.numpy as jnp
import haiku as hk
from haiku.data_structures import to_immutable_dict, to_mutable_dict
import optax
from transformers import RobertaTokenizer

from src.DataLoaders.xml import load_xml_data
from src.DataLoaders.json import load_reddit_data
from src.Tokenizers.thread_tokenizer import Thread_Tokenizer
from src.model.transformers import TransformerFeaturizer, FineTuningExtendedEncoder
from src.model.utils import logits_to_ar_classifier_params, print_keys, get_pretrained_weights, copy_available_keys
from src.model.purified_jitted_fns import get_fn_to_transform, get_pure_jitted_fn
from src.optimizers.adam import get_adam_opt
from src.Tokenizers.masking_utils import get_masking_func

from config import config
from loss_eval_utils import ft_loss, get_params, get_classification_report, flatten_dict

import wandb

def load_pretrained_tokenizer():
    """Loads Pre-Trained Tokenizers if config['initialize_pretrained'] is specified, into the global config"""

    if 'initialize_pretrained' in config and config['initialize_pretrained']!='':

        huggingface_tokenizer = RobertaTokenizer.from_pretrained(config['initialize_pretrained'])

        config['pt_hf_tokenizer'] = huggingface_tokenizer

def get_dataloaders():
    data_loader = load_reddit_data(config)

    train_data_loader = load_xml_data(config, split='train/')

    valid_data_loader = load_xml_data(config, split='valid/')

    test_data_loader = load_xml_data(config, split='test/')
    
    return data_loader, train_data_loader, valid_data_loader, test_data_loader

def train_tokenizer(data_loader):
    if config['initialize_pretrained'] == '':

        lm_tokeniser = Thread_Tokenizer(config)
        lm_tokeniser.train_tokenizer(str_iter=data_loader.get_sentences())

    else: 
        #Will automatically load pre-trained version if config['pt_hf_tokenizer'] is defined.
        lm_tokeniser = Thread_Tokenizer(config)
    return lm_tokeniser

def update_config(config, train_data_loader):
    print("Vocabulary : ", lm_tokeniser.tokenizer.get_vocab())

    config['vocab_size'] = lm_tokeniser.tokenizer.get_vocab_size()

    #Tokenization ids  
    config['mask_id'] = lm_tokeniser.tokenizer.token_to_id("<mask>")
    config['pad_id'] = lm_tokeniser.tokenizer.token_to_id("<pad>")
    config['sos_id'] = lm_tokeniser.tokenizer.token_to_id("<s>")
    config['eos_id'] = lm_tokeniser.tokenizer.token_to_id("</s>")
    config['dsm_list'] = [lm_tokeniser.tokenizer.token_to_id(token)
                                for token in lm_tokeniser.dms]
    config['total_steps'] = len([0 for thread in train_data_loader.thread_generator()])
    print("Total steps: ", config['total_steps'])
    return config

def load_pretrained_wts(featurizer_params, ExtendedEncoder_params):
    """Merging pre-trained and initialised parameters"""
    
    if config['params_file']!='':
        
        with open(config['params_file'], 'rb') as f:
            pt_wts = pickle.load(f)

        featurizer_params = to_mutable_dict(featurizer_params)

        featurizer_params = copy_available_keys(pt_wts['comments_encoder'], featurizer_params,)
        
        ExtendedEncoder_params = to_mutable_dict(ExtendedEncoder_params)

        ExtendedEncoder_params = copy_available_keys(pt_wts['mlm_predictor'], ExtendedEncoder_params,)

    else:
        print("No pretrained wts file was provided, initializing with random wts. Provide the pt wts file\
               in config['param_file'], if you wish to use pretrained weights.")
    
    params = to_immutable_dict( {'comments_encoder' : featurizer_params, 
                                'ar_classifier' : ExtendedEncoder_params } )
    return params

def jit_fns(pure_featurizer_fn, pure_loss_fn, pure_pred_fn):
    
    global featurizer_f, loss_f, eval_featurizer_f, eval_pred_f, loss, accuracy 
    
    featurizer_f = get_pure_jitted_fn(pure_featurizer_fn, True, config)
    loss_f = get_pure_jitted_fn(pure_loss_fn, True, config)
    
    loss = partial(ft_loss, featurizer_f, loss_f, mode='loss')
    
    eval_featurizer_f = get_pure_jitted_fn(pure_featurizer_fn, False, config)
    eval_pred_f = get_pure_jitted_fn(pure_pred_fn, False, config)

    accuracy = partial(ft_loss, eval_featurizer_f, eval_pred_f, mode='accuracy')

def update(opt_state, params, key, thread, config):
    turn = 0
    (batch_loss, remaining_comments), grad = jax.value_and_grad(loss, has_aux=True)(params, key, thread, config, turn)
    turn += 1

    while remaining_comments:
        print("Big tree, turn: ", turn)
        tup, grads = jax.value_and_grad(loss, has_aux=True)(params, key, thread, config, turn)
        turn += 1
        batch_loss += tup[0]
        grad = jax.tree_util.tree_multimap(lambda x,y: x+y, grad, grads) 
        remaining_comments = tup[1]
    
    updates, opt_state = opt.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss

def thread_accuracy(params, key, thread, config):
    turn = 0
    all_preds, all_labels = [], []
    remaining_comments = True

    while remaining_comments:
        tup, remaining_comments = accuracy(params, key, thread, config, turn)
        all_preds += tup[0]
        all_labels += tup[1]
        turn += 1

    return all_preds, all_labels

def evaluate(config, params, data_loader, key):
    all_preds = []
    all_labels = []

    for step, thread in enumerate(data_loader.thread_generator()):
        if step%100==0:
            print(f'[Step {step}]')
        
        thread = lm_tokeniser.tokenize_thread(thread)

        key, subkey = jax.random.split(key)
        tup = thread_accuracy(params, subkey, thread, config)
        
        all_preds += tup[0]
        all_labels += tup[1]
    
    return all_preds, all_labels

def train(config, params, train_data_loader, key, opt_state):
    losses = []
    val_losses = []

    for _ in range(config['n_epochs']):

        for step, thread in enumerate(train_data_loader.thread_generator()):
                        
            if step%(config['total_steps'])==0:
                print(f'[Step {step}]')

            thread = lm_tokeniser.tokenize_thread(thread)

            key, subkey = jax.random.split(key)
            params, opt_state, batch_loss = update(opt_state, params, subkey,
                                                   thread, config)
            
            losses.append(batch_loss.item())

            if step%(config['total_steps'])==0:
                print(sum(losses)/len(losses))
                losses = []

            if step==config['total_steps']-1:
                all_preds, all_labels = evaluate(config, params, valid_data_loader, key)
                wandb.log(flatten_dict({'Validation' : get_classification_report(config, all_labels, all_preds)}))
    
                all_preds, all_labels = evaluate(config, params, test_data_loader, key)
                wandb.log(flatten_dict({'Test' : get_classification_report(config, all_labels, all_preds)}))
    
    return val_losses

    
if __name__=='__main__' :
    
    global lm_tokenizer, featurizer_f, loss_f, mask_batch_mlm, eval_featurizer_f, eval_pred_f, loss, accuracy, opt

    load_pretrained_tokenizer()

    data_loader, train_data_loader, valid_data_loader, test_data_loader = get_dataloaders()
    
    lm_tokeniser = train_tokenizer(data_loader)

    config = update_config(config, train_data_loader)
    
    wandb.init(project='autoregressive-mlm-ft', config=config)

    config = hk.data_structures.to_immutable_dict(config)

    pure_featurizer_fn = hk.transform( get_fn_to_transform(TransformerFeaturizer) )
    pure_loss_fn = hk.transform( get_fn_to_transform(FineTuningExtendedEncoder) )
    pure_pred_fn = hk.transform( get_fn_to_transform(FineTuningExtendedEncoder, training=False) )

    key, subkey = jax.random.split( jax.random.PRNGKey(42) )
    featurizer_params, ExtendedEncoder_params = get_params(config, key, pure_loss_fn, pure_featurizer_fn)
    params = load_pretrained_wts(featurizer_params, ExtendedEncoder_params)

    mask_batch_mlm = get_masking_func(config)

    jit_fns(pure_featurizer_fn, pure_loss_fn, pure_pred_fn)
    
    lrs = [1e-3]
    drs = [0.1]
    valid_epoch_losses = []

    for lr in lrs:
        for dr in drs:

            config = hk.data_structures.to_mutable_dict(config)
            config['learning_rate'] = lr
            config['classifier_drop_rate']= dr
            config = hk.data_structures.to_immutable_dict(config)
            
            opt = get_adam_opt(config)
            opt_state = opt.init(params)
            
            jit_fns(pure_featurizer_fn, pure_loss_fn, pure_pred_fn)

            init_params = copy.deepcopy(params)

            val_losses = train(config, init_params, train_data_loader, key, opt_state)
            
            valid_epoch_losses.append( val_losses )
            
            wandb.log({'learning_rate':lr, 'dropout_rate': dr})
            print(f"Learning rate={lr}, Dropout Rate={dr} Losses : ", valid_epoch_losses[-1])
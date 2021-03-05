## For pre-training from RoBERTa initialised weights.
import copy, os
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

from src.DataLoaders.json import load_reddit_data
from src.Tokenizers.tree_tokenizer import Tree_Tokenizer
from src.model.transformer import TransformerFeaturizer, ExtendedEncoder
from src.model.utils import get_pretrained_weights, copy_available_keys
from src.model.purified_jitted_fns import get_fn_to_transform, get_pure_jitted_fn
from src.optimizers.adam import get_adam_opt
from src.Tokenizers.masking_utils import get_masking_func
from src.Tokenizers.utils import tree_to_batch, batch_to_tree, gather_batch_parents
from config import config
import wandb

from finetune import load_pretrained_tokenizer
from loss_eval_utils import cross_entropy


def get_dataloaders():
    data_loader = load_reddit_data(config)

    eval_data_loader = load_reddit_data(config, mode='eval')
    return data_loader, eval_data_loader

def get_tokenizer(data_loader):
    if config['initialize_pretrained'] == '':
        lm_tokeniser = Tree_Tokenizer(config)
        lm_tokeniser.train_tokenizer(str_iter=data_loader.get_sentences())
    else: 
        lm_tokeniser = Tree_Tokenizer(config)
    return lm_tokeniser

def update_config(config, data_loader):
    print("Vocabulary : ", lm_tokeniser.tokenizer.get_vocab())
    
    config['vocab_size'] = lm_tokeniser.tokenizer.get_vocab_size()
    config['mask_id'] = lm_tokeniser.tokenizer.token_to_id("<mask>")
    config['pad_id'] = lm_tokeniser.tokenizer.token_to_id("<pad>")
    config['sos_id'] = lm_tokeniser.tokenizer.token_to_id("<s>")
    config['eos_id'] = lm_tokeniser.tokenizer.token_to_id("</s>")
    config['dsm_list'] = [lm_tokeniser.tokenizer.token_to_id(token)
                                for token in lm_tokeniser.dms]
    config['total_steps'] = len([0 for tree in data_loader.tree_generator()])
    print("Total steps: ", config['total_steps'])
    return config

def load_pretrained_wts(featurizer_params, ExtendedEncoder_params):
    """Merging pre-trained and initialised parameters"""
    
    param_idx = config['restart_from']//config['total_steps']
    if os.path.isfile(config['params_dir']+f'params_{param_idx}'):
        with open(config['params_dir']+f'params_{param_idx}', 'rb') as f:
            params = pickle.load(f)
            return params
    
    if config['initialize_pretrained']!='':
        
        pt_wts = get_pretrained_weights(config)

        featurizer_params = to_mutable_dict(featurizer_params)

        featurizer_params = copy_available_keys(pt_wts, featurizer_params, 
                                            [('embeddings/word_embeddings/weight', ('encoder/embedding/~/embed', 'embeddings')), 
                                            ('embeddings/position_embeddings/weight', ('encoder/embedding/position_embeddings', 'position_embeddings')),
                                            ('embeddings/LayerNorm', ('encoder/embedding/layer_norm',))])
        
        ExtendedEncoder_params = to_mutable_dict(ExtendedEncoder_params)

        ExtendedEncoder_params = copy_available_keys(pt_wts, ExtendedEncoder_params, 
                                                [('embeddings/word_embeddings/weight', ('encoder/~/embedding/~/embed', 'embeddings')), 
                                                ('embeddings/position_embeddings/weight', ('encoder/~/embedding/position_embeddings', 'position_embeddings')),
                                                ('embeddings/LayerNorm', ('encoder/~/embedding/layer_norm',))])
    else:
        print("No pretrained MLM model (e.g. distilbert, roberta..) was specified, initializing with random wts. Provide a pretrained \
                model name in config['initialize_pretrained'], if you wish to use pretrained weights of that model.")
    
    params = to_immutable_dict( {'comments_encoder' : featurizer_params, 
                                'mlm_predictor' : ExtendedEncoder_params } )
    return params

def embed_batches(params, key, config, batches):
    
    encodings = []
    for batch in batches:
        key, subkey = jax.random.split(key)
        features = featurizer_f(params['comments_encoder'], subkey, 
                                batch[:, :config['featurizer_max_length']])
        encodings.append(features)
    return encodings

def compute_ar_loss(params, key, config, tree, batches, comment_batches, turn):
    loss = 0.0
    empty_elem = jnp.asarray([0]*config['d_model'], dtype=jnp.int16)    
    
    for i, (original_batch, comment_batch) in enumerate( zip(batches, comment_batches) ):
        
        if i<turn*config['max_losses']:
            continue

        if i==(turn+1)*config['max_losses']:
            return (loss, True)
     
        parent_comment_embds, mask_for_embds = gather_batch_parents(tree, comment_batch, 
                                                                    config['max_length'], key='comment_embds', 
                                                                    empty_elem=empty_elem)
        key, subkey = jax.random.split(key)
        masked_batch, original_batch = mask_batch_mlm(subkey, original_batch)

        key, subkey = jax.random.split(key)
        loss += loss_f(params['mlm_predictor'], subkey, parent_comment_embds, 
                             mask_for_embds, masked_batch, original_batch)

    return (loss, False)

def loss(params, key, init_tree, config, turn=0):
    """
    Calculates loss for all nodes of a single tree.
    The masked tokens of each location in a comment are predicted 
    conditioned on the embeddings of all the parent comments.
    """
    tree = deepcopy(init_tree)

    empty_elem = jnp.asarray([config['pad_id']]*config['max_length'], dtype=jnp.int16)
    batches = tree_to_batch(tree, config['featurizer_batch_size'],
                            key='tokenized_inputs', empty_elem=empty_elem)
    
    key, subkey = jax.random.split(key) 
    encodings = embed_batches(params, subkey, config, batches)
    
    tree = batch_to_tree(tree, encodings, config['featurizer_batch_size'], 
                         key='comment_embds')

    comment_batches = tree_to_batch(tree, config['mlm_batch_size'], key=None, 
                                    empty_elem={}, include_root=False)
    
    batches = tree_to_batch(tree, config['mlm_batch_size'],
                            key='tokenized_inputs', empty_elem=empty_elem,
                            include_root=False)
    
    key, subkey = jax.random.split(key)
    loss, remaining_comments = compute_ar_loss(params, subkey, config, tree, batches, comment_batches, turn)
    
    return loss, remaining_comments

def update(opt_state, params, key, tree, config):
    turn = 0
    key, subkey = jax.random.split(key)
    (batch_loss, remaining_comments), grad = jax.value_and_grad(loss, has_aux=True)(params, key, tree, config, turn)
    turn += 1

    while remaining_comments:
        print("Big tree, turn: ", turn)
        key, subkey = jax.random.split(key)
        tup, grads = jax.value_and_grad(loss, has_aux=True)(params, key, tree, config, turn)
        turn += 1
        batch_loss += tup[0]
        grad = jax.tree_util.tree_multimap(lambda x,y: x+y, grad, grads) 
        remaining_comments = tup[1]
    
    updates, opt_state = opt.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)    
    return new_params, opt_state, batch_loss


if __name__=='__main__':
    
    global featurizer_f, loss_f, mask_batch_mlm, opt

    data_loader, eval_data_loader = get_dataloaders()
    lm_tokeniser = get_tokenizer(data_loader)
    
    config = update_config(config)
    
    wandb.init(project='autoregressive-mlm', config=config)
    config = hk.data_structures.to_immutable_dict(config)

    pure_featurizer_fn = hk.transform( get_fn_to_transform(TransformerFeaturizer, config) )
    pure_loss_fn = hk.transform( get_fn_to_transform(ExtendedEncoder, config) )

    featurizer_f = get_pure_jitted_fn(pure_featurizer_fn, True, config)
    loss_f = get_pure_jitted_fn(pure_loss_fn, True, config)

    mask_batch_mlm = get_masking_func(config)

    key, subkey = jax.random.split( jax.random.PRNGKey(42) )
    params = get_params(config, key, pure_loss_fn, pure_featurizer_fn)

    opt = get_adam_opt(config)
    opt_state = opt.init(params)
    
    for _ in range(config['n_epochs']):
        
        losses = []
        for step, tree in enumerate(data_loader.tree_generator()):
            
            if _*config['total_steps']+step <= config['restart_from']:
            continue
            
            if step%100==0:
                print(f'[Step {step}]')

            tree = lm_tokeniser.tokenize_tree(tree)

            key, subkey = jax.random.split(key)
            params, opt_state, batch_loss = update(opt_state, params, subkey,
                                                tree, config)

            losses.append(batch_loss)

            if step%100==0 and step!=0:
                print(sum(losses)/100)
                wandb.log({'loss_on_100_batches':sum(losses).item()/100})
                losses = []

            if step%1000==0 and step!=0:
                with open(config['params_dir']+f'params{_}.pkl', 'wb+') as f:
                    pickle.dump(params, f)
                wandb.save(config['params_dir']+f'params{_}.pkl')
                print("Wrote params to disk")
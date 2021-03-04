from copy import deepcopy

from sklearn.metrics import classification_report
import numpy as np
import jax.numpy as jnp
import jax
import haiku as hk

def get_sample_inp(config):
    
    token_ids = np.random.randint(config['vocab_size'], size=(config['mlm_batch_size'], config['max_length']))
    comment_ids = np.random.randint(config['vocab_size'], size=(config['featurizer_batch_size'], config['max_length']))
    return token_ids, comment_ids

def get_params(config, key, pure_loss_fn, pure_featurizer_fn):
    
    token_ids, comment_ids = get_sample_inp(config)

    masked_token_ids = token_ids
    masked_token_ids[np.random.randint(config['mlm_batch_size'], size=3),np.random.randint(config['mlm_batch_size'], size=3)] = 0

    key, subkey = jax.random.split(key)
    featurizer_params = pure_featurizer_fn.init(subkey, comment_ids, True, config)
    comment_embds = pure_featurizer_fn.apply(featurizer_params, subkey, comment_ids, True, config)

    comment_embds = jnp.tile(comment_embds, config['max_length']).reshape(config['mlm_batch_size'], config['max_length'], -1)
    comment_mask = jnp.ones_like(comment_embds[:,:,0])
    
    key, subkey = jax.random.split(key)
    
    ExtendedEncoder_params = pure_loss_fn.init(subkey, comment_embds, 
                                               comment_mask, masked_token_ids, token_ids,
                                               True, config)
    
    return featurizer_params, ExtendedEncoder_params

def get_batched_version(elem_lis, batch_size, empty_elem):
    extra = len(elem_lis)%batch_size
    
    if extra!=0:
        elem_lis += [empty_elem]*(batch_size-extra)
    
    return [ jnp.stack(elem_lis[i*batch_size : (i+1)*batch_size]) 
             for i in range(len(elem_lis)//batch_size) ]

def remove_pad_preds(all_preds, all_labels):
    pad_removed_preds = []
    pad_removed_labels = []
    
    for i in range(len(all_preds)):
        
        if all_preds[i]!=-1:
            pad_removed_labels.append(all_labels[i])
            pad_removed_preds.append(all_preds[i])
    
    return pad_removed_preds, pad_removed_labels

def get_pred_list(config, labels, preds, batch):

    logits_mask = (batch!=config['pad_id'])
    preds = jnp.where(logits_mask, preds, -1)
    
    all_preds = preds.reshape(-1).tolist()
    all_labels = labels.reshape(-1).tolist()

    return remove_pad_preds(all_preds, all_labels)

def get_encoder_batches(config, thread):
    empty_elem = jnp.asarray([config['pad_id']]*config['max_length'], dtype=jnp.int16)
    return get_batched_version(thread, config['featurizer_batch_size'], empty_elem)

def get_decoder_batches(config, thread, encodings, labels):
    batches = get_encoder_batches(config, thread)
    
    parent_encodings = [jnp.stack( encodings[:i]+[ jnp.zeros_like(encodings[0]) ]*(config['max_length']-i) ) 
                        for i in range(min(len(encodings), config['max_length']))]
    
    parent_mask_lis = [jnp.asarray( [0]*i +[1]*(config['max_length']-i) , dtype=jnp.int16) 
                        for i in range(min(len(encodings), config['max_length']))]
    
    parent_encodings = get_batched_version(parent_encodings, config['featurizer_batch_size'], parent_encodings[0])
    parent_mask_lis = get_batched_version(parent_mask_lis, config['featurizer_batch_size'], parent_mask_lis[0])
    label_batches = get_batched_version(labels, config['featurizer_batch_size'], labels[0])
    
    return zip(batches, parent_encodings, parent_mask_lis, label_batches)

"""## Loss for FineTuning"""
def compute_ar_loss(loss_f, params, key, config, thread, encodings, labels):
    loss = 0.0

    for i, (batch, parent_batch, parent_mask, labels) in enumerate( get_decoder_batches(config, thread, encodings, labels) ):
        if i<turn*config['max_losses']:
            continue
        if i==(turn+1)*config['max_losses']:
            return (loss, True)
    
        key, subkey = jax.random.split(key)
        loss += loss_f(params['ar_classifier'], subkey, parent_batch, 
                        parent_mask, batch, labels)
          
    return (loss, False)

def compute_ar_accuracy(pred_f, params, key, config, thread, encodings, labels, turn)        
    all_preds = []
    all_labels = []
    
    for i, (batch, parent_batch, parent_mask, labels) in enumerate( get_decoder_batches(config, thread, encodings, labels) ):
        if i<turn*config['max_losses']:
            continue
        if i==(turn+1)*config['max_losses']:
            return ((all_preds, all_labels), True)
    
        key, subkey = jax.random.split(key)
        preds = pred_f(params['ar_classifier'], subkey, parent_batch, 
                        parent_mask, batch)
        
        preds, labels = get_pred_list(config, labels, preds, batch)	
        
        all_preds += preds
        all_labels += labels
    
    return ((all_preds, all_labels), False)
            
def ft_loss(featurizer_f, loss_f, params, key, init_thread, config, turn=0, mode='loss'):
    """
    init_thread:  list of jnp.arrays having token ids.
    mode: can be loss or accuracy.
    """
    thread = deepcopy(init_thread[0])
    labels = deepcopy(init_thread[1])
    
    encodings = []
    for batch in get_encoder_batches(config, thread):
        key, subkey = jax.random.split(key)
        features = featurizer_f(params['comments_encoder'], subkey, batch)
        encodings+=[elem for elem in features]    

    key, subkey = jax.random.split(key)    
    
    if mode=='accuracy':
        return compute_ar_accuracy(loss_f, params, subkey, config, thread, encodings, labels, turn)    
    
    return compute_ar_loss(loss_f, params, subkey, config, thread, encodings, labels, turn)
from sklearn.metrics import classification_report
import jax.numpy as jnp
import jax
import haiku as hk
from copy import deepcopy

def ft_cross_entropy(config, original_batch, logits, masked_token_ids):
    logits_mask = (masked_token_ids!=config['pad_id'])
    logits = jax.vmap(jnp.multiply, (None,2), 2)(logits_mask,logits)
    labels = hk.one_hot(original_batch, config['n_classes'])
    softmax_xent = -jnp.sum(labels*jax.nn.log_softmax(logits))
    total_masks = jnp.sum(logits_mask)
    if total_masks==0:
        return jnp.zeros(())
    softmax_xent /= total_masks
    return softmax_xent

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

def accuracy_pred(config, labels, logits, batch):

    preds = jnp.argmax(logits, axis=-1)
    logits_mask = (batch!=config['pad_id'])
    preds = jnp.where(logits_mask, preds, -1)
    
    all_preds = preds.reshape(-1).tolist()
    all_labels = labels.reshape(-1).tolist()

    return remove_pad_preds(all_preds, all_labels)


"""## Loss for FineTuning"""

def ft_loss(featurizer_f, logits_f, params, key, init_thread, config, turn=0, mode='loss'):
    """
    init_thread:  list of jnp.arrays having token ids.
    mode: can be loss or accuracy.
    """
    thread = deepcopy(init_thread[0])
    labels = deepcopy(init_thread[1])
    
    if mode=='accuracy':
        all_preds = []
        all_labels = []
    else:
        loss = 0.0
    
    remaining_comments = False

    empty_elem = jnp.asarray([config['pad_id']]*config['max_length'], dtype=jnp.int16)
    
    batches = get_batched_version(thread, config['featurizer_batch_size'], empty_elem)
    
    encodings = []
    for batch in batches:
        key, subkey = jax.random.split(key)
        features = featurizer_f(params['comments_encoder'], subkey, 
                                            batch)
        encodings+=[elem for elem in features]
    
    parent_encodings = [jnp.stack( encodings[:i]+[ jnp.zeros_like(encodings[0]) ]*(config['max_length']-i) ) 
                        for i in range(min(len(encodings), config['max_length']))]
    
    parent_mask_lis = [jnp.asarray( [0]*i +[1]*(config['max_length']-i) , dtype=jnp.int16) 
                        for i in range(min(len(encodings), config['max_length']))]
    
    parent_encodings = get_batched_version(parent_encodings, config['featurizer_batch_size'], parent_encodings[0])
    parent_mask_lis = get_batched_version(parent_mask_lis, config['featurizer_batch_size'], parent_mask_lis[0])
    label_batches = get_batched_version(labels, config['featurizer_batch_size'], labels[0])
    
    for i, (batch, parent_batch, parent_mask, labels) in enumerate( zip(batches, parent_encodings, parent_mask_lis, label_batches) ):
        
        if i<turn*config['max_losses']:
            continue
        
        if i==(turn+1)*config['max_losses']:
            remaining_comments=True
            break
    
        key, subkey = jax.random.split(key)
        logits = logits_f(params['ar_classifier'], subkey, parent_batch, 
                             parent_mask, batch)
        
        if mode=='accuracy':
            preds, labels = accuracy_pred(config, labels, logits, batch)	
            all_preds += preds
            all_labels += labels
        else:
            loss += ft_cross_entropy(config, labels, logits, batch)
        
    if mode=='accuracy':
        return (all_preds, all_labels), remaining_comments
        
    return loss, remaining_comments
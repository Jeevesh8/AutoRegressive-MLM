## For pre-training from scratch.
import jax
import jax.numpy as jnp
import haiku as hk
from haiku.data_structures import to_immutable_dict
import optax

import copy
import numpy as np
from functools import partial
from copy import deepcopy

from src.DataLoaders.json import load_reddit_data
from src.Tokenizers.tree_tokenizer import Tree_Tokenizer
from src.model.transformer import TransformerFeaturizer, ExtendedEncoder
from src.optimizers.adam import get_adam_opt
from src.Tokenizers.masking_utils import mask_batch_mlm
from src.Tokenizers.utils import tree_to_batch, batch_to_tree, gather_batch_parents

config = {
          #Data Parameters
          'max_length' : 128, 
          'featurizer_batch_size' : 4,
          'mlm_batch_size' : 4,
          'n_epochs' : 10,
          'data_files' : ['/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/train_period_data.jsonlist'],
          'discourse_markers_file' : '/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/Discourse_Markers.txt',
          
          #Model Parameters
          'intermediate_size' : 256,
          'n_heads' : 2,
          'n_layers' : 2,
          'hidden_size' : 128,
          'd_model' : 128,                                                      #same as hidden_size
          'max_losses' : 10,                                                    #max. number of losses to backpropagate at once
          'max_tree_size' : 60,
         
          #Embeddings Parameters
          'embed_dropout_rate' : 0.1,
          
          #MHA parameters
          'attention_drop_rate' : 0.1,
          
          #MLP parameters
          'fully_connected_drop_rate' : 0.1,
          
          #Training Parameters
          'learning_rate' : 1e-5,
          'max_grad_norm' : 1.0,
          'l2' : 0.1,

           #colab parameter
          'restart_from' : 0,
          }


data_loader = load_reddit_data(config)

def get_sentences():
    for tree in data_loader.tree_generator():
        yield tree['title'] + ' ' + tree['selftext']
        for id, comment in tree['comments'].items():
            yield comment['body']

lm_tokeniser = Tree_Tokenizer(config)
lm_tokeniser.train_tokenizer(str_iter=get_sentences())

print("Vocabulary : ", lm_tokeniser.tokenizer.get_vocab())

config['vocab_size'] = lm_tokeniser.tokenizer.get_vocab_size()

#Tokenization ids  
config['mask_id'] = lm_tokeniser.tokenizer.token_to_id("<mask>")
config['pad_id'] = lm_tokeniser.tokenizer.token_to_id("<pad>")
config['sos_id'] = lm_tokeniser.tokenizer.token_to_id("<s>")
config['eos_id'] = lm_tokeniser.tokenizer.token_to_id("</s>")
config['total_steps'] = len([0 for tree in data_loader.tree_generator()])
config['dsm_list'] = [lm_tokeniser.tokenizer.token_to_id(token)
                            for token in lm_tokeniser.dms]
config = hk.data_structures.to_immutable_dict(config)

print("Total steps: ", config['total_steps'])

def featurizer(token_ids, training, config):
    features = TransformerFeaturizer(config)(token_ids, training=training)
    return features

def logits_fn(comment_embds, comment_mask, masked_token_ids, training, config):
    logits = ExtendedEncoder(config)(comment_embds, comment_mask, 
                                     masked_token_ids, training=training)
    return logits

key, subkey = jax.random.split( jax.random.PRNGKey(42) )
pure_logits_fn = hk.transform(logits_fn)
pure_featurizer_fn = hk.transform(featurizer)

comment_encoding = lm_tokeniser.batch_encode_plus(['sample sentence']*config['featurizer_batch_size'])
token_encoding = lm_tokeniser.batch_encode_plus(['sample sentence']*config['mlm_batch_size'])

token_ids = np.asarray(lm_tokeniser.get_token_ids(token_encoding), dtype=np.int16)
comment_ids = np.asarray(lm_tokeniser.get_token_ids(comment_encoding), dtype=np.int16)

masked_token_ids, original_batch = mask_batch_mlm(subkey, config, token_ids)

key, subkey = jax.random.split(key)
featurizer_params = pure_featurizer_fn.init(subkey, comment_ids, True, config)

key, subkey = jax.random.split(key)
comment_embds = pure_featurizer_fn.apply(featurizer_params, subkey, comment_ids, True, config)

print(comment_embds.shape)
print(jnp.tile(comment_embds, config['max_length']).reshape(config['mlm_batch_size'], config['max_length'], -1).shape)
print(comment_embds.dtype, masked_token_ids.dtype)

key, subkey = jax.random.split(key)

comment_embds = jnp.tile(comment_embds, config['max_length']).reshape(config['mlm_batch_size'], config['max_length'], -1)
comment_mask = jnp.ones_like(comment_embds[:,:,0])

ExtendedEncoder_params = pure_logits_fn.init(subkey, comment_embds, 
                                             comment_mask, masked_token_ids,
                                             True, config)

params = to_immutable_dict( {'comments_encoder' : featurizer_params, 
                             'mlm_predictor' : ExtendedEncoder_params } )

@partial(jax.jit, static_argnums=(3,4))
def pure_featurizer(params, key, token_ids, training, config):
    key, subkey = jax.random.split(key)
    comment_embds = pure_featurizer_fn.apply(params, subkey, comment_ids, True, config)
    return comment_embds

@partial(jax.jit, static_argnums=(5,6))
def pure_logits(params, key, comment_embds, comment_mask, masked_token_ids, training, config):
    key, subkey = jax.random.split(key)
    logits = pure_logits_fn.apply(params, subkey, comment_embds, comment_mask, masked_token_ids, training=training, config=config)
    return logits

def cross_entropy(config, original_batch, logits, masked_token_ids):
    logits_mask = (masked_token_ids==config['mask_id'])
    logits = jax.vmap(jnp.multiply, (None,2), 2)(logits_mask,logits)
    labels = hk.one_hot(original_batch, config['vocab_size'])
    softmax_xent = -jnp.sum(labels*jax.nn.log_softmax(logits))
    total_masks = jnp.sum(logits_mask)
    if total_masks==0:
        return jnp.zeros(())
    softmax_xent /= total_masks
    return softmax_xent

def loss(params, key, init_tree, config, turn=0):
    """
    Calculates loss for all nodes of a single tree.
    The masked tokens of each location in a comment are predicted 
    conditioned on the embeddings of all the parent comments.
    """
    tree = deepcopy(init_tree)
    loss = 0.0
    remaining_comments = False

    #Prepare embeddings of each comment
    empty_elem = jnp.asarray([config['pad_id']]*config['max_length'], dtype=jnp.int16)
    batches = tree_to_batch(tree, config['featurizer_batch_size'],
                            key='tokenized_inputs', empty_elem=empty_elem)
    
    #print(len(batches))
    encodings = []
    for batch in batches:
        key, subkey = jax.random.split(key)
        features = pure_featurizer(params['comments_encoder'], subkey, 
                                            batch, True, config)
        encodings.append(features)
    tree = batch_to_tree(tree, encodings, config['featurizer_batch_size'], 
                         key='comment_embds')

    #Calculate loss for each masked position in each comment.
    comment_batches = tree_to_batch(tree, config['mlm_batch_size'], key=None, 
                                    empty_elem={}, include_root=False)
    
    batches = tree_to_batch(tree, config['mlm_batch_size'],
                            key='tokenized_inputs', empty_elem=empty_elem,
                            include_root=False)
    
    empty_elem = jnp.asarray([0]*config['d_model'], dtype=jnp.int16)
    
    for i, (original_batch, comment_batch) in enumerate( zip(batches, comment_batches) ):
        
        if i<turn*config['max_losses']:
            continue
        
        if i==(turn+1)*config['max_losses']:
            remaining_comments=True
            break
    
        parent_comment_embds, mask_for_embds = gather_batch_parents(tree, comment_batch, 
                                                                    config['max_length'], key='comment_embds', 
                                                                    empty_elem=empty_elem)
        key, subkey = jax.random.split(key)
        masked_batch, original_batch = mask_batch_mlm(subkey, config, original_batch)

        key, subkey = jax.random.split(key)
        logits = pure_logits(params['mlm_predictor'], subkey, parent_comment_embds, 
                             mask_for_embds, masked_batch, True, config)
        
        loss += cross_entropy(config, original_batch, logits, masked_batch)
        
        
    return loss, remaining_comments

#Load Pre-trained weights
import pickle
with open('/content/drive/MyDrive/2SCL/Argumentation/params.pkl', 'rb') as f:
    params = pickle.load(f)

opt = get_adam_opt(config)
opt_state = opt.init(params)

def update(opt_state, params, key, tree, config):
    turn = 0
    (batch_loss, remaining_comments), grad = jax.value_and_grad(loss, has_aux=True)(params, key, tree, config, turn)
    turn += 1

    while remaining_comments:
        print("Big tree, turn: ", turn)
        tup, grads = jax.value_and_grad(loss, has_aux=True)(params, key, tree, config, turn)
        turn += 1
        batch_loss += tup[0]
        grad = jax.tree_util.tree_multimap(lambda x,y: x+y, grad, grads) 
        remaining_comments = tup[1]
    
    updates, opt_state = opt.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)    
    return new_params, opt_state, batch_loss


#Training Loop
for _ in range(config['n_epochs']):
    
    losses = []
    for step, tree in enumerate(data_loader.tree_generator()):
        
        if _*config['n_examples']+step <= config['restart_from']:
          continue
        
        if step%100==0:
            print(f'[Step {step}]')

        tree = lm_tokeniser.tokenize_tree(tree)

        key, subkey = jax.random.split(key)
        params, opt_state, batch_loss = update(opt_state, params, subkey,
                                               tree, config)
        #print(batch_loss)
        losses.append(batch_loss)

        if step%100==0 and step!=0:
            print(sum(losses)/100)
            losses = []

        if step%1000==0:
            with open('/content/drive/MyDrive/2SCL/Argumentation/params.pkl', 'wb+') as f:
                pickle.dump(params, f)
            print("Wrote params to disk")
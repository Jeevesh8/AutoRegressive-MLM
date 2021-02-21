## For Fine-Tuning on downstream task of Argument Classification
import jax
import jax.numpy as jnp
import haiku as hk
from haiku.data_structures import to_immutable_dict
import optax
from sklearn.metrics import classification_report

import copy
import numpy as np
from functools import partial
from copy import deepcopy

from src.DataLoaders.xml import load_xml_data
from src.DataLoaders.json import load_reddit_data
from src.Tokenizers.thread_tokenizer import Thread_Tokenizer
from src.model.transformer import TransformerFeaturizer, FineTuningExtendedEncoder
from src.model.utils import logits_to_ar_classifier_params, print_keys
from src.optimizers.adam import get_adam_opt
from config import config
from loss_eval_utils import ft_loss
import wandb

"""## Loading Pre-Trained Tokenizers"""

if config['initialize_pretrained']=='RoBERTa':
    from src.model.utils import get_pretrained_weights, copy_available_keys
    
    from transformers import RobertaTokenizer

    huggingface_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    config['pt_hf_tokenizer'] = huggingface_tokenizer

"""## Data Loaders"""

data_loader = load_reddit_data(config)

train_data_loader = load_xml_data(config, split='train/')

valid_data_loader = load_xml_data(config, split='valid/')

test_data_loader = load_xml_data(config, split='test/')

"""## Training Tokenizer, if not using pre-trained one. """

if config['initialize_pretrained'] == '':

    lm_tokeniser = Thread_Tokenizer(config)
    lm_tokeniser.train_tokenizer(str_iter=data_loader.get_sentences())

"""## Or Load Pre-Trained Tokenizer"""
else: 
    #Will automatically load pre-trained version if config['pt_hf_tokenizer'] is defined.
    lm_tokeniser = Thread_Tokenizer(config)

print("Vocabulary : ", lm_tokeniser.tokenizer.get_vocab())

"""### Updating Config"""

config['vocab_size'] = lm_tokeniser.tokenizer.get_vocab_size()

#Tokenization ids  
config['mask_id'] = lm_tokeniser.tokenizer.token_to_id("<mask>")
config['pad_id'] = lm_tokeniser.tokenizer.token_to_id("<pad>")
config['sos_id'] = lm_tokeniser.tokenizer.token_to_id("<s>")
config['eos_id'] = lm_tokeniser.tokenizer.token_to_id("</s>")
config['dsm_list'] = [lm_tokeniser.tokenizer.token_to_id(token)
                            for token in lm_tokeniser.dms]
config['total_steps'] = len([0 for tree in train_data_loader.tree_generator()])

wandb.init(project='autoregressive-mlm-ft', config=config)
config = hk.data_structures.to_immutable_dict(config)

print("Total steps: ", config['total_steps'])

"""## Purifying the Model Functions and Getting Parameters"""

def featurizer(token_ids, training, config):
    features = TransformerFeaturizer(config)(token_ids, training=training)
    return features

def logits_fn(comment_embds, comment_mask, masked_token_ids, training, config):
    logits = FineTuningExtendedEncoder(config)(comment_embds, comment_mask, 
                                     masked_token_ids, training=training)
    return logits

key, subkey = jax.random.split( jax.random.PRNGKey(42) )

""" ### Purifying the impure functions"""

pure_logits_fn = hk.transform(logits_fn)
pure_featurizer_fn = hk.transform(featurizer)

""" ### Getting initial parameters """
comment_encoding = lm_tokeniser.batch_encode_plus(['sample sentence']*config['featurizer_batch_size'])
token_encoding = lm_tokeniser.batch_encode_plus(['sample sentence']*config['mlm_batch_size'])

token_ids = np.asarray(lm_tokeniser.get_token_ids(token_encoding), dtype=np.int16)
comment_ids = np.asarray(lm_tokeniser.get_token_ids(comment_encoding), dtype=np.int16)

mask_batch_mlm = get_masking_func(config)
masked_token_ids, original_batch = mask_batch_mlm(subkey, token_ids)

key, subkey = jax.random.split(key)
featurizer_params = pure_featurizer_fn.init(subkey, comment_ids, True, config)

key, subkey = jax.random.split(key)
comment_embds = pure_featurizer_fn.apply(featurizer_params, subkey, comment_ids, True, config)

key, subkey = jax.random.split(key)

comment_embds = jnp.tile(comment_embds, config['max_length']).reshape(config['mlm_batch_size'], config['max_length'], -1)
comment_mask = jnp.ones_like(comment_embds[:,:,0])

ExtendedEncoder_params = pure_logits_fn.init(subkey, comment_embds, 
                                             comment_mask, masked_token_ids,
                                             True, config)


""" ## Merging pre-trained and initialised parameters"""

if config['initialized_pretrained']!='':
    
    with open(config['initialized_pretrained'], 'rb') as f:
        pt_wts = pickle.load(f)

    featurizer_params = to_mutable_dict(featurizer_params)

    featurizer_params = copy_available_keys(pt_wts['comments_encoder'], featurizer_params,)
    
    ExtendedEncoder_params = to_mutable_dict(ExtendedEncoder_params)

    ExtendedEncoder_params = copy_available_keys(pt_wts['mlm_predictor'], ExtendedEncoder_params,)

params = to_immutable_dict( {'comments_encoder' : featurizer_params, 
                             'ar_classifier' : ExtendedEncoder_params } )

def pure_featurizer(training, config, params, key, token_ids):
    key, subkey = jax.random.split(key)
    comment_embds = pure_featurizer_fn.apply(params, subkey, comment_ids, True, config)
    return comment_embds

def pure_logits(training, config, params, key, comment_embds, comment_mask, masked_token_ids):
    key, subkey = jax.random.split(key)
    logits = pure_logits_fn.apply(params, subkey, comment_embds, comment_mask, masked_token_ids, training=training, config=config)
    return logits

def get_featurizer(training, config):
    return jax.jit(partial(pure_featurizer, training, config))

def get_logits_fn(training, config):
    return jax.jit(partial(pure_logits, training, config))

featurizer_f = get_featurizer(True, config)
logits_f = get_logits_fn(True, config)

loss = partial(ft_loss, featurizer_f, logits_f, mode='loss')

"""## Optimizer"""

opt = get_adam_opt(config)
opt_state = opt.init(params)

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

"""## Evaluation Functions """
eval_featurizer_f = get_featurizer(False, config)
eval_logits_f = get_logits_fn(False, config)

accuracy = partial(ft_loss, eval_featurizer_f, eval_logits_f, mode='accuracy')

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

def evaluate(config, params, data_loader, lm_tokeniser, key):
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


"""## Training Function"""
def train(config, params, train_data_loader, key, opt_state):
    losses = []
    val_losses = []

    for _ in range(config['n_epochs']):

        for step, thread in enumerate(train_data_loader.thread_generator()):
                        
            if step%(config['total_steps']//3)==0:
                print(f'[Step {step}]')

            thread = lm_tokeniser.tokenize_thread(thread)

            key, subkey = jax.random.split(key)
            params, opt_state, batch_loss = update(opt_state, params, subkey,
                                                thread, config)
            
            losses.append(batch_loss)

            if step%(config['total_steps']//3)==0:
                print(sum(losses)/len(losses))
                losses = []

            if step==config['total_steps']-1:
                all_preds, all_labels = evaluate(config, params, valid_data_loader, key)
                wandb.log({'Validation' : classification_report(all_labels, all_preds, labels=[0,1,2], 
                                            target_names=['Non-Argumentative', 'Claim', 'Premise'])})
    
                evaluate(config, params, test_data_loader, key)
                wandb.log({'Test' : classification_report(all_labels, all_preds, labels=[0,1,2], 
                                            target_names=['Non-Argumentative', 'Claim', 'Premise'])})
    
    return val_losses

"""## Loop over HyperParameters"""

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
        
        init_params = copy.deepcopy(params)

        val_losses = train(config, init_params, train_data_loader, key, opt_state)
        
        valid_epoch_losses.append( val_losses )
        
        wandb.log({'learning_rate':lr, 'dropout_rate': dr})
        print(f"Learning rate={lr}, Dropout Rate={dr} Losses : ", valid_epoch_losses[-1])
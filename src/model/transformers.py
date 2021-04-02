import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import gin

from src.model.transformer import TransformerBlock, TransformerDecoderBlock
from src.model.embeddings import Embedding

from src.model.misc import crf_layer, GRU, Linear

@gin.configurable
class TransformerFeaturizer(hk.Module):
    """
    A transformer module that outputs the features of running 
    config['n_layers'] TransformerBlock's on the input sequence.
    """

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config

    def get_mask(self, token_ids):
        return (jnp.bitwise_or(token_ids==self.config['pad_id'], 
                               token_ids==self.config['mask_id'])).astype(jnp.float32)
    
    def __call__(self, token_ids, training=False, is_autoregressive=False):
        
        x = Embedding(self.config)(token_ids, lang_ids=None, training=training)
        
        mask = self.get_mask(token_ids)

        for layer_num in range(self.config['n_layers']):
            x = TransformerBlock(self.config, layer_num)(x, mask,
                                                        training=training, 
                                                        is_autoregressive=is_autoregressive)
        
        x = jax.lax.stop_gradient(x[:,0,:])
        return x

@gin.configurable
class LogitsTransformer(hk.Module):
    """
    This module consists of TransformerFeaturizer followed by a 
    linear layer to map the features to logits for the vocabulary.
    """

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config

    def __call__(self, token_ids, lang_ids=None, training=False, is_autoregressive=False):
        x = TransformerFeaturizer(self.config)(token_ids, lang_ids, 
                                               training=training, is_autoregressive=is_autoregressive)
        logits = hk.Linear(output_size=self.config['vocab_size'])(x)
        return logits

@gin.configurable
class VaswaniTransformer(hk.Module):
    """
    This module implements a complete Encoder-Decoder Transformer architecture
    as presented in the original paper by Vaswani et. al. 
    Outputs the logits over the target vocabulary.
    """

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
    
    def get_mask(self, token_ids):
        return (jnp.bitwise_or(token_ids==self.config['pad_id'], 
                               token_ids==self.config['mask_id'])).astype(jnp.float32)
        
    def __call__(self, src_token_ids, tgt_token_ids, src_lang_ids=None, tgt_lang_ids=None, training=False):
        
        x_embds = TransformerFeaturizer(self.config)(src_token_ids, lang_ids=src_lang_ids,
                                                    training=True)
        
        src_mask = self.get_mask(src_token_ids)
        tgt_mask = self.get_mask(tgt_token_ids)

        y = Embedding(self.config)(tgt_token_ids, lang_ids=tgt_lang_ids, training=training)

        for layer_num in range(self.config['n_layers']):
            y = TransformerDecoderBlock(self.config, layer_num)(y, tgt_mask, src_mask, x_embds, training=training)
        
        tgt_features = y
        logits = hk.Linear(output_size=self.config['tgt_vocab_size'])(tgt_features)

        return logits

@gin.configurable
class BaseExtendedEncoder(hk.Module):
    """
    This module obtains features corresponding to an input sequence,
    by passing it through config['n_layers'] TransformerDecoderBlock's.
    Each TransformerDecoderBlock is conditioned on additional sequence of embeddings
    passed via the "comment_embds" argument in __call__().

    It beahves like a Transformer Encoder, extended so as to condition the output
    features on the additional sequence of pre-computed embeddings provided.

    In our model, we obtain the "comment_embds" by concatenating features of every
    previous comment via TransformerFeaturizer.
    """

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
        self.embed_layer = Embedding(self.config)

    def get_mask(self, token_ids):
        return (jnp.bitwise_or(token_ids==self.config['pad_id'], 
                               token_ids==self.config['mask_id'])).astype(jnp.float32)
    
    def __call__(self, comment_embds, comments_mask, masked_token_ids, training=False):
        
        y = self.embed_layer(masked_token_ids, lang_ids=None, training=training)

        tgt_mask = self.get_mask(masked_token_ids)

        for layer_num in range(self.config['n_layers']):
            y = TransformerDecoderBlock(self.config, layer_num)(y, tgt_mask, 
                                                                  comments_mask, comment_embds,
                                                                  training=training)
        
        return y

@gin.configurable
class ExtendedEncoder(BaseExtendedEncoder):
    """
    This module consists of the BaseExtendedEncoder, followed by a 
    linear layer to map the features to logits corresponding to the 
    target vocabulary. 
    
    This module is used to pre-training for the AutoRegressive-MLM task.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name=name)
    
    def __call__(self, comment_embds, comments_mask, masked_token_ids, training=False):
        y = super().__call__(comment_embds, comments_mask, masked_token_ids, training=training)
        w = self.embed_layer.word_emb_layer.embeddings
        logits = jnp.tensordot(y, w, (-1,-1))
        return logits

    def __call__(self, comment_embds, comment_mask, masked_token_ids, original_batch, training=False):
        
        logits = self.__call__(comment_embds, comments_mask, masked_token_ids, training=training)

        logits_mask = (masked_token_ids==self.config['mask_id'])
        logits = jax.vmap(jnp.multiply, (None,2), 2)(logits_mask, logits)
        labels = hk.one_hot(original_batch, self.config['vocab_size'])
        softmax_xent = -jnp.sum(labels*jax.nn.log_softmax(logits))
        total_masks = jnp.sum(logits_mask)
        
        if total_masks==0:
            return jnp.zeros(())
        
        softmax_xent /= total_masks
        
        return softmax_xent
    
    def predict(self, comment_embds, comments_mask, masked_token_ids, training=False):
        logits = self.__call__(comment_embds, comments_mask, masked_token_ids, training=training)
        return jnp.argmax(logits, axis=-1)

@gin.configurable
class FineTuningExtendedEncoder(BaseExtendedEncoder):
    """
    For finetuning, BaseExtendedEncoder with additional 
    classification modules like hk.Linear or GRU or CRF.

    There are two overloaded __call__() functions, one takes 
    labels, and the other doesn't.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name=name)
        
        self.config = config
        
        if config['last_layer']=='GRU':
            self.last_layer = GRU(config['n_classes'])
        
        elif config['last_layer']=='Linear':
            self.last_layer = Linear(output_size=config['n_classes'])
        
        elif config['last_layer']=='crf':
            transition_init = hk.initializers.Constant(jnp.array(config['transition_init']))
         
            self.lin = hk.Linear(config['n_classes'])
            self.last_layer = crf_layer(n_classes=config['n_classes'],
                                       transition_init=transition_init,
                                       scale_factors=config['scale_factors'],
                                       init_alphas=config['init_alphas'])
        else:
            raise NotImplementedError("No implementation for finetuning with last layer as : ", config['last_layer'])
    
    def __call__(self, comment_embds, comments_mask, token_ids, training=False):
        """
        Computes features using ExtendedEncoder and computes 
        further logits.
        """
        embds = super().__call__(comment_embds, comments_mask, token_ids, training=training)
        
        if training:
            embds = hk.dropout(rng=hk.next_rng_key(),
                                rate=self.config['classifier_drop_rate'],
                                x=embds)
        
        if self.config['last_layer']=='crf':
            return self.lin(embds)
            
        return self.last_layer(embds)
    
    def __call__(self, comment_embds, comments_mask, token_ids, labels, training=False):
        """
        Returns the loss using the predictions from the logits predicted inside last_layer.
        """
        embds = super().__call__(comment_embds, comments_mask, token_ids, training=training)
        
        if training:
            embds = hk.dropout(rng=hk.next_rng_key(),
                               rate=self.config['classifier_drop_rate'],
                               x=embds)
        
        if self.config['last_layer']=='crf':
            embds = self.lin(embds)
        
        lengths = jnp.sum((token_ids!=self.config['pad_id']), axis=-1)

        return self.last_layer(embds, lengths, labels)
    
    def predict(self, comment_embds, comments_mask, token_ids, training=False):
        """
        Predicts the most likely sequence of taggings for the input sequence 
        provided via the token_ids argument. 
        """
        embds = super().__call__(comment_embds, comments_mask, token_ids, training=training)
        
        if training:
            embds = hk.dropout(rng=hk.next_rng_key(),
                               rate=self.config['classifier_drop_rate'],
                               x=embds)
        
        if self.config['last_layer']=='crf':
            embds = self.lin(embds)
            lengths = jnp.sum((token_ids!=self.config['pad_id']), axis=-1)
            return self.last_layer.batch_viterbi_decode(embds, lengths)[0]        
        
        else:
            return jnp.argmax(self.last_layer(embds), axis=-1)
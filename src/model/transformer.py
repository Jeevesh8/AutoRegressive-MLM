import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax
from src.model.embeddings import Embedding
from src.model.utils import Scope

class TransformerBlock(hk.Module):

    def __init__(self, config, layer_num=None):
        super().__init__()
        self.config = config
        self.n = layer_num
        self.pt = 'pretrained' in config
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, f'encoder/layer_{self.n}/')
    
    def get_init(self, elem):
        return  hk.initializers.Constant(elem) if self.pt else None

    def __call__(self, x, mask, training=False, is_autoregressive=False):

        attention_output = MultiHeadAttention(self.config, self.n)(x, x, mask,
                                                                   training=training, 
                                                                   is_autoregressive=is_autoregressive)
        
        residual = attention_output+x

        attention_output = hk.LayerNorm(axis=-1,
                                        create_scale=True,
                                        create_offset=True,
                                        scale_init=self.pt_wts['attention/output/LayerNorm/gamma'],
                                        offset_init=self.pt_wts['attention/output/LayerNorm/beta'],)(residual)

        mlp_output = TransformerMLP(self.config, self.n)(attention_output, training=training)

        output_residual = mlp_output+attention_output
        
        layer_output = hk.LayerNorm(axis=-1,
                                    create_scale=True,
                                    create_offset=True,
                                    scale_init=self.pt_wts['output/LayerNorm/gamma'],
                                    offset_init=self.pt_wts['output/LayerNorm/beta'])(output_residual)
        
        return layer_output

class TransformerDecoderBlock(hk.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.n = layer_num
        self.pt = 'pretrained' in config
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, f'encoder/layer_{self.n}/')
    
    def __call__(self, y, tgt_mask, src_mask, x_embds, training=False):

        attention_output = MultiHeadAttention(self.config, self.n)(y, y, tgt_mask,
                                                                  training=training, 
                                                                  is_autoregressive=False)
        
        residual = attention_output+y

        self_attention_output = hk.LayerNorm(axis=-1,
                                             create_scale=True,
                                             create_offset=True,
                                             scale_init=self.pt_wts['attention/output/LayerNorm/gamma'],
                                             offset_init=self.pt_wts['attention/output/LayerNorm/beta'],)(residual)
        
        attention_output = MultiHeadAttention(self.config)(x_embds, self_attention_output, src_mask, 
                                                           training=training, is_autoregressive=False)
        
        residual = attention_output+self_attention_output

        attention_output = hk.LayerNorm(axis=-1,
                                        create_scale=True,
                                        create_offset=True,)(residual)
        
        mlp_output = TransformerMLP(self.config, self.n)(attention_output, training=training)

        output_residual = mlp_output+attention_output

        layer_output = hk.LayerNorm(axis=-1,
                                    create_scale=True,
                                    create_offset=True,
                                    scale_init=self.pt_wts['output/LayerNorm/gamma'],
                                    offset_init=self.pt_wts['output/LayerNorm/beta'],)(output_residual)
        
        return layer_output

class MultiHeadAttention(hk.Module):
    def __init__(self, config, layer_num=None):
        super().__init__()
        self.config = config
        self.n = layer_num
        self.pt = 'pretrained' in config
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, f'encoder/layer_{self.n}/attention/')

    def _split_into_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.config['n_heads'], x.shape[2]//self.config['n_heads']])
    
    def get_attn_mask(self, seq_len):
        mask = jnp.ones([seq_len, seq_len])
        mask = jnp.triu(mask, k=1)
        return mask*-2**32
    
    def __call__(self, x, y, mask, training=False, is_autoregressive=False):
        
        queries = hk.Linear(output_size=self.config['d_model'],
                            w_init=self.pt_wts['query/kernel'],
                            b_init=self.pt_wts['query/bias'])(y)
        
        keys = hk.Linear(output_size=self.config['d_model'],
                        w_init=self.pt_wts['key/kernel'],
                        b_init=self.pt_wts['key/bias'])(x)
        
        values = hk.Linear(output_size=self.config['d_model'],
                           w_init=self.pt_wts['value/kernel'],
                           b_init=self.pt_wts['value/bias'])(x)
        
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)

        attention_logits = jnp.einsum('btnh,bsnh->bnts', queries, keys)
        attention_logits /= jnp.sqrt(queries.shape[-1])

        attention_logits += jnp.reshape(mask*-2**32, [mask.shape[0],1,1,mask.shape[1]])
        
        if is_autoregressive:
            attention_logits += self.get_attn_mask(y.shape[1])

        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        per_head_attention_output = jnp.einsum('bsnh,bnts->btnh', values, attention_weights)
        
        attention_output = jnp.reshape(per_head_attention_output, 
                                       [per_head_attention_output.shape[0], per_head_attention_output.shape[1], -1])

        attention_output = hk.Linear(output_size=self.config['d_model'],
                                     w_init=self.pt_wts['output/dense/kernel'],
                                     b_init=self.pt_wts['output/dense/bias'])(attention_output)
        
        if training:
            attention_output = hk.dropout(rng=hk.next_rng_key(),
                                          rate=self.config['attention_drop_rate'],
                                          x=attention_output)
        
        return attention_output


def gelu(x):
    return x*0.5*(1.0+jax.scipy.special.erf(x / jnp.sqrt(2.0)))


class TransformerMLP(hk.Module):

    def __init__(self, config, layer_num=None):
        super().__init__()
        self.config = config
        self.n = layer_num
        self.pt = 'pretrained' in config
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, f'encoder/layer_{self.n}/')

    def __call__(self, x, training=False):

        intermediate_output = hk.Linear(output_size=self.config['intermediate_size'],
                                        w_init=self.pt_wts['intermediate/dense/kernel'],
                                        b_init=self.pt_wts['intermediate/dense/bias'],)(x)

        intermediate_output = gelu(intermediate_output)

        output = hk.Linear(output_size=self.config['d_model'],
                           w_init=self.pt_wts['output/dense/kernel'],
                           b_init=self.pt_wts['output/dense/bias'],)(intermediate_output)
        
        if training:
            output = hk.dropout(rng=hk.next_rng_key(),
                                rate=self.config['fully_connected_drop_rate'],
                                x=output)
        
        return output


class TransformerFeaturizer(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_mask(self, token_ids):
        return (jnp.bitwise_or(token_ids==self.config['pad_id'], 
                               token_ids==self.config['mask_id'])).astype(jnp.float32)
    
    def __call__(self, token_ids, lang_ids=None, training=False, is_autoregressive=False):
        
        x = Embedding(self.config)(token_ids, lang_ids=lang_ids, training=training)
        
        mask = self.get_mask(token_ids)

        for layer_num in range(self.config['n_layers']):
            x = TransformerBlock(self.config, layer_num)(x, mask,
                                                        training=training, 
                                                        is_autoregressive=is_autoregressive)
        
        x = jnp.average(x, axis=1)
        return x

class LogitsTransformer(hk.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, token_ids, lang_ids=None, training=False, is_autoregressive=False):
        x = TransformerFeaturizer(self.config)(token_ids, lang_ids, 
                                               training=training, is_autoregressive=is_autoregressive)
        logits = hk.Linear(output_size=self.config['vocab_size'])(x)
        return logits

class VaswaniTransformer(hk.Module):
    
    def __init__(self, config):
        super().__init__()
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

class ExtendedEncoder(hk.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pt = 'pretrained' in config
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, f'model/masked-language-model/')
        self.embed_layer = Embedding(self.config)

    def init_final_layer_bias(self):
        b = self.pt_wts['output_bias'].constant
        n_extra = len(self.config['extra_tokens'])
        extra_b = jnp.zeros((n_extra,), dtype=b.dtype)
        return jnp.concatenate([b,extra_b], axis=0)

    def get_mask(self, token_ids):
        return (jnp.bitwise_or(token_ids==self.config['pad_id'], 
                               token_ids==self.config['mask_id'])).astype(jnp.float32)
    
    def __call__(self, comment_embds, comments_mask, masked_token_ids, training=False):
        
        y = self.embed_layer(masked_token_ids, lang_ids=None, training=training)

        tgt_mask = self.get_mask(masked_token_ids)

        for layer_num in range(self.config['n_layers']):
            y = TransformerDecoderBlock(self.config, layer_num+6)(y, tgt_mask, 
                                                                  comments_mask, comment_embds,
                                                                  training=training)
        
        if self.pt:
            logits = self.embed_layer.word_emb_layer.embeddings*y + hk.get_parameter('output_bias', 
                                                                    [self.config['vocab_size']], 
                                                                    w_init=hk.initializers.Constant(self.init_final_layer_bias()))
        else:
            logits = hk.Linear(output_size=config['vocab_size'],)(y)
        return logits
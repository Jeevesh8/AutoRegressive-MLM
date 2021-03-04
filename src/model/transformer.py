import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax
import gin

@gin.configurable
class TransformerBlock(hk.Module):

    def __init__(self, config, layer_num=None, name=None):
        if layer_num is not None:
            name = name+'_'+str(layer_num)
        super().__init__(name=name)
        self.config = config
        self.n = layer_num

    def __call__(self, x, mask, training=False, is_autoregressive=False):

        attention_output = MultiHeadAttention(self.config, self.n)(x, x, mask,
                                                                   training=training, 
                                                                   is_autoregressive=is_autoregressive)
        
        residual = attention_output+x

        attention_output = hk.LayerNorm(axis=-1,
                                        create_scale=True,
                                        create_offset=True,
                                        name='attention_output_LayerNorm',)(residual)

        mlp_output = TransformerMLP(self.config, self.n)(attention_output, training=training)

        output_residual = mlp_output+attention_output
        
        layer_output = hk.LayerNorm(axis=-1,
                                    create_scale=True,
                                    create_offset=True,
                                    name='output_LayerNorm',)(output_residual)
        
        return layer_output

@gin.configurable
class TransformerDecoderBlock(hk.Module):

    def __init__(self, config, layer_num, name=None):
        if layer_num is not None:
            name = name+'_'+str(layer_num)
        super().__init__(name=name)
        self.config = config
        self.n = layer_num
        
    def __call__(self, y, tgt_mask, src_mask, x_embds, training=False):

        attention_output = MultiHeadAttention(self.config, self.n)(y, y, tgt_mask,
                                                                  training=training, 
                                                                  is_autoregressive=False)
        
        residual = attention_output+y

        self_attention_output = hk.LayerNorm(axis=-1,
                                             create_scale=True,
                                             create_offset=True,
                                             name='attention_output_LayerNorm',)(residual)
        
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
                                    name='output_LayerNorm',)(output_residual)
        
        return layer_output

@gin.configurable
class MultiHeadAttention(hk.Module):
    def __init__(self, config, layer_num=None, name=None):
        super().__init__(name=name)
        self.config = config
        self.n = layer_num

    def _split_into_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.config['n_heads'], x.shape[2]//self.config['n_heads']])
    
    def get_attn_mask(self, seq_len):
        mask = jnp.ones([seq_len, seq_len])
        mask = jnp.triu(mask, k=1)
        return mask*-2**32
    
    def __call__(self, x, y, mask, training=False, is_autoregressive=False):
        
        queries = hk.Linear(output_size=self.config['d_model'],
                            name='query',)(y)
        
        keys = hk.Linear(output_size=self.config['d_model'],
                         name='key',)(x)
        
        values = hk.Linear(output_size=self.config['d_model'],
                           name='value',)(x)
        
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
                                     name='output_dense',)(attention_output)
        
        if training:
            attention_output = hk.dropout(rng=hk.next_rng_key(),
                                          rate=self.config['attention_drop_rate'],
                                          x=attention_output)
        
        return attention_output


def gelu(x):
    return x*0.5*(1.0+jax.scipy.special.erf(x / jnp.sqrt(2.0)))

@gin.configurable
class TransformerMLP(hk.Module):

    def __init__(self, config, layer_num=None, name=None):
        super().__init__(name=name)
        self.config = config
        self.n = layer_num
        
    def __call__(self, x, training=False):

        intermediate_output = hk.Linear(output_size=self.config['intermediate_size'],
                                        name='intermediate_dense',)(x)

        intermediate_output = gelu(intermediate_output)

        output = hk.Linear(output_size=self.config['d_model'],
                           name='output_dense',)(intermediate_output)
        
        if training:
            output = hk.dropout(rng=hk.next_rng_key(),
                                rate=self.config['fully_connected_drop_rate'],
                                x=output)
        
        return output
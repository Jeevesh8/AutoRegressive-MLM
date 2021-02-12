import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np
from src.model.utils import Scope

class Embedding(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pt = 'pretrained' in config
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, 'embeddings/')
        
        if self.pt:
            self.word_emb_layer = hk.Embed(embedding_matrix=self.make_pt_init_wts())
        else:
            self.word_emb_layer = hk.Embed(vocab_size=config['vocab_size'],
                                           embed_dim=config['d_model'])
    
    def make_pt_init_wts(self):
        w = self.pt_wts['word_embeddings'].constant
        stddev = 1. / np.sqrt(self.config['d_model'])
        
        n_extra = len(self.config['extra_tokens'])
        key, subkey = jax.random.split( jax.random.PRNGKey(42) )
        extra_w = stddev*jax.random.truncated_normal(subkey, -2., 2., 
                                                     shape=[n_extra, self.config['d_model']])        
        return jnp.concatenate([w, extra_w], axis=0)
        
    def __call__(self, token_ids, lang_ids=None, training=False):
        """
        token_ids: ints of shape (batch, n_seq)
        """
        
        flat_token_ids = jnp.reshape(token_ids, [-1])
        
        flat_token_embeddings = self.word_emb_layer(flat_token_ids)

        token_embeddings = jnp.reshape(flat_token_embeddings, [token_ids.shape[0], -1, self.config['d_model']])
        
        embeddings = token_embeddings + PositionEmbeddings(self.config)()
        
        if lang_ids is not None:
            embeddings += LanguageEmbeddings(self.config)(lang_ids)
        
        embeddings = hk.LayerNorm(axis=-1,
                                  create_scale=True,
                                  create_offset=True,                                  
                                  scale_init=self.pt_wts['LayerNorm/gamma'],
                                  offset_init=self.pt_wts['LayerNorm/beta'],)(embeddings)
        if training:
            embeddings = hk.dropout(hk.next_rng_key(),
                                    rate=self.config['embed_dropout_rate'],
                                    x=embeddings)
        
        return embeddings


class PositionEmbeddings(hk.Module):
    """
    A position embedding of size [max_seq_leq, word_embedding_dim]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pt = 'pretrained' in config
        self.offset = 2 if self.pt else 0
        self.pt_wts = Scope( self.config['pretrained'] if self.pt else None, 'embeddings/')

    def get_init_pe(self):
        
        pe = np.zeros([self.config['max_length'], self.config['d_model']])
        
        position = np.arange(0, self.config['max_length']).reshape(-1,1)
        
        div_term = np.exp(np.arange(0, self.config['d_model'],2)*
                          -np.log(10000.0)/self.config['d_model'])
        
        pe[:, 0::2] = np.sin(position*div_term)
        pe[:, 1::2] = np.cos(position*div_term)
        
        return pe

    def get_init(self):
        pretrained_embeds = self.pt_wts['position_embeddings']
        if pretrained_embeds is not None:
            return pretrained_embeds.constant
        else:
            return self.get_init_pe()
    
    def __call__(self):
        
        position_weights = hk.get_parameter("position_embeddings",
                                            [self.config['max_length']+self.offset, self.config['d_model']],
                                            init=hk.initializers.Constant(self.get_init()))
        
        start = self.offset
        end = self.offset+self.config['max_length']
        
        return position_weights[start:end]


class LanguageEmbeddings(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, lang_ids):

        return hk.Embed(vocab_size=len(self.config['lang2id'])+1, 
                        embed_dim=self.config['d_model'])(lang_ids)
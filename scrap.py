@gin.configurable
class FineTuningExtendedEncoderCrf(BaseExtendedEncoder):
    """
    For finetuning, base extended encoder with additional 
    CRF module.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name=name)
        
        transition_init = hk.initializers.constant(jnp.array(config['transition_init']))
        
        self.last_layer = crf_layer(n_classes=config['n_classes'],
                                    transition_init=transition_init,)

    def __call__(self, comment_embds, comments_mask, masked_token_ids, labels, training=False):
        y = super().__call__(comment_embds, comments_mask, masked_token_ids, training=training)
        
        if training:
            new_embds = hk.dropout(rng=hk.next_rng_key(),
                                   rate=self.config['classifier_drop_rate'],
                                   x=y)
        
        lengths = jnp.sum((masked_token_ids!=self.config['pad_id']), axis=-1)

        return self.last_layer(new_embds, lengths, labels)

    def predict(self, comment_embds, comments_mask, masked_token_ids, training=False):
        y = super().__call__(comment_embds, comments_mask, masked_token_ids, training=training)
        
        if training:
            new_embds = hk.dropout(rng=hk.next_rng_key(),
                                   rate=self.config['classifier_drop_rate'],
                                   x=y)
        
        lengths = jnp.sum((masked_token_ids!=self.config['pad_id']), axis=-1)

        return self.last_layer.batch_viterbi_decode(new_embds, lengths)

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

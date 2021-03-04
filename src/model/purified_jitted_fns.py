#Simple functions to transform
def get_fn_to_transform(module, config, training=True):

    def fn_to_transform(*args, **kwargs):
        return module(config)(*args, **kwargs)
    def pred_fn_to_transform(*args, **kwargs):
        return module(config).predict(*args, **kwargs)
    if not training:
        return pred_fn_to_transform    
    return fn_to_transform

def get_jittable_fn(transformed_fn):
    
    def pure_featurizer(training, config, params, key, *args):
        key, subkey = jax.random.split(key)
        comment_embds = transformed_fn.apply(params, subkey, *args, training=training, config=config)
        return comment_embds
    
    return pure_featurizer

def get_jitted_fn(pure_fn, training, config):
    return jax.jit(partial(pure_fn, training, config))

def get_pure_jitted_fn(transformed_fn, training, config):
    return get_jitted_fn(get_jittable_fn(transformed_fn), training, config)
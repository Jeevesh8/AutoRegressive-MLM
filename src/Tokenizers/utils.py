import jax.numpy as jnp

def decode_to_str(batch_text, max_len=512) :
    """
    Converts bytes string data to text. And truncates to max_len. 
    """
    return [ ' '.join(text.decode('utf-8').split()[:max_len]) if isinstance(text, bytes) else text[:max_len]
             for text in batch_text ]

def concat_and_decode(train_batch1, train_batch2, max_len=512, sep_tok='</s>'):
    """
    For concatenating and decoding two batches of parallel sentences for TLM.
    """
    max_len = (max_len//2) - 1

    batch_text = [
                  (' '.join(text1.decode('utf-8').split()[:max_len]) if isinstance(text1, bytes) else text1[:max_len]) + sep_token +
                  (' '.join(text2.decode('utf-8').split()[:max_len]) if isinstance(text2, bytes) else text2[:max_len])
                  for text1, text2 in zip(train_batch1, train_batch2)
    ]

    return batch_text

def tree_to_batch(tree, batch_size, key='tokenized_inputs', empty_elem='', include_root=True):
    """
    Takes in a tree and batches together elements of a particular key,
    returning a list of batches.
    """
    elems = ([tree[key]] if include_root else []) \
             + [ (comment if key is None else comment[key]) 
                for id, comment in tree['comments'].items()]

    elems += [empty_elem]*( batch_size-(len(elems)%batch_size) )
    
    batches = [jnp.asarray(elems[i*(batch_size) : (i+1)*batch_size], dtype=jnp.int16) 
                for i in range(len(elems)//batch_size)]
    
    return batches

def batch_to_tree(tree, batches, batch_size, key='comment_embds', include_root=True):
    """
    Takes in a List of batches, and allocates each element of the batch to 
    the appropriate element of the tree.
    """
    elems = [batch[i] for i in range(batch_size) for batch in batches]
    idx=-1
    if include_root:
        idx+=1
        tree[key] = elems[idx]
    
    for id, comment in tree['comments'].items():
        idx+=1
        comment[key]=elems[idx]
    
    return tree

def gather_parents(tree, elem, key='tokenized_inputs', include_root=True):
    """
    Returns a list of attributes represented by key of all parents,
    of elem.
    """
    lis=[]
    if 'parent_id' in elem:
        parent_id = elem['parent_id']
        while parent_id != tree['id']:
            elem = tree['comments'][parent_id]
            parent_id = elem['parent_id']
            lis.append(elem[key])
        return ([tree[key]] if include_root else []) +lis
    return lis

def gather_batch_parents(tree, elems, max_length, key='tokenized_inputs', empty_elem='', include_root=True):
    """
    Batched version of gather_parents.
    Returns array of size [ len(elems), max_length, len(elem[key]) ] 
    and corresponding mask.
    """
    batch = []
    mask = []
    for elem in elems:
        parent_attrs = gather_parents(tree, elem, 
                                      key=key, include_root=include_root)[:max_length]

        parent_attrs = parent_attrs + [empty_elem]*(max_length-len(parent_attrs))
        mask.append([0]*len(parent_attrs) + [1]*(max_length-len(parent_attrs))) 
        batch.append(parent_attr)
    return jnp.asarray(batch, dtype=jnp.float32), jnp.asarray(mask, dtype=jnp.int16)
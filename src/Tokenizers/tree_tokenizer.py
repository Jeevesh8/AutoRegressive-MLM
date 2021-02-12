from src.Tokenizers.base_tokenizer import Base_Tokenizer

class Tree_Tokenizer(Base_Tokenizer):
    
    def __init__(self, config):
        super().__init__(self, config)
    
    def tokenize_tree(self, tree):
        
        lis = ['<s> '+tree['title']+' </s> '+tree['selftext']+' </s>' ]
        
        for id, comment in tree['comments'].items():
            lis.append(comment['body'])
        
        token_ids = jnp.asarray( self.get_token_ids(self.batch_encode_plus(lis)), dtype=jnp.int16)
        
        i=0
        tree['tokenized_inputs'] = token_ids[i]
        
        for id, comment in tree['comments'].items():
            i+=1
            comment['tokenized_inputs']=token_ids[i]
        
        return tree
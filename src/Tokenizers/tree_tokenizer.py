from src.Tokenizers.base_tokenizer import Base_Tokenizer
from tokenizers.processors import TemplateProcessing

class Tree_Tokenizer(Base_Tokenizer):
    
    def __init__(self, config):
        super().__init__(config)
    
    def set_up_tokenizer(self):
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('<pad>'),
                                      length=self.config['max_length'])
        
        self.tokenizer.enable_truncation(self.config['max_length'])

        self.tokenizer.post_processor = TemplateProcessing(single = "<s>:1 $A:1 </s>:1",
                                                           pair = "<s>:1 $A:1 </s>:1 </s>:2 $B:2 </s>:2",
                                                           special_tokens=[('<s>',1), ('</s>',2)])
    def tokenize_tree(self, tree):
        
        i=0
        if 'author' not in tree:
            lis = ['<s> <unu> '+tree['title']+' </s> '+tree['selftext']+' </s>' ]
        else:
            lis = [f'<s> <user_{i}> '+tree['title']+' </s> '+tree['selftext']+' </s>' ]
            authors = {tree['author']:i}        

        for id, comment in tree['comments'].items():
            if 'author' not in comment:
                lis.append('<unu> ' + comment['body'])
            else:
                if comment['author'] not in authors:
                    i+=1
                    author[comment['author']] = i
                
                if author[comment['author']]<self.config['max_labelled_users_per_tree']:
                    author_idx = author[comment['author']]
                    lis.append( f'<user_{author_idx}> ' + comment['body'] )
                else:
                    lis.append('<unu> 'comment['body'])
        
        token_ids = jnp.asarray( self.get_token_ids(self.batch_encode_plus(lis)), dtype=jnp.int16)
        
        i=0
        tree['tokenized_inputs'] = token_ids[i]
        
        for id, comment in tree['comments'].items():
            i+=1
            comment['tokenized_inputs']=token_ids[i]
        
        return tree
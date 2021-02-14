from src.Tokenizers.base_tokenizer import Base_Tokenizer

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
        lis = [f'<s> <user_{i}>'+tree['title']+' </s> '+tree['selftext']+' </s>' ]
        authors = {tree['author']:i}        

        for id, comment in tree['comments'].items():
            
            if comment['author'] not in authors:
                i+=1
                author[comment['author']] = i
            
            if author[comment['author']]<self.config['max_labelled_users_per_tree']:
                lis.append( f'<user_{author[comment['author']]}>' + comment['body'] )
            
            else:
                lis.append(comment['body'])
        
        token_ids = jnp.asarray( self.get_token_ids(self.batch_encode_plus(lis)), dtype=jnp.int16)
        
        i=0
        tree['tokenized_inputs'] = token_ids[i]
        
        for id, comment in tree['comments'].items():
            i+=1
            comment['tokenized_inputs']=token_ids[i]
        
        return tree
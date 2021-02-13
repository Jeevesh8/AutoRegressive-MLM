from src.Tokenizers.base_tokenizer import Base_Tokenizer

class Thread_Tokenizer(Base_Tokenizer):

    def __init__(self, config):
        super().__init__(config)

    def join_tokenized(self, parts, type_ids):
        """
        Joins together tokenized and encoded parts of a comment/post.
        """
        tokenized_str = []
        tokenwise_type_ids = []

        for part, typ in zip(parts, type_ids):
            tokenized_str+=part
            tokenwise_type_ids+=[typ]*len(part)

        if len(tokenized_str)<self.config['max_length']:
            tokenized_str+=[self.config['pad_id']]*(self.config['max_length']-len(tokenized_str))
            tokenwise_type_ids+=[-1]*(self.config['max_length']-len(tokenwise_type_ids))

        tokens = jnp.asarray(tokenized_str[:self.config['max_length']], dtype=jnp.int16)
        types =  jnp.asarray(tokenwise_type_ids[:self.config['max_length']], dtype=jnp.int16)

        return tokens, types

    def tokenize_thread(self, thread):
        """
        thread:  A list, whose each element is a list of contiguous parts 
                 of a post/comment; where each part corresponds to a 
                 claim/premise or neither.
        """
        tokenized_pcs = []
        type_ids = []

        for pc in thread:
            str_lis, type_lis = pc[0], pc[1]
            str_lis[0], str_lis[-1] = '<s> '+str_lis[0], str_lis[-1]+' </s>'
            encoded_parts = self.get_token_ids(self.batch_encode_plus(str_lis))
            tokens, types = self.join_tokenized(encoded_parts, type_lis)
            tokenized_pcs.append(tokens)
            type_ids.append(types)

        return tokenized_pcs, type_ids
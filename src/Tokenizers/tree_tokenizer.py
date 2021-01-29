from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import jax.numpy as jnp


class Tree_Tokenizer:
    
    def __init__(self, config):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.dms = self.get_discourse_markers(config['discourse_markers_file'])
        self.config = config
        if 'pt_hf_tokenizer' in self.config:
            self.load_from_pretrained()
        else:
            self.trainer = BpeTrainer(special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>', '<url>']+self.dms)

    def load_from_pretrained(self):
        json_tok = self.save_and_modify()
        with open('./final_tokensier.json', 'w+') as f:
            json.dump(json_tok, f)
        self.tokenizer = Tokenizer.from_file('./final_tokensier.json')
        self.set_up_tokenizer()
    
    def save_and_modify(self):
        
        self.tokenizer.save('./empty_tokenizer.json')
        
        with open('./empty_tokenizer.json') as f:
            json_tok = json.load(f)
        
        roberta_vocab, roberta_merges = self.get_vocab_merges()
        json_tok['model']['vocab'] = roberta_vocab
        json_tok['model']['merges'] = roberta_merges
        return json_tok
    
    def get_vocab_merges(self):
        self.config['hf_pt_tokenizer'].save_vocabulary('.')

        with open('./vocab.json') as f:
            roberta_vocab = json.load(f)
        
        with open('./merges.txt') as f:
            roberta_merges = f.readlines()[1:]
            roberta_merges = [merge.rstrip('\n') for merge in roberta_merges]
        
        roberta_vocab = self.add_tokens(roberta_vocab)

        return roberta_vocab, roberta_merges
    
    def get_missing_dms(self, vocab):
        missing = []
        vocab = [key[1:] if key.startswith('Ġ') else key for key in roberta_vocab.keys()]
        for dm in dms:
            for word in dm.split():
                if word not in vocab and word not in missing:
                    missing.append(word)
        return missing
    
    def add_tokens(self, vocab):
        self.extra_tokens = ['<url>']+self.get_missing_dms(self, vocab) #Tokens to add to RoBertA vocab
        for word in self.extra_tokens:
            vocab[word] = len(vocab)
        self.config['extra_tokens'] = self.extra_tokens
        return vocab

    def train_tokenizer(self, data_files=None, binary_iterator=None, str_iter=None):
        
        if data_files is not None:
            self.tokenizer.train(self.trainer, data_files)
        
        else:
            str_iter = str_iter if str_iter is not None else self.make_str_iter(binary_iterator)
            self.tokenizer.train_from_iterator(trainer=self.trainer, iterator=str_iter)
        
        self.set_up_tokenizer()
    
    def make_str_iter(self, binary_iterator):
        def str_iter():
            for batch in binary_iterator:
                yield self.decode_to_str(batch)
        return str_iter()
    
    def set_up_tokenizer(self):
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('<pad>'),
                                      length=self.config['max_length'])
        
        self.tokenizer.enable_truncation(self.config['max_length'])

        self.tokenizer.post_processor = TemplateProcessing(single = "<s>:1 $A:1 </s>:1",
                                                           pair = "<s>:1 $A:1 </s>:1 </s>:2 $B:2 </s>:2",
                                                           special_tokens=[('<s>',1), ('</s>',2)])

    def decode_to_str(self, batch_text) :
        """
        Converts bytes string data to text. And truncates to max_len. 
        """
        max_len = self.config['max_length']
        return [ ' '.join(text.decode('utf-8').split()[:max_len] if isinstance(text, bytes) 
                                                                 else text.split()[:max_len])
             for text in batch_text ]

    def batch_encode_plus(self, batch1, batch2=None):
        """
        Two batches correspond to sequences of different type/language.
        """
        if batch2 is None :
            return self.tokenizer.encode_batch( self.decode_to_str(batch1) )
        
        else :
            lis = [ (seq1,seq2) for seq1, seq2 in zip( self.decode_to_str(batch1), self.decode_to_str(batch2) ) ]
            return self.tokenizer.encode_batch(lis)
    
    def get_token_ids(self, token_encoding):
        return [elem.ids for elem in token_encoding]
    
    def get_lang_ids(self, token_encoding):
        return [elem.type_ids for elem in token_encoding]

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
    
    def get_discourse_markers(self, filename):
        with open(filename) as f:
            dms = f.readlines()[2:]
            dms = [elem.split(' ', 1)[1].rstrip('\n') for elem in dms]
        return dms
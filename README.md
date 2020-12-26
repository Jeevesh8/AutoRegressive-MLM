# NLP-Journey

A follow up repository of [Jax-Journey](https://github.com/deterministic-algorithms-lab/Jax-Journey). This repository provides a selection of notebooks for various NLP tasks, which are completely see-through (i.e., you can see the implementation till the basic Jax/Haiku modules, in a single notebook). These were meant to be used as further tutorials in Jax for NLP, and as a guide for the coding style followed in this [awesome article](https://www.pragmatic.ml/finetuning-transformers-with-jax-and-haiku/) by Madison May. 

These notebooks, although mostly code, also mention the nuanced features, often missed when using off-the-shelf models. Moreover, they allow you to optimize everything right to the innermost modules. Also, we mention how to adapt the model to your use case, in each notebook. 

## Transformer for Sequence Classification

A basic introductory notebook consisting of the original [RoBERTa initialized version](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/main/classification/basic_transformer.ipynb) and [randomly initialized version](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/main/classification/transformer_to_pretrain.ipynb) .

## Transformers for Language Modelling Tasks

Here we realise the need for restructuring the code, and correspondingly, place all the code component-wise in ```src/``` . The new things we code over the original implementation are: 
* The masking function for MLM [here](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/main/src/Tokenizers/masking_utils.py#L6),
* A [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) based tokenizer, [here](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/main/src/Tokenizers/hf_tokenizer.py)
* A Language Embedding for TLM task, [here](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/81f7a7568db6676d561aaf7f579f8af99c99b28a/src/model/embeddings.py#L77). 
* Additionally, we include an option to make the transformer auto-regressive and add a mask for the same, [here](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/81f7a7568db6676d561aaf7f579f8af99c99b28a/src/model/transformer.py#L64). This is needed for CLM.

The final notebook can be found [here](https://github.com/deterministic-algorithms-lab/NLP-Journey/blob/main/LanguageModelling/CLM_MLM_TLM.ipynb).

## Steps

1. Pick a jsonlist object; clean data to make a text tree. [[Can use tensorflow datasets to make a dataset out of this generator]]
2. Flatten the text tree.
3. Batch text sequences to encode them all. Store them alongside the text. Store a masked copy of each text sequence too.
4. Unflatten tree. Tree now consists of nodes of form (text, encoding, masked_text). 
5. Start BFS on tree using a generator that yields (prev_embdngs, final_masked_seq). Use this to prepare a batch.
6. Send the batch to decoder-like non-autoregressive block to predict masked tokens, then backpropagate. Pick another batch from BFS generator and continue till no more batch left.  [[Won't use tensorflow datasets to make a dataset out of this generator.]]
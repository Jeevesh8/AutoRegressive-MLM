# AutoRegressive MLM

We build upone [this](https://github.com/deterministic-algorithms-lab/NLP-Journey) original repo to allow for an AutoRegressive MLM model. In social media platforms, like Reddit, we see a tree of comments forming, with the post at the root. The task now is to condition the MLM prediction of a masked tokens in any particular comment on the post and all the comments on the shortest path from the root to the comment. We try to do the same, using an encoder-decoder like architecture, while making the following modifications :

1. The decoder's self attention is not masked, so it can look both forward and backward in the current comment for predicting masked tokens.

2. The encoder encodes all comments and posts first

3. Then each of those post's representation is pooled over. to get fixed length vector for each post and comment.

4. Then we pick a particular post, we mask tokens in it, send this masked sequence to the decoder.

5. And a sequence of fixed length vectors corresponding to each parent post of the currently picked post, are concatenated and sent in the place where the decoder accepts the encoder output.

**To Try :** Adding positional encodings to the output of the encoder.  

# Detailed Explanation

See [here](https://github.com/Jeevesh8/AutoRegressive-MLM/blob/main/AutoRegressive_MLM.pdf) for a detailed explanation and model architecture.
------------------------------------------------------------------------------------------------------------------------------
# Original README.md (NLP-Journey)

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

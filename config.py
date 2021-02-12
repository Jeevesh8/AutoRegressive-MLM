config = {
          'pre_training' : True,
          'initialize_pretrained' : 'RoBERTa',

          #Data Parameters
          'max_length' : 128, 
          'featurizer_batch_size' : 4,
          'mlm_batch_size' : 4,
          'n_epochs' : 10,
          'data_files' : ['/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/train_period_data.jsonlist'],
          'discourse_markers_file' : '/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/Discourse_Markers.txt',
          
          #Model Parameters
          'intermediate_size' : 256,
          'n_heads' : 2,
          'n_layers' : 2,
          'hidden_size' : 128,
          'd_model' : 128,                                                      #same as hidden_size
          'max_losses' : 10,                                                    #max. number of losses to backpropagate at once
          'max_tree_size' : 60,
         
          #Embeddings Parameters
          'embed_dropout_rate' : 0.1,
          
          #MHA parameters
          'attention_drop_rate' : 0.1,
          
          #MLP parameters
          'fully_connected_drop_rate' : 0.1,
          
          #Training Parameters
          'learning_rate' : 1e-5,
          'max_grad_norm' : 1.0,
          'l2' : 0.1,

           #colab parameter
          'restart_from' : 0,
          }

## For Roberta
RoBERTa_config = {
          #Data Parameters
          'max_length' : 512, 
          'featurizer_batch_size' : 4,
          'mlm_batch_size' : 4,
          'n_epochs' : 10,
          'data_files' : ['/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/train_period_data.jsonlist'],
          'discourse_markers_file' : '/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/Discourse_Markers.txt',

          #Model Parameters
          'intermediate_size' : 3072,
          'n_heads' : 12,
          'n_layers' : 6,
          'hidden_size' : 768,
          'd_model' : 768,                                                      #same as hidden_size
          'max_losses' : 2,                                                     #max. number of losses to backpropagate at once
          'max_tree_size' : 20,
          
          #Embeddings Parameters
          'embed_dropout_rate' : 0.1,
          
          #MHA parameters
          'attention_drop_rate' : 0.1,
          
          #MLP parameters
          'fully_connected_drop_rate' : 0.1,
          
          #Training Parameters
          'learning_rate' : 1e-5,
          'max_grad_norm' : 1.0,
          'l2' : 0.1,
          
          #colab parameter
          'restart_from' : 0,
          }

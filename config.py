# For Pre-Training From Scratch
config = {
          'pre_training' : True,
          'initialize_pretrained' : 'distilroberta-base',

          #Data Parameters
          'featurizer_max_length' : 128,
          'max_length' : 512,
          'featurizer_batch_size' : 4,
          'mlm_batch_size' : 4,
          'n_epochs' : 10,
          
          #Folders containing train_period_data.jsonlist and heldout_period_data.jsonlist
          'data_folders' : ['/content/drive/MyDrive/Argumentation/first_batch_data/'],
          'discourse_markers_file' : '/content/drive/MyDrive/Argumentation/first_batch_data/Discourse_Markers.txt',
          'params_dir' : '/content/drive/MyDrive/Argumentation/',          #Directory to read from/write to params

          #Model Parameters
          'intermediate_size' : 768*4,
          'n_heads' : 12,
          'n_layers' : 6,
          'hidden_size' : 768,
          'd_model' : 768,                                                      #same as hidden_size
          'max_losses' : 10,                                                    #max. number of losses to backpropagate at once
          'max_tree_size' : 4*5,
          'max_labelled_users_per_tree':10,
          'last_layer' : '',                                                    #Specify(Linear/GRU) when pre_training=False.

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
          'restart_from' : 55000,
          }

## For pre-training from RoBERTa weights
'''
config = {
          'pre_training' : True,
          'initialize_pretrained' : 'distilroberta-base',

          #Data Parameters
          'max_length' : 512, 
          'featurizer_batch_size' : 4,
          'mlm_batch_size' : 4,
          'n_epochs' : 10,
          
          #Folders containing train_period_data.jsonlist and heldout_period_data.jsonlist
          'data_folders' : ['/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/'],
          'discourse_markers_file' : '/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/Discourse_Markers.txt',
          'params_dir' : '/content/drive/MyDrive/2SCL/Argumentation/',          #Directory to read from/write to params 

          #Model Parameters
          'intermediate_size' : 3072,
          'n_heads' : 12,
          'n_layers' : 6,
          'hidden_size' : 768,
          'd_model' : 768,                                                      #same as hidden_size
          'max_losses' : 2,                                                     #max. number of losses to backpropagate at once
          'max_tree_size' : 20,
          'max_labelled_users_per_tree':8,
          'last_layer' : '',                                                    #Specify(Linear/GRU) when pre_training=False.

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
'''

## For FineTuning
'''
config = {
          'pre_training' : False,
          'initialize_pretrained' : <path-to-pretrained-wts-file>,

          #Data Parameters
          'max_length' : 512, 
          'featurizer_batch_size' : 4,
          'mlm_batch_size' : 4,
          'n_epochs' : 10,
          
          ## Folders containing fine-tuning data. Each of these must have train/, test/, valid/, subfolders.
          'ft_data_folders' : [ '/content/drive/MyDrive/2SCL/Argumentation/finetune-data/change-my-view-modes/v2.0/negative/', 
                                '/content/drive/MyDrive/2SCL/Argumentation/finetune-data/change-my-view-modes/v2.0/positive/'],

          'discourse_markers_file' : '/content/drive/MyDrive/2SCL/Argumentation/first_batch_data/Discourse_Markers.txt',
          'params_dir' : '/content/drive/MyDrive/2SCL/Argumentation/',          #Directory to read from/write to params 

          #Model Parameters
          'intermediate_size' : 3072,
          'n_heads' : 12,
          'n_layers' : 6,
          'hidden_size' : 768,
          'd_model' : 768,                                                      #same as hidden_size
          'max_losses' : 2,                                                     #max. number of losses to backpropagate at once
          'max_tree_size' : 20,
          'max_labelled_users_per_tree':8,
          'n_classes' : 3,                                                      #Number of classes for argument classification.
          'last_layer' : 'GRU',                                                 #Specify(Linear/GRU) when pre_training=False.

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
'''

import haiku as hk
from haiku.data_structures import to_immutable_dict, to_mutable_dict

def print_keys(params, n=0):
    for key in params:
        print('\t'*n+key)
        try :
            print_keys(params[key], n+1)
        except:
            pass

def change_keys(dic, key, replace_with):
    dic1 = {}
    for k in dic.keys():
        dic1[k.replace(key, replace_with, 1)] = dic[k]
    return dic1

def logits_to_ar_classifier_params(pretrained_params, classifier_params):
    
    pretrained_params = to_mutable_dict(pretrained_params)
    pretrained_params['ar_classifier'] = pretrained_params['mlm_predictor']
    pretrained_params.pop('mlm_predictor')
    pretrained_params['ar_classifier'] = change_keys(pretrained_params['ar_classifier'], 'extended_encoder', 'auto_regressive_classifier')
    pretrained_params['ar_classifier'] = change_keys(pretrained_params['ar_classifier'], 'auto_regressive_classifier/~/', 'auto_regressive_classifier/')
    #pretrained_params['ar_classifier']['extended_encoder/linear'] = classifier_params['ar_classifier']['auto_regressive_classifier/linear']
    pretrained_params['ar_classifier']['auto_regressive_classifier/~/gru/~/gru'] = classifier_params['ar_classifier']['auto_regressive_classifier/~/gru/~/gru']
    return to_immutable_dict(pretrained_params)

class Scope(object):
    """
    A tiny utility to help make looking up into our dictionary cleaner.
    There's no haiku magic here.
    """
    def __init__(self, weights, prefix):
        self.weights = weights
        self.prefix = prefix

    def __getitem__(self, key):
        return hk.initializers.Constant(self.weights[self.prefix + key]) if self.weights is not None else None
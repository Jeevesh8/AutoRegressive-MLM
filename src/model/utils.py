import haiku as hk
from haiku.data_structures import to_immutable_dict, to_mutable_dict

def logits_to_ar_classifier_params(pretrained_params, classifier_params):
    
    pretrained_params = to_mutable_dict(pretrained_params)
    pretrained_params['ar_classifier'] = pretrained_params['mlm_predictor']
    pretrained_params.pop('mlm_predictor')
    pretrained_params['ar_classifier']['extended_encoder/linear'] = classifier_params['ar_classifier']['extended_encoder/linear']

    return to_immutable_dict(pretrained_params)
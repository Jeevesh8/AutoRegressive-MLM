import gin

from src.model.transformer import *
from src.model.embeddings import *

gin.bind_parameter('TransformerFeaturizer.name', 'encoder')
gin.bind_parameter('TransformerBlock.name', 'layer')
gin.bind_parameter('MultiHeadAttention.name', 'attention')
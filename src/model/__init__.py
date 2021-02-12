import gin

from src.model.transformer import *
from src.model.embeddings import *
from src.model.utils import *

gin.bind_parameter('TransformerFeaturizer.name', 'encoder')
gin.bind_parameter('TransformerBlock.name', 'layer')
gin.bind_parameter('TransformerDecoderBlock.name', 'layer')
gin.bind_parameter('MultiHeadAttention.name', 'attention')
gin.bind_parameter('ExtendedEncoder.name', 'encoder')
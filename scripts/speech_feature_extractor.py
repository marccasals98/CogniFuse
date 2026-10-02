import logging
import torchaudio
import torch
from torch import nn

# ---------------------------------------------------------------------
#region Logging

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion
# ---------------------------------------------------------------------

class SpeechFeatureExtractor(nn.Module):

    def __init__(self, input_parameters):
        super().__init__()

        self.speech_feature_extractor_type = input_parameters.speech_feature_extractor
        self.init_feature_extractor()
        self.init_layers_weights()


    def init_feature_extractor(self):

        if self.speech_feature_extractor_type == "WAVLM_BASE":
            bundle = torchaudio.pipelines.WAVLM_BASE
            self.num_layers = 12 # Layers of the Transformer of the WavLM model
            # every layer has features with 768 dimension
        elif self.speech_feature_extractor_type == "WAVLM_BASE_PLUS":
            bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
            self.num_layers = 12 # Layers of the Transformer of the WavLM model
            # every layer has features with 768 dimension
        elif self.speech_feature_extractor_type == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.speech_feature_extractor_type == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.speech_feature_extractor_type == "WAV2VEC2_XLSR_300M":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        elif self.speech_feature_extractor_type == "WAV2VEC2_XLSR_1B":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
            self.num_layers = 48 # Layers of the Transformer of the WavLM model
            # every layer has features with 1280 dimension
        elif self.speech_feature_extractor_type == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
            self.num_layers = 24 # Layers of the Transformer of the WavLM model
            # every layer has features with 1024 dimension
        else:
            raise Exception('No speech_feature_extractor choice found.') 

        self.speech_feature_extractor = bundle.get_model()

    
    def init_layers_weights(self):

        # set weights for every layer as learnable parameters
        self.layer_weights = nn.Parameter(nn.functional.softmax((torch.ones(self.num_layers) / self.num_layers), dim=-1))
        

    def extract_features(self, waveform):
            
        features, _ = self.speech_feature_extractor.extract_features(waveform)
        # level_features dims: (#B, #num_vectors, #dim_vectors = )
        
        hidden_states = torch.stack(features, dim=1)
        averaged_hidden_states = (hidden_states * self.layer_weights.view(-1, 1, 1)).sum(dim=1)

        # TODO set a parameter as an option to choose between weighted layers or the last one
        # HACK to get only the last layer
        #averaged_hidden_states = features[-1]

        return averaged_hidden_states


    def __call__(self, waveform):

        logger.debug(f"waveform.size(): {waveform.size()}")

        features = self.extract_features(waveform)
        logger.debug(f"features.size(): {features.size()}")

        return features
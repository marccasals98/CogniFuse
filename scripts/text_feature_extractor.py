import logging
import torchaudio
#from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
import torch
from torch import nn
import os

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
    

class ASRModel(nn.Module):

    # https://pytorch.org/audio/main/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle

    def __init__(self):
        super().__init__()

        self.init_transcriptor()

    
    def init_transcriptor(self):

        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.sample_rate = bundle.sample_rate
        logger.info(f"ASRModel sample rate: {self.sample_rate}")


    def transcript(self, waveform):

        with torch.no_grad():

            # Produce mel-scale spectrogram features.
            logger.debug(f"waveform.size(): {waveform.size()}")

            # HACK we need to squeeze the waveform because it is needed that way: 
            # https://pytorch.org/audio/main/generated/torchaudio.pipelines.RNNTBundle.html#torchaudio.pipelines.RNNTBundle
            # Search for the line 'length = torch.tensor([features.shape[0]])' in https://pytorch.org/audio/0.11.0/_modules/torchaudio/pipelines/rnnt_pipeline.html 
            features, length = self.feature_extractor(waveform.squeeze())

            # Generate top-10 hypotheses.
            hypotheses = self.decoder(features, length, 10)

        # For top hypothesis, convert predicted tokens to text.
        transcription = self.token_processor(hypotheses[0][0])

        return transcription
    

class TextFeatureExtractor(nn.Module):

    def __init__(self, input_parameters):
        super().__init__()

        self.text_feature_extractor_type = input_parameters.text_feature_extractor
        self.init_extractor()
    
    
    def init_extractor(self):

        if self.text_feature_extractor_type == "BERT_BASE_UNCASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            # model outputs features with 768 dimension
        elif self.text_feature_extractor_type == "BERT_BASE_CASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
            # model outputs features with 768 dimension
        elif self.text_feature_extractor_type == "BERT_LARGE_UNCASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-uncased')
            # model outputs features features with 1024 dimension
        elif self.text_feature_extractor_type == "BERT_LARGE_CASED":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-cased')
            # model outputs features features with 1024 dimension
        elif self.text_feature_extractor_type == "ROBERTA_LARGE":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'roberta-large')
            # model outputs features features with 1024 dimension
        elif self.text_feature_extractor_type == "MODERN_BERT_BASE":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'answerdotai/ModernBERT-base')
            # model outputs features with 768 dimension
        elif self.text_feature_extractor_type == "MODERN_BERT_LARGE":
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'answerdotai/ModernBERT-large', reference_compile=False)
            # model outputs features with 1024 dimension
        else:
            raise Exception('No text_feature_extractor choice found.')


    def extract_features(self, transcription_tokens_padded, transcription_tokens_mask):
  
        if not (transcription_tokens_padded.is_cuda and transcription_tokens_mask.is_cuda):
            logger.warning(f"transcription_tokens_padded.is_cuda: {transcription_tokens_padded.is_cuda}, transcription_tokens_mask.is_cuda: {transcription_tokens_mask.is_cuda}")

        with torch.no_grad():

            output = self.model(transcription_tokens_padded, transcription_tokens_mask)
            
            # if we want bert's pooled vector: features = output.pooler_output
            
            # TODO set a parameter as an option to choose between weighted layers or the last one
            # we obtain the last layer features
            # features dims: (#B, #num_vectors, #dim_vectors = 768)
            features = output.last_hidden_state

            logger.debug(f"features.size(): {features.size()}")

        return features

    
    def __call__(self, transcription_tokens_padded, transcription_tokens_mask):

        features = self.extract_features(transcription_tokens_padded, transcription_tokens_mask)

        return features
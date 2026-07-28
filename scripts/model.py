import logging
from torch import nn
import torch
import numpy as np
from speech_feature_extractor import SpeechFeatureExtractor
from text_feature_extractor import TextFeatureExtractor
from adapter import NoneAdapter, LinearAdapter, NonLinearAdapter
from poolings import CrossAttentionReduced, NoneSeqToSeq, SelfAttention, MultiHeadAttention, TransformerStacked, ReducedMultiHeadAttention, CrossAttention
from poolings import StatisticalPooling, AttentionPooling
from classifier_layer import ClassifierLayer

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


class Classifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
     
        self.device = device
        self.init_speech_feature_extractor(parameters)
        self.init_text_feature_extractor(parameters) 
        self.init_adapter_layers(parameters)
        self.init_pooling_component(parameters)
        self.init_classifier_layer(parameters)
    

    def init_speech_feature_extractor(self, parameters):

        self.speech_feature_extractor = SpeechFeatureExtractor(parameters)
        
        for name, parameter in self.speech_feature_extractor.named_parameters():
            
            # Unncomment to freeze all speech_feature_extractor parameters
            #if True:
            #    logger.info(f"Setting {name} to requires_grad = False")
            #    parameter.requires_grad = False
            
            # Unncomment to freeze all speech_feature_extractor parameters, 
            # except layers weights and the last layer
            #if name != "layer_weights" and "transformer.layers.11" not in name:
                #logger.debug(f"Setting {name} to requires_grad = False")
                #parameter.requires_grad = False
                
            # Unncomment to freeze all speech_feature_extractor parameters, 
            # except layers weights
            if name != "layer_weights":
                logger.debug(f"Setting {name} to requires_grad = False")
                parameter.requires_grad = False

            # Unncomment to only freeze the encoder
            #if (name != "layer_weights") and ("transformer.layers." not in name):
            #    logger.info(f"Setting {name} to requires_grad = False")
            #    parameter.requires_grad = False

        self.speech_feature_extractor_norm_layer = nn.LayerNorm(parameters.speech_feature_extractor_output_vectors_dimension)

    
    def init_text_feature_extractor(self, parameters):

        self.text_feature_extractor = TextFeatureExtractor(parameters)

        for name, parameter in self.text_feature_extractor.named_parameters():
            
            # TODO allow to train some parameters
            # TODO only freeze the encoder
            
            # Freeze all BERT parameters except layers weights and the last layer
            #if name != "layer_weights" and "transformer.layers.11" not in name:
            # Freeze all BERT parameters except layers weights
            #if "encoder.layer.11" not in name:
            if True:
                logger.debug(f"Setting {name} to requires_grad = False")
                parameter.requires_grad = False
        
        self.text_feature_extractor_norm_layer = nn.LayerNorm(parameters.text_feature_extractor_output_vectors_dimension)
        #self.text_feature_extractor_norm_layer = nn.LayerNorm(1024)
        

    def init_speech_adapter_layer(self, parameters):

        if parameters.speech_adapter == 'NoneAdapter':
            self.speech_adapter_layer = NoneAdapter()
            self.speech_adapter_output_vectors_dimension = parameters.speech_feature_extractor_output_vectors_dimension
        elif parameters.speech_adapter == 'LinearAdapter':
            self.speech_adapter_layer = LinearAdapter(parameters.speech_feature_extractor_output_vectors_dimension, parameters.speech_adapter_output_vectors_dimension)
            self.speech_adapter_output_vectors_dimension = parameters.speech_adapter_output_vectors_dimension
        elif parameters.speech_adapter == 'NonLinearAdapter':
            self.speech_adapter_layer = NonLinearAdapter(parameters.speech_feature_extractor_output_vectors_dimension, parameters.speech_adapter_output_vectors_dimension)
            self.speech_adapter_output_vectors_dimension = parameters.speech_adapter_output_vectors_dimension
        else:
            raise Exception('No Adapter choice found.') 
        

    def init_text_adapter_layer(self, parameters):

        if parameters.text_adapter == 'NoneAdapter':
            self.text_adapter_layer = NoneAdapter()
            self.text_adapter_output_vectors_dimension = parameters.text_feature_extractor_output_vectors_dimension
        elif parameters.text_adapter == 'LinearAdapter':
            self.text_adapter_layer = LinearAdapter(parameters.text_feature_extractor_output_vectors_dimension, parameters.text_adapter_output_vectors_dimension)
            self.text_adapter_output_vectors_dimension = parameters.text_adapter_output_vectors_dimension
        elif parameters.text_adapter == 'NonLinearAdapter':
            self.text_adapter_layer = NonLinearAdapter(parameters.text_feature_extractor_output_vectors_dimension, parameters.text_adapter_output_vectors_dimension)
            self.text_adapter_output_vectors_dimension = parameters.text_adapter_output_vectors_dimension
        else:
            raise Exception('No Adapter choice found.') 

    
    def init_adapter_layers(self, parameters):
        
        self.init_speech_adapter_layer(parameters)
        self.init_text_adapter_layer(parameters)

        assert self.speech_adapter_output_vectors_dimension == self.text_adapter_output_vectors_dimension, f"speech_adapter_output_vectors_dimension ({self.speech_adapter_output_vectors_dimension}) must be equal to text_adapter_output_vectors_dimension ({self.text_adapter_output_vectors_dimension})"

    
    def init_seq_to_seq_layer(self, parameters):
        
            self.seq_to_seq_method = parameters.seq_to_seq_method
            
            # self.speech_adapter_output_vectors_dimension is equal to self.text_adapter_output_vectors_dimension
            self.seq_to_seq_input_vectors_dimension = self.speech_adapter_output_vectors_dimension

            self.seq_to_seq_input_dropout = nn.Dropout(parameters.seq_to_seq_input_dropout)

            # HACK ReducedMultiHeadAttention seq to seq input and output dimensions don't match
            if self.seq_to_seq_method == 'ReducedMultiHeadAttention':
                self.seq_to_seq_output_vectors_dimension = self.seq_to_seq_input_vectors_dimension // parameters.seq_to_seq_heads_number
            else:
                self.seq_to_seq_output_vectors_dimension = self.seq_to_seq_input_vectors_dimension

            if self.seq_to_seq_method == 'NoneSeqToSeq':
                self.seq_to_seq_layer = NoneSeqToSeq()
            
            elif self.seq_to_seq_method == 'SelfAttention':
                self.seq_to_seq_layer = SelfAttention()

            elif self.seq_to_seq_method == 'MultiHeadAttention':
                self.seq_to_seq_layer = MultiHeadAttention(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    heads = parameters.seq_to_seq_heads_number,
                    skip_connections = parameters.skip_connections,
                )

            elif self.seq_to_seq_method == 'TransformerStacked':
                self.seq_to_seq_layer = TransformerStacked(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    n_blocks = parameters.transformer_n_blocks,
                    expansion_coef = parameters.transformer_expansion_coef,
                    drop_out_p = parameters.transformer_drop_out,
                    heads = parameters.seq_to_seq_heads_number,
                )
            
            elif self.seq_to_seq_method == 'ReducedMultiHeadAttention':
                self.seq_to_seq_layer = ReducedMultiHeadAttention(
                    encoder_size = self.seq_to_seq_input_vectors_dimension,
                    heads_number = parameters.seq_to_seq_heads_number,
                )
            elif self.seq_to_seq_method == 'CrossAttention':
                self.seq_to_seq_layer = CrossAttention(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    heads = parameters.seq_to_seq_heads_number,
                    skip_connections = parameters.skip_connections,
                )
            elif self.seq_to_seq_method == 'CrossAttentionReduced':
                self.seq_to_seq_layer = CrossAttentionReduced(
                    emb_in = self.seq_to_seq_input_vectors_dimension,
                    heads = parameters.seq_to_seq_heads_number,
                    skip_connections = parameters.skip_connections,
                )


            else:
                raise Exception('No Seq to Seq choice found.')  
    

    def init_seq_to_one_layer(self, parameters):

            self.seq_to_one_method = parameters.seq_to_one_method
            self.seq_to_one_input_vectors_dimension = self.seq_to_seq_output_vectors_dimension
            self.seq_to_one_output_vectors_dimension = self.seq_to_one_input_vectors_dimension

            self.seq_to_one_input_dropout = nn.Dropout(parameters.seq_to_one_input_dropout)

            if self.seq_to_one_method == 'StatisticalPooling':
                self.seq_to_one_layer = StatisticalPooling(
                        emb_in = self.seq_to_one_input_vectors_dimension,
                    )

            elif self.seq_to_one_method == 'AttentionPooling':
                self.seq_to_one_layer = AttentionPooling(
                    emb_in = self.seq_to_one_input_vectors_dimension,
                )
            
            else:
                raise Exception('No Seq to One choice found.') 
            
    
    def init_pooling_component(self, parameters):    

        # Set the pooling component that will take the extracted features and summarize them in a context vector
        # This component applies first a sequence to sequence layer and then a sequence to one layer.

        self.init_seq_to_seq_layer(parameters)
        self.init_seq_to_one_layer(parameters)
    

    def init_classifier_layer(self, parameters):

        self.classifier_layer_input_vectors_dimension = self.seq_to_one_output_vectors_dimension
        logger.debug(f"self.classifier_layer_input_vectors_dimension: {self.classifier_layer_input_vectors_dimension}")
        
        self.classifier_layer = ClassifierLayer(parameters, self.classifier_layer_input_vectors_dimension)


    def forward(self, input_tensor, transcription_tokens_padded = None, transcription_tokens_mask = None):
        """
        Set the net's forward pass (mandatory torch method)
        """
        
        logger.debug(f"input_tensor.size(): {input_tensor.size()}")

        # Text-based components
        text_feature_extractor_output = self.text_feature_extractor(transcription_tokens_padded, transcription_tokens_mask)
        text_feature_extractor_output = self.text_feature_extractor_norm_layer(text_feature_extractor_output)
        logger.debug(f"text_feature_extractor_output.size(): {text_feature_extractor_output.size()}")

        # Speech-based components
        speech_feature_extractor_output = self.speech_feature_extractor(input_tensor)
        speech_feature_extractor_output = self.speech_feature_extractor_norm_layer(speech_feature_extractor_output)
        logger.debug(f"speech_feature_extractor_output.size(): {speech_feature_extractor_output.size()}")

        speech_adapter_output = self.speech_adapter_layer(speech_feature_extractor_output)
        logger.debug(f"speech_adapter_output.size(): {speech_adapter_output.size()}")
        speech_adapter_output = self.seq_to_seq_input_dropout(speech_adapter_output)

        text_adapter_output = self.text_adapter_layer(text_feature_extractor_output)
        logger.debug(f"text_adapter_output.size(): {text_adapter_output.size()}")
        text_adapter_output = self.seq_to_seq_input_dropout(text_adapter_output)

        # All speech and text features goes into the same seq_to_seq component
        seq_to_seq_output = self.seq_to_seq_layer(speech_adapter_output, text_adapter_output)
        seq_to_seq_output = self.seq_to_one_input_dropout(seq_to_seq_output)
        
        seq_to_one_output = self.seq_to_one_layer(seq_to_seq_output)
        logger.debug(f"seq_to_one_output.size(): {seq_to_one_output.size()}")

        # classifier_output are logits, softmax will be applied within the loss
        classifier_input = seq_to_one_output
        logger.debug(f"classifier_input.size(): {classifier_input.size()}")

        classifier_output = self.classifier_layer(classifier_input)
        logger.debug(f"classifier_output.size(): {classifier_output.size()}")
    
        return classifier_output

    
    def predict(self, input_tensor, transcription_tokens_padded = None, transcription_tokens_mask = None, thresholds_per_class = None):

        # HACK awfull hack, we are going to assume that we are going to predict over single tensors (no batches)
        
        predicted_logits = self.forward(input_tensor, transcription_tokens_padded, transcription_tokens_mask)
        predicted_probas = torch.nn.functional.log_softmax(predicted_logits, dim = 1)
        predicted_probas = predicted_probas.squeeze().to("cpu").numpy()
        logger.debug(f"predicted_probas: {predicted_probas}")

        if thresholds_per_class is not None:
            logger.debug("Entered threshold_per_class")
            max_proba_class = np.argmax(predicted_probas)
            logger.debug(f"max_proba_class: {max_proba_class}")
            threshold_check = predicted_probas[max_proba_class] >= thresholds_per_class[max_proba_class]
            logger.debug(f"threshold_check: {threshold_check}, {predicted_probas[max_proba_class]}, {thresholds_per_class[max_proba_class]}")

            if threshold_check == True:
                logger.debug("Entered threshold_check")
                predicted_class = max_proba_class
            else:
                logger.debug("Entered filtered_probas")
                filtered_probas = predicted_probas.copy()
                filtered_probas[max_proba_class] = -np.inf
                logger.debug(f"filtered_probas: {filtered_probas}")
                predicted_class = np.argmax(filtered_probas)
        else:
            logger.debug("Entered normal prediction")
            predicted_class = np.argmax(predicted_probas)

        return torch.tensor([predicted_class]).int()







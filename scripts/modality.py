import logging
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

#region Modality ablation

MODALITIES = ['both', 'speech', 'text']

# These methods attend from one modality to the other, so they need both.
CROSS_MODAL_SEQ_TO_SEQ_METHODS = ['CrossAttention', 'CrossAttentionReduced']


def check_modality(modality, seq_to_seq_method, model_class = 'MireiaClassifier'):
    """Raise a clear error if the modality / seq_to_seq_method / model_class combination is not supported."""
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got '{modality}'.")
    if modality != 'both' and model_class != 'MireiaClassifier':
        raise ValueError(f"--modality {modality} is only implemented in MireiaClassifier. Add --model_class MireiaClassifier.")
    if modality != 'both' and seq_to_seq_method in CROSS_MODAL_SEQ_TO_SEQ_METHODS:
        raise ValueError(
            f"--modality {modality} is not compatible with --seq_to_seq_method {seq_to_seq_method}, "
            f"which needs both speech and text. Use one of the concatenation methods instead."
        )


def drop_sequence(features, mask=None):
    """Turn a [batch, tokens, dim] sequence (and its [batch, tokens] mask) into a zero-length one."""
    return features[:, :0], None if mask is None else mask[:, :0]


class ModalitySelector(nn.Module):

    """
    Keeps only the selected modality before the seq_to_seq component.
    The discarded modality becomes a zero-length sequence, so the concatenation
    done by the seq_to_seq methods and sequence_mask() works without changes.
    modality = 'both' returns the inputs untouched (default behaviour).
    """

    def __init__(self, modality = 'both', seq_to_seq_method = None):

        super().__init__()

        check_modality(modality, seq_to_seq_method)
        self.modality = modality
        if self.modality != 'both':
            logger.info(f"Modality ablation: only '{self.modality}' features reach the seq_to_seq component.")


    def forward(self, speech, text, speech_mask = None, text_mask = None):

        if self.modality == 'speech':
            text, text_mask = drop_sequence(text, text_mask)
        elif self.modality == 'text':
            speech, speech_mask = drop_sequence(speech, speech_mask)

        return speech, text, speech_mask, text_mask

#endregion

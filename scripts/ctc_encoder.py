"""Pretrained Spanish CTC emissions; forced alignment is a separate next step."""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from utils.audio_frame_timing import frame_timing
from utils.transcript_normalization import normalize_ctc_words


DEFAULT_CTC_MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish'


class SpanishCTCEncoder(torch.nn.Module):
    """Keep the trained encoder/head pair together; never use baseline features.

    Input is one clean, mono waveform at 16 kHz. The internal encoder produces
    [1, T, D]; its trained linear head produces [1, T, V]. This module is an
    inference-only alignment component, independent of the clinical classifier.
    """

    def __init__(self, processor, model, model_id=DEFAULT_CTC_MODEL):
        super().__init__()
        self.processor, self.model, self.model_id = processor, model, model_id
        self.vocabulary = processor.tokenizer.get_vocab()
        self.blank_id = model.config.pad_token_id
        self.delimiter = processor.tokenizer.word_delimiter_token
        if (self.blank_id != processor.tokenizer.pad_token_id
                or self.blank_id not in self.vocabulary.values()):
            raise ValueError('Model and tokenizer disagree on the CTC blank')
        if set(self.vocabulary.values()) != set(range(model.lm_head.out_features)):
            raise ValueError('CTC head and tokenizer vocabulary sizes disagree')
        if processor.feature_extractor.sampling_rate != 16000:
            raise ValueError('Expected a 16-kHz CTC model')
        if getattr(model.config, 'add_adapter', False):
            raise ValueError('Temporal adapters need a separate timing implementation')
        self.model.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_CTC_MODEL, *, device='cpu',
                        local_files_only=False, revision=None):
        options = {'local_files_only': local_files_only}
        if revision is not None:
            options['revision'] = revision
        model, loading = Wav2Vec2ForCTC.from_pretrained(
            model_id, output_loading_info=True, **options,
        )
        # Loading an encoder-only checkpoint must never silently create a head.
        missing = [key for key in loading.get('missing_keys', [])
                   if not key.endswith('masked_spec_embed')]
        if missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError(f'Checkpoint does not contain a complete trained CTC model: {loading}')
        resolved = getattr(model.config, '_commit_hash', None)
        if resolved:
            options['revision'] = resolved
        processor = Wav2Vec2Processor.from_pretrained(model_id, **options)
        return cls(processor, model, model_id).to(device)

    def normalize_words(self, words):
        if getattr(self, 'skip_unsupported_words', False):
            from utils.ctc_skipped_words import normalize_skipping_unsupported
            return normalize_skipping_unsupported(words, self.vocabulary, self.blank_id, self.delimiter)
        return normalize_ctc_words(words, self.vocabulary, self.blank_id, self.delimiter)

    @torch.inference_mode()
    def forward(self, waveform, sample_rate=16000):
        if sample_rate != 16000 or waveform.ndim != 1:
            raise ValueError('CTC input must be one mono waveform sampled at 16 kHz')
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError('CTC input must be nonempty and finite')
        expected = waveform.numel()
        for kernel, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            expected = (expected - kernel) // stride + 1
        timing = frame_timing(self.model.config, waveform.numel(), sample_rate, expected)
        self.eval()
        device = next(self.model.parameters()).device
        inputs = self.processor(waveform.detach().cpu().numpy(), sampling_rate=sample_rate,
                                padding=False, return_tensors='pt').to(device)
        if inputs['input_values'].shape != (1, waveform.numel()):
            raise ValueError('CTC processor changed the unpadded waveform length')
        if 'attention_mask' in inputs and not inputs['attention_mask'].bool().all():
            raise ValueError('Unexpected padded CTC input')
        hidden = self.model.wav2vec2(**inputs).last_hidden_state
        logits = self.model.lm_head(self.model.dropout(hidden))
        if logits.shape != (1, expected, len(self.vocabulary)) or not torch.isfinite(logits).all():
            raise ValueError('Invalid CTC emission shape or nonfinite values')
        return {
            'ctc_logits': logits.cpu(),
            'ctc_log_probs': logits.float().log_softmax(dim=-1).cpu(),
            'valid_frame_mask': torch.ones(1, expected, dtype=torch.bool),
            'timing': timing,
        }

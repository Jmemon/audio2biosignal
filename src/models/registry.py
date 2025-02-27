from typing import Dict
from src.models.base import BaseEncoder, BaseDecoder, BaseAudio2EDA
from src.models.encoders.wavenet import WavenetEncoder
from src.models.encoders.transformer import TransformerEncoder
from src.models.decoders.cnn import CNNDecoder
from src.models.decoders.lstm import LSTMDecoder
from src.models.decoders.transformer import TransformerDecoder
from src.configs import ModelConfig

class ModelRegistry:
    def get_encoder(self, encoder_config: Dict) -> BaseEncoder:
        encoder_type = encoder_config.get('type')
        if encoder_type == 'wavenet':
            return WavenetEncoder(encoder_config)
        elif encoder_type == 'transformer':
            return TransformerEncoder(encoder_config)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def get_decoder(self, decoder_config: Dict) -> BaseDecoder:
        decoder_type = decoder_config.get('type')
        if decoder_type == 'cnn':
            return CNNDecoder(decoder_config)
        elif decoder_type == 'lstm':
            return LSTMDecoder(decoder_config)
        elif decoder_type == 'transformer':
            return TransformerDecoder(decoder_config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def get_model(self, model_config: ModelConfig) -> BaseAudio2EDA:
        encoder = self.get_encoder(model_config.encoder_params)
        decoder = self.get_decoder(model_config.decoder_params)
        return BaseAudio2EDA(encoder, decoder)

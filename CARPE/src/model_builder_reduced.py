"""
Module that builds pytorch models from configuration files. 
"""
import math
import sys

#import ipdb
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

#sys.path.append('perceiver-pytorch')
#from perceiver_pytorch import Perceiver as PyPerceiver

#from LayerNormLSTM import LayerNormLSTM
#from reformer_pytorch import ReformerLM
#from models.xresnet1d import xresnet1d101
from padding_helpers import get_same_padding

from collections import OrderedDict

from IPython import embed

class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.resweight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x1, x2):
        return x1 + self.resweight * x2

class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You
    Need".  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention
    is all you need. In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement in a different way during
    application.
    This class is adapted from the pytorch source code.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
            (default=2048).
        dropout: the dropout value (default=0.1).
        norm: Normalization to apply, one of 'layer' or 'rezero'.
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 norm='layer'):
        super(TransformerEncoderLayer, self).__init__()
        if norm == 'layer':
            def get_residual():
                def residual(x1, x2):
                    return x1 + x2
                return residual

            def get_norm():
                return nn.LayerNorm(d_model)
        elif norm == 'rezero':
            def get_residual():
                return ReZero()

            def get_norm():
                return nn.Identity()
        else:
            raise ValueError('Invalid normalization: {}'.format(norm))

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = get_norm()
        self.norm2 = get_norm()
        self.residual1 = get_residual()
        self.residual2 = get_residual()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = self.residual1(src, self.dropout1(src2))
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.residual2(src, self.dropout2(src2))
        src = self.norm2(src)
        return src


class PositionalEncoding():
    """Apply positional encoding to instances."""

    def __init__(self, min_timescale, max_timescale, n_channels,
                 positions_key='times'):
        """PositionalEncoding.
        Args:
            min_timescale: minimal scale of values
            max_timescale: maximal scale of values
            n_channels: number of channels to use to encode position
        """
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.n_channels = n_channels
        self.positions_key = positions_key

        self._num_timescales = self.n_channels // 2
        self._inv_timescales = self._compute_inv_timescales()

    def _compute_inv_timescales(self):
        log_timescale_increment = (
            math.log(float(self.max_timescale) / float(self.min_timescale))
            / (float(self._num_timescales) - 1)
        )
        inv_timescales = (
            self.min_timescale
            * np.exp(
                np.arange(self._num_timescales)
                * -log_timescale_increment
            )
        )
        return inv_timescales

    def __call__(self, instance):
        """Apply positional encoding to instances."""
        # instance = instance.copy()  # We only want a shallow copy
        positions = instance[self.positions_key]
        scaled_time = (
            positions[:, np.newaxis] *
            self._inv_timescales[np.newaxis, :]
        )
        signal = np.concatenate(
            (np.sin(scaled_time), np.cos(scaled_time)),
            axis=1
        )
        positional_encoding = np.reshape(signal, (-1, self.n_channels))
        instance[self.positions_key+'_embedded'] = positional_encoding
        return instance

class AttentionModel(nn.Module):
    """Sequence to sequence model based on MultiHeadAttention."""

    def __init__(self, config, d_model=128, n_layers=4, n_heads=4, dropout=0.3,
                 norm='layer', indicators=False,
                 **kwargs):
        """AttentionModel.
        Args:
            d_model: Dimensionality of the model
            n_layers: Number of MultiHeadAttention layers
            n_heads: Number of attention heads
            indicators: flag if missingness indicators should be applied
        """
        super().__init__(**kwargs)
        for k, v in config.items():
            setattr(self, k, v)
        ff_dim = 4*self.d_model # dimensionality of ff layers: hard-coded default
        self.penultimate_input_dim = self.d_model

        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels, self.d_model)]
            + [
                TransformerEncoderLayer(
                    self.d_model, self.n_heads, ff_dim,
                    self.dropout, norm=self.norm)
                for n in range(self.n_layers)
            ]
            + [nn.Linear(self.d_model, self.d_model)]
        )

    @property
    def transforms(self):
        parent_transforms = super().transforms
        parent_transforms.extend([
            PositionalEncoding(1, 500, 10),  # apply positional encoding
            self.to_observation_tuples            # mask nan with zero add indicator
        ])
        return parent_transforms

    def forward(self, x):
        """Apply attention model to input x."""
        # We work with input dims: B x C x L
        offset = 0
        # Invert mask as multi head attention ignores values which are true
        mask = None
        future_mask = None
        x = self.layers[0](x.permute(0, 2, 1))

        x = x.permute(1, 0, 2)
        for layer in self.layers[1:]:
            if isinstance(layer, TransformerEncoderLayer):
                x = layer(
                    x, src_key_padding_mask=mask, src_mask=future_mask)
            else:
                x = layer(x)
        x = x.permute(1, 0, 2)
        # Remove first element if statics are present
        x = x[:, offset:, :]
        return x

class Perceiver(nn.Module):
    def __init__(self, config: dict, seq_length: int=None, args=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for k, v in vars(args).items():
            setattr(self, k, v)

        for k, v in config.items():
            setattr(self, k, v)

        self.penultimate_input_dim = self.latent_dim

        self.model = PyPerceiver(
            input_channels = self.in_channels,  # number of channels for each token of the input
            input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = self.num_freq_bands,          # number of freq bands, with original value (2 * K + 1)
            max_freq = self.max_freq,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = self.depth, # 6
            num_latents = self.num_latents,  # 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = self.latent_dim,  # 512,            # latent dimension
            cross_heads = self.cross_heads,  #1,             # number of heads for cross attention. paper said 1
            latent_heads = self.latent_heads,  # 8,            # number of heads for latent self attention, 8
            cross_dim_head = self.cross_dim_head,  # 64,
            latent_dim_head = self.latent_dim_head,  # 64,
            num_classes = 1,  # output number of classes
            attn_dropout = self.attn_dropout,  # 0.,
            ff_dropout = self.ff_dropout,  # 0.,
            weight_tie_layers = self.weight_tie_layers,  # False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = self.fourier_encode_data,  # True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = self.self_per_cross_attn  # 2      # number of self attention blocks per cross attention
        )

    def forward(self, inputs):
        # We work with input dims: B x C x L
        inputs = inputs.permute(0, 2, 1)
        preds = self.model(inputs)
        
        return preds


class Conv_Resnet(nn.Module):
    """ Implementation of [1] using convolutions and residual blocks.

    .. [1] Ribeiro et al., Automatic diagnosis of the 12-lead ECG using
    a deep neural network
    """
    def __init__(self, config: dict, seq_length: int=None, args=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for k, v in config.items():
            setattr(self, k, v)

        # First layer
        padding = get_needed_padding(seq_length, seq_length, 1,
                                     self.kernel_size, 1)

        res_length = get_seq_length(seq_length, padding, 1,
                                    self.kernel_size, 1)

        self.first = []
        self.first.append(nn.Conv1d(self.in_channels,
                                              self.num_kernels_first,
                                              self.kernel_size, 1, padding,
                                   bias=False))
        self.first.append(nn.BatchNorm1d(self.num_kernels_first))
        self.first.append(nn.ReLU())
        self.first = nn.Sequential(*self.first)

        self.res_blocks = []
        last_seq_length = res_length
        for res_unit in self.res_units:
            res_unit['seq_length'] = last_seq_length
            res = ResidualBlock(**res_unit)
            self.res_blocks.append(res)
            last_seq_length = res.lengths[-1]

        self.res_ = nn.Sequential(*self.res_blocks)

        n_filters_last = self.res_blocks[-1].n_filters_out
        seq_length_last = self.res_blocks[-1].lengths[-1]

        self.penultimate_input_dim = n_filters_last * seq_length_last

    def forward(self, inputs):
        # We work with input dims: B x C x L
        inputs = self.first[0](inputs)

        X, Y = self.res_([inputs, inputs])

        X = X.view(X.size(0), -1)
        return X


def get_seq_length(input_length, padding, dilation, kernel_size, stride):
    """ Computes the length of a TS after applying
    a 1d convolution with respective parameters.
    """
    a = input_length + 2 * padding - dilation * (kernel_size - 1) -1
    b = stride
    return math.floor((a/b)+1)

def get_needed_padding(l_in, l_out, dilation, kernel_size, stride):
    """ Computes the necessary padding to get a sequence of length
    `l_out` when applyting a convolution with respective parameters to
    a sequence of length `l_in`
    """
    #a = dilation * (kernel_size - 1) + stride * (l_out - 1) + 1 - l_in
    a = dilation * (kernel_size - 1) + (l_in - 1) * (stride - 1)
    pad = math.floor(a / 2)
    return pad

def get_deconv_padding(l_in, l_out, dilation, kernel_size, stride, padding_out):
    p = dilation * (kernel_size - 1) + padding_out + l_in * stride - l_out - stride + 1
    return 0.5 * p

def get_deconv_out_len(l_in, dilation, kernel_size, stride, pad, pad_out):
    return (l_in - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + pad_out + 1

class ResidualBlock(nn.Module):

    def __init__(self, seq_length, in_channels, n_samples_out, n_filters_out,
                 dropout_rate=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False):
        super().__init__()

        self.seq_length = seq_length
        self.in_channels = in_channels
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.lengths = [] # Output length of skip connection

        # We assume inputs of dim B x C x L
        n_samples_in = seq_length
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = self.in_channels

        # Skip connection. Here self.lengths is set.
        self.skip = self._skip_connection(downsample, n_filters_in)
        self.skip = nn.Sequential(*self.skip)

        # 1st conv layer
        self.first = []
        self.first.append(ResidualBlock.get_same_pad_layer(self.lengths[-1],
                                                          self.kernel_size,
                                                          1, 1))
        self.first.append(nn.Conv1d(self.in_channels,
                                              self.n_filters_out,
                                              self.kernel_size,
                                              bias=False))
        self.first += self._batch_norm_plus_activation(self.n_filters_out)
        if self.dropout_rate > 0:
            self.first.append(nn.Dropout(self.dropout_rate))

        # 2nd conv layer
        self.first.append(ResidualBlock.get_same_pad_layer(self.lengths[-1],
                                                          self.kernel_size,
                                                          downsample, 1))
        self.ds = downsample
        self.first.append(nn.Conv1d(self.n_filters_out, self.n_filters_out,
                   self.kernel_size, stride=downsample, bias=False))

        self.first = nn.Sequential(*self.first)
        self.bn_and_relu = nn.Sequential(*self._batch_norm_plus_activation(self.n_filters_out))
        self.do = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()

    @staticmethod
    def get_same_pad_layer(seq_length, k, s, d):
        padding = get_same_padding(seq_length, k, 1, 1)
        padding_left = padding // 2
        padding_right = padding - padding // 2

        return nn.ConstantPad1d([padding_left, padding_right], value=0)

    def _skip_connection(self, downsample, n_filters_in):
        """Implement skip connection."""
        modules = []
        # Deal with downsampling
        #if downsample > 1:
        # Compute length of output after padding
        padding = get_needed_padding(l_in=self.seq_length, l_out=self.seq_length,
                                     dilation=1,
                                     kernel_size=downsample,
                                     stride=downsample)
        pool_output_l = get_seq_length(self.seq_length, padding=padding % downsample,
                                       dilation=1,
                                       kernel_size=downsample,
                                       stride=downsample)
        self.lengths.append(pool_output_l)
        modules.append(ResidualBlock.get_same_pad_layer(self.seq_length,
                                                      downsample,
                                                      downsample, 1))
        modules.append(nn.MaxPool1d(downsample, stride=downsample))
        #else:
        #    pass
        #    #raise ValueError("Number of samples should always decrease.")

        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            modules.append(nn.Conv1d(self.in_channels, self.n_filters_out,
                          1, bias=False))
        return modules

    def _batch_norm_plus_activation(self, num_channels):
        if self.postactivation_bn:
            # Here we assume that X has dims B x C x L
            return [nn.ReLU(), nn.BatchNorm1d(num_channels)]
        return [nn.BatchNorm1d(num_channels), nn.ReLU()]

    def forward(self, inputs, lengths: int=None):
        X, Y = inputs
        s = self.skip(Y)
        X = self.first(X)
        if self.preactivation:
          combined = s.add(X)
          Y = combined
          X = self.bn_and_relu(combined)
          if self.dropout_rate > 0:
              X = self.do(X)
        else:
            X = self.bn(X)
            combined = s.add(X)
            X = self.relu(combined)
            if self.dropout_rate > 0:
                X = self.do(X)
            Y = X
        return X, Y

def get_clin_layers(input_dim, output_dim):
    return Sequential(*[
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.BatchNorm1d(16),
        nn.Linear(16, output_dim),
        nn.ReLU(),
        nn.BatchNorm1d(output_dim),
        nn.Dropout(0.5)])

def get_eimi_layers(input_dim, output_dim):
    layers = [
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, output_dim)]
    layers[-1].bias.data.fill_(-0.66329)
    return Sequential(*layers)

def get_ext_pheno_layers(input_dim, output_dim):
    layers = [
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, output_dim)]
    return Sequential(*layers)

def get_MPSSXS_layers(input_dim, output_dim):
    return Sequential(*[
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, output_dim)])

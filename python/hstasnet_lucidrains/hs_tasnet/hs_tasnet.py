from __future__ import annotations

import time

import pickle
from pathlib import Path
from functools import partial, wraps

import torchaudio
from torchaudio import transforms as T
from torchaudio.functional import resample
# from torchcodec.encoders import AudioEncoder

import sounddevice as sdencoders

from loguru import logger

import torch
import torch.nn.functional as F
from torch.fft import irfft
from torch import nn, compiler, Tensor, tensor, is_tensor, cat, stft, hann_window, view_as_complex, view_as_real
from torch.nn import LSTM, GRU, ConvTranspose1d, Module, ModuleList

from numpy import ndarray
import numpy as np

import einx
from einx import add, multiply, divide
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

import librosa
import matplotlib.pyplot as plt

# ein tensor notation:

# b - batch
# t - sources
# n - length (audio or embed)
# d - dimension / channels
# s - stereo [2]
# c - complex [2]

def store_tensor_to_npy(path, tensor):
    npy_path = Path(path)
    tensor_array = tensor.cpu().detach().numpy()
    np.save(str(npy_path), tensor_array)


def store_tensor_to_readable(path, tensor):
    npy_path = Path(path)
    tensor_array = tensor.cpu().detach().numpy()
    np.savetxt(str(npy_path), tensor_array)

# constants

LSTM = partial(LSTM, batch_first = True)
GRU = partial(GRU, batch_first = True)

(
    view_as_real,
    view_as_complex
) = tuple(compiler.disable()(fn) for fn in (
    view_as_real,
    view_as_complex
))

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t):
    return t

def is_empty(t: Tensor):
    return t.numel() == 0

def current_time_ms():
    return time.time() * 1000

def round_down_to_multiple(num, mult):
    return (num // mult) * mult

def lens_to_mask(lens: Tensor, max_len):
    seq = torch.arange(max_len, device = lens.device)
    return einx.greater('b, n -> b n', lens, seq)

# measure latency

def decorate_print_latency(log_prefix = None):
    def inner(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):

            start_time = current_time_ms()
            out = fn(*args, **kwargs)
            elapsed_ms = current_time_ms() - start_time

            latency_msg = f'{int(elapsed_ms):.2f} ms'

            if exists(log_prefix):
                latency_msg = f'{log_prefix} {latency_msg}'

            logger.debug(latency_msg)
            return out

        return decorated
    return inner

# residual

def residual(fn):

    @wraps(fn)
    def decorated(t, *args, **kwargs):
        out, hidden = fn(t, *args, **kwargs)
        return t + out, hidden

    return decorated

# fft related

class STFT(Module):
    """
    need this custom module to address an issue with certain window and no centering in istft in pytorch - https://github.com/pytorch/pytorch/issues/91309
    this solution was retailored from the working solution used at vocos https://github.com/gemelo-ai/vocos/blob/03c4fcbb321e4b04dd9b5091486eedabf1dc9de0/vocos/spectral_ops.py#L7
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        eps = 1e-11
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = hann_window(win_length)
        self.register_buffer('window', window)

        self.eps = eps

    @compiler.disable()
    def inverse(self, spec):
        n_fft, hop_length, win_length, window = self.n_fft, self.hop_length, self.win_length, self.window

        batch, freqs, frames = spec.shape

        # inverse FFT

        ifft = irfft(spec, n_fft, dim = 1, norm = 'backward')

        store_tensor_to_readable("debug_ifft_0.txt", ifft[0,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_1.txt", ifft[1,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_2.txt", ifft[2,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_3.txt", ifft[3,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_4.txt", ifft[4,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_5.txt", ifft[5,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_6.txt", ifft[6,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_7.txt", ifft[7,:,:].transpose(0,1))

        ifft = multiply('b w f, w', ifft, window)

        # overlap and add

        store_tensor_to_readable("debug_ifft_w_0.txt", ifft[0,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_1.txt", ifft[1,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_2.txt", ifft[2,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_3.txt", ifft[3,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_4.txt", ifft[4,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_5.txt", ifft[5,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_6.txt", ifft[6,:,:].transpose(0,1))
        store_tensor_to_readable("debug_ifft_w_7.txt", ifft[7,:,:].transpose(0,1))

        output_size = (frames - 1) * hop_length + win_length

        y = F.fold(
            ifft,
            output_size = (1, output_size),
            kernel_size = (1, win_length),
            stride = (1, hop_length),
        )[:, 0, 0]

        store_tensor_to_readable("debug_y.txt", y)

        # window envelope

        window_sq = repeat(window.square(), 'w -> 1 w t', t = frames)

        window_envelope = F.fold(
            window_sq,
            output_size = (1, output_size),
            kernel_size = (1, win_length),
            stride = (1, hop_length)
        )

        store_tensor_to_readable("window_envelope.txt", window_envelope[0,0,0,:])

        window_envelope = rearrange(window_envelope, '1 1 1 n -> n')

        # normalize out

        res = divide('b n, n', y, window_envelope.clamp(min = self.eps))

        store_tensor_to_readable("stft_res.txt", res)
        return res

    @compiler.disable()
    def forward(self, audio):

        # store_tensor_to_readable("debug_audio.txt", audio[:, 0:10000])
        # store_tensor_to_readable("debug_window.txt", self.window)

        stft = torch.stft(
            audio,
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            center = False,
            window = self.window,
            return_complex = True
        )
        # for i in range(8):
        #     store_tensor_to_readable(f"debug_stft_{i}.txt", stft[0, :, i])
        
        return stft, stft.abs()

# Tasnet applies a hann window to the convtranspose1d

class ConvTranspose1DWithHannWindow(ConvTranspose1d):
    def __init__(
        self,
        dim,
        dim_out,
        filters,
        **kwargs
    ):
        super().__init__(dim, dim_out, filters, **kwargs)

        self.register_buffer('window', hann_window(filters))

    def forward(self, x):

        filters = self.weight

        windowed_filters = multiply('o i k, k', filters, self.window)
        store_tensor_to_npy("data_53_conv_decode_windowed_filters.npy", windowed_filters)

        return F.conv_transpose1d(
            x,
            windowed_filters,
            stride = self.stride,
            padding = self.padding
        )

# classes

class HSTasNet(Module):
    def __init__(
        self,
        *,
        dim = 500,            # they have 500 hidden units for the network, with 1000 at fusion (concat from both representation branches)
        small = False,        # params cut in half by 1 layer lstm vs 2, fusion uses summed representation
        stereo = True,
        num_basis = 1500,
        segment_len = 1024,
        overlap_len = 512,
        n_fft = 1024,
        sample_rate = 44_100, # they use 41k sample rate
        num_sources = 4,      # drums, bass, vocals, other
        torch_compile = False,
        use_gru = False,
        spec_branch_use_phase = True,
        norm_before_mask_estimate = True # for some reason, training is unstable without a norm - improvise by adding an RMSNorm before final projection
    ):
        super().__init__()

        # auto-saving config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._config = pickle.dumps(_locals)

        # hyperparameters

        audio_channels = 2 if stereo else 1

        self.audio_channels = audio_channels
        self.num_sources = num_sources

        assert overlap_len < segment_len

        self.segment_len = segment_len
        self.overlap_len = overlap_len

        assert divisible_by(segment_len, 2)
        self.causal_pad = segment_len // 2

        # sample rate - 41k in paper

        self.sample_rate = sample_rate

        # spec branch encoder stft hparams

        self.n_fft = n_fft
        self.win_length = segment_len
        self.hop_length = overlap_len

        self.stft = STFT(
            n_fft = n_fft,
            win_length = segment_len,
            hop_length = overlap_len
        )

        real_imag_dim = 2 if spec_branch_use_phase else 1
        spec_dim_input = (n_fft // 2 + 1) * audio_channels * real_imag_dim

        self.spec_branch_use_phase = spec_branch_use_phase # use the phase, proven out in another music sep paper

        self.spec_encode = nn.Sequential(
            Rearrange('(b s) f n ... -> b n (s f ...)', s = audio_channels),
            nn.Linear(spec_dim_input, dim)
        )

        self.to_spec_masks = nn.Sequential(
            nn.RMSNorm(dim) if norm_before_mask_estimate else nn.Identity(),
            nn.Linear(dim, spec_dim_input * num_sources),
            Rearrange('b n (s f c t) -> (b s) f n c t', s = audio_channels, t = num_sources, c = real_imag_dim)
        )

        # waveform branch encoder

        self.stereo = stereo

        self.conv_encode = nn.Conv1d(audio_channels, num_basis * 2, segment_len, stride = overlap_len)

        self.basis_to_embed = nn.Sequential(
            nn.Conv1d(num_basis, dim, 1),
            Rearrange('b c l -> b l c')
        )

        self.to_waveform_masks = nn.Sequential(
            nn.RMSNorm(dim) if norm_before_mask_estimate else nn.Identity(),
            nn.Linear(dim, num_sources * num_basis),
            Rearrange('... (t basis) -> ... basis t', t = num_sources)
        )

        self.conv_decode = ConvTranspose1DWithHannWindow(num_basis, audio_channels, segment_len, stride = overlap_len)

        # init mask to identity

        nn.init.zeros_(self.to_spec_masks[1].weight)
        nn.init.constant_(self.to_spec_masks[1].bias, 1.)

        nn.init.zeros_(self.to_waveform_masks[1].weight)
        nn.init.constant_(self.to_waveform_masks[1].bias, 1.)

        # they do a single layer of lstm in their "small" variant

        self.small = small
        lstm_num_layers = 1 if small else 2

        # rnn

        rnn_klass = LSTM if not use_gru else GRU

        self.pre_spec_branch = rnn_klass(dim, dim, lstm_num_layers)
        self.post_spec_branch = rnn_klass(dim, dim, lstm_num_layers)

        dim_fusion = dim * (2 if not small else 1)

        self.fusion_branch = rnn_klass(dim_fusion, dim_fusion, lstm_num_layers)

        self.pre_waveform_branch = rnn_klass(dim, dim, lstm_num_layers)
        self.post_waveform_branch = rnn_klass(dim, dim, lstm_num_layers)

        # torch compile forward

        if torch_compile:
            self.forward = torch.compile(self.forward)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    # get a spectrogram figure based on hparams of the model

    def save_spectrogram_figure(
        self,
        save_path: str | Path,
        audio: ndarray | Tensor,
        figsize = (10, 4),
        overwrite = False
    ):

        if isinstance(save_path, str):
            save_path = Path(save_path)

        assert overwrite or not save_path.exists(), f'{str(save_path)} already exists'

        # cast to torch tensor

        if isinstance(audio, ndarray):
            audio = torch.from_numpy(audio)

        if audio.ndim == 2:
            # average stereo for now
            audio = reduce(audio, 's n -> n', 'mean')

        assert audio.ndim == 1
        audio = audio.detach().cpu()

        # stft to magnitude to db

        stft = T.Spectrogram(
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            power = 2.
        )(audio)

        magnitude = stft.abs()

        db = T.AmplitudeToDB()(magnitude)

        # pyplot and librosa

        plt.figure(figsize = figsize)

        librosa.display.specshow(
            db.numpy(),
            hop_length = self.hop_length,
            sr = self.sample_rate,
            x_axis = 's',
            y_axis = 'hz'
        )

        plt.colorbar(format = '%+2.0f dB')
        plt.title('Power/Frequency (dB)')
        plt.tight_layout()

        plt.savefig(str(save_path))
        plt.close()

    # saving and loading

    def save_tensor_to_file(
        self,
        output_file: str | Path,
        audio_tensor: Tensor,
        overwrite = False,
        verbose = False
    ):
        if isinstance(output_file, str):
            output_file = Path(output_file)

        assert not exists(output_file) or overwrite, f'{str(output_file)} already exists, set `overwrite = True` if you wish to overwrite'

        if audio_tensor.ndim == 1:
            audio_tensor = repeat(audio_tensor, 'n -> s n', s = 2 if self.stereo else 1)

        # encoder = AudioEncoder(audio_tensor.cpu(), sample_rate = self.sample_rate)
        # encoder.to_file(str(output_file))

        # if verbose:
        #     print(f'audio saved to {str(output_file)}')

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.state_dict(),
            config = self._config,
        )

        torch.save(pkg, str(path))

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location = 'cpu')

        self.load_state_dict(pkg['model'], strict = strict)

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        model = cls(**config)
        model.load_state_dict(pkg['model'], strict = strict)
        return model

    # returns a function that closures the hiddens and past audio chunk for integration with sounddevice audio callback

    def init_stateful_transform_fn(
        self,
        device = None,
        return_reduced_sources: list[int] | None = None,
        auto_convert_to_stereo = True,
        print_latency = False
    ):
        chunk_len = self.overlap_len
        self.eval()

        past_audio = torch.zeros((self.audio_channels, self.overlap_len), device = device)
        hiddens = None

        @torch.inference_mode()
        def fn(audio_chunk: ndarray | Tensor):
            assert audio_chunk.shape[-1] == chunk_len

            nonlocal hiddens
            nonlocal past_audio

            is_numpy_input = isinstance(audio_chunk, ndarray)

            if is_numpy_input:
                audio_chunk = torch.from_numpy(audio_chunk)

            squeezed_audio_channel = audio_chunk.ndim == 1

            if squeezed_audio_channel:
                audio_chunk = rearrange(audio_chunk, '... -> 1 ...')

            if exists(device):
                audio_chunk = audio_chunk.to(device)

            # auto repeat mono to stereo if model is trained with stereo but received audio is mono

            if auto_convert_to_stereo and self.stereo and audio_chunk.shape[0] == 1:
                audio_chunk = repeat(audio_chunk, '1 d -> s d', s = 2)

            # add past audio chunk

            full_chunk = cat((past_audio, audio_chunk), dim = -1)

            full_chunk = rearrange(full_chunk, '... -> 1 ...')

            # forward chunk with past overlap through model

            transformed, hiddens = self.forward(full_chunk, hiddens = hiddens, return_reduced_sources = return_reduced_sources)

            transformed = rearrange(transformed, '1 ... -> ...')

            if squeezed_audio_channel:
                transformed = rearrange(transformed, '... 1 n -> ... n')

            if is_numpy_input:
                transformed = transformed.cpu().numpy()

            # save next overlap chunk for next timestep

            past_audio = audio_chunk

            return transformed[..., -chunk_len:]

        # print latency if needed

        if print_latency:
            fn = decorate_print_latency('stream chunk')(fn)

        return fn

    def sounddevice_stream(
        self,
        return_reduced_sources: list[int],
        duration_seconds = 10,
        channels = None,
        device = None,
        auto_convert_to_stereo = True,
        print_latency = False,
        **stream_kwargs
    ):
        assert len(return_reduced_sources) > 0

        transform_fn = self.init_stateful_transform_fn(
            return_reduced_sources = return_reduced_sources,
            device = device,
            auto_convert_to_stereo = auto_convert_to_stereo,
            print_latency = print_latency
        )

        # sounddevice stream callback, where raw audio can be transformed

        def callback(indata, outdata, frames, time, status):
            assert indata.shape[0] == self.overlap_len

            indata = rearrange(indata, 'n c -> c n')

            transformed = transform_fn(indata)

            transformed = rearrange(transformed, 'c n -> n c')

            outdata[:] = transformed

        # variables

        duration_ms = int(duration_seconds * 1000)

        # sounddevice streaming

        with sd.Stream(
            **stream_kwargs,
            channels = channels,
            callback = callback,
            samplerate = self.sample_rate
        ):
            sd.sleep(duration_ms)

    def process_audio_file(
        self,
        input_file: str | Path,
        return_reduced_sources: list[int],
        output_file: str | Path | None = None,
        auto_convert_to_stereo = True,
        overwrite = False
    ):
        if isinstance(input_file, str):
            input_file = Path(input_file)

        assert len(return_reduced_sources) > 0
        assert input_file.exists(), f'{str(input_file)} not found'

        audio_tensor, sample_rate = torchaudio.load(input_file)

        # resample if need be

        if sample_rate != self.sample_rate:
            audio_tensor = resample(audio_tensor, sample_rate, self.sample_rate)

        # curtail to divisible segment lens

        audio_len = audio_tensor.shape[-1]
        rounded_down_len = round_down_to_multiple(audio_len, self.segment_len)

        audio_tensor = audio_tensor[..., :rounded_down_len]

        # add batch

        audio_tensor = rearrange(audio_tensor, '... -> 1 ...')

        # maybe mono to stereo

        mono_to_stereo = self.stereo and auto_convert_to_stereo and audio_tensor.shape[1] == 1

        if mono_to_stereo:
            audio_tensor = repeat(audio_tensor, '1 1 n -> 1 s n', s = 2)

        # transform

        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            self.eval()
            transformed, _ = self.forward(audio_tensor, return_reduced_sources = return_reduced_sources)

        # remove batch

        transformed = rearrange(transformed, '1 ... -> ...')

        # maybe stereo to mono

        if mono_to_stereo:
            transformed = reduce(transformed, 's n -> 1 n', 'mean')

        # save output file

        if not exists(output_file):
            output_file = Path(input_file.parents[-2] / f'{input_file.stem}-out.wav')

        assert output_file != input_file

        self.save_tensor_to_file(str(output_file), transformed.cpu(), overwrite = overwrite)

    def forward(
        self,
        audio,             # (b {s} n)  - {} meaning optional dimension for stereo or not
        hiddens = None,
        targets = None,    # (b t {s} n)
        audio_lens = None, # (b)
        return_reduced_sources: list[int] | None = None,
        auto_causal_pad = None,
        auto_curtail_length_to_multiple = True,
        return_unreduced_loss = False,
        return_targets_with_loss = False
    ):
        auto_causal_pad = default(auto_causal_pad, self.training)

        assert auto_curtail_length_to_multiple or divisible_by(audio.shape[-1], self.segment_len)

        # take care of audio being passed in that isn't multiple of segment length

        if auto_curtail_length_to_multiple:
            round_down_audio_len = round_down_to_multiple(audio.shape[-1], self.segment_len)
            audio = audio[..., :round_down_audio_len]

            if exists(targets):
                targets = targets[..., :round_down_audio_len]

        # variables

        batch, audio_len, device = audio.shape[0], audio.shape[-1], audio.device

        # validate lengths

        assert not exists(targets) or audio.shape[-1] == targets.shape[-1]

        assert not is_empty(audio), f'audio is empty, probably insufficient lengthed audio given segment length and auto-rounding down the length'

        # their small version probably does not need a skip connection

        maybe_residual = residual if not self.small else identity

        if exists(targets):
            assert targets.shape == (batch, self.num_sources, *audio.shape[1:])

        # handle audio shapes

        audio_is_squeezed = audio.ndim == 2 # input audio is (batch, length) shape, make sure output is correspondingly squeezed

        if audio_is_squeezed: # (b l) -> (b c l)
            audio = rearrange(audio, 'b l -> b 1 l')

        assert not (self.stereo and audio.shape[1] != 2), 'audio channels must be 2 if training stereo'

        # handle masking

        need_audio_mask = exists(audio_lens)

        if need_audio_mask:
            audio_lens = round_down_to_multiple(audio_lens, self.segment_len)
            assert (audio_lens > 0).all()

            audio_mask = lens_to_mask(audio_lens, audio_len)

        # pad the audio manually on the left side for causal, and set stft center False

        if auto_causal_pad:
            audio = F.pad(audio, (self.causal_pad, 0), value = 0.)

        # handle spec encoding

        store_tensor_to_npy('prog_0_audio_input.npy', audio)
        spec_audio_input = rearrange(audio, 'b s ... -> (b s) ...')
        # print(spec_audio_input[0, 1500:2000])

        complex_spec, magnitude = self.stft(spec_audio_input)
        store_tensor_to_npy('prog_1_stft_complex.npy', complex_spec)
        store_tensor_to_npy('prog_2_stft_magnitude.npy', magnitude)

        # print the complex_spec for the first few entries

        for i in range(8):
            print(complex_spec[0, :, i])


        if self.spec_branch_use_phase:
            real_imag_spec = torch.view_as_real(complex_spec)

            self.spec_encode._modules['0'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_3_spec_reordered.npy', output))
            store_tensor_to_npy("data_0_spec_encode_bias.npy", self.spec_encode._modules['1'].bias.data)
            store_tensor_to_npy("data_1_spec_encode_weight.npy", self.spec_encode._modules['1'].weight.data)
            
            spec = self.spec_encode(real_imag_spec)
            store_tensor_to_npy('prog_4_spec_encoded.npy', spec)
        else:

            spec = self.spec_encode(magnitude)

        # handle encoding as detailed in original tasnet
        # to keep non-negative, they do a glu with relu on main branch

        to_relu, to_sigmoid = self.conv_encode(audio).chunk(2, dim = 1)
        store_tensor_to_npy('data_2_audio_conv_encode_bias.npy', self.conv_encode.bias)
        store_tensor_to_npy('data_3_audio_conv_encode_weight.npy', self.conv_encode.weight)
        store_tensor_to_npy('prog_5_audio_conv_encode_relu_pre.npy', to_relu)
        store_tensor_to_npy('prog_6_audio_conv_encode_sigmoid_pre.npy', to_sigmoid)

        basis = to_relu.relu() * to_sigmoid.sigmoid() # non-negative basis (1024)
        store_tensor_to_npy('prog_7_audio_basis.npy', basis)

        # basis to waveform embed for mask estimation
        # paper mentions linear for any mismatched dimensions

        store_tensor_to_npy("data_4_basis_to_embed_bias.npy",self.basis_to_embed._modules['0'].bias.data)
        store_tensor_to_npy("data_5_basis_to_embed_weight.npy",self.basis_to_embed._modules['0'].weight.data)
        self.basis_to_embed._modules['0'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_8_basis_to_embed_conv.npy', output))
        # self.basis_to_embed._modules['0'].register_forward_hook(lambda module, input, output: store_tensor_to_readable('basis_to_embed_conv.txt', output[0,:,:]))
        waveform = self.basis_to_embed(basis)
        store_tensor_to_npy('prog_9_waveform.npy', waveform)
        # store_tensor_to_readable('waveform.txt', waveform[0,:,:])
        # handle previous hiddens

        hiddens = default(hiddens, (None,) * 5)

        (
            pre_spec_hidden,
            pre_waveform_hidden,
            fusion_hidden,
            post_spec_hidden,
            post_waveform_hidden
        ) = hiddens

        # residuals

        spec_residual, waveform_residual = spec, waveform

        store_tensor_to_npy("data_6_pre_spec_branch_weight_ih_l0.npy",self.pre_spec_branch.weight_ih_l0.data)
        store_tensor_to_npy("data_7_pre_spec_branch_weight_hh_l0.npy",self.pre_spec_branch.weight_hh_l0.data)
        store_tensor_to_npy("data_8_pre_spec_branch_bias_ih_l0.npy",self.pre_spec_branch.bias_ih_l0.data)
        store_tensor_to_npy("data_9_pre_spec_branch_bias_hh_l0.npy",self.pre_spec_branch.bias_hh_l0.data)
 
        store_tensor_to_npy("data_10_pre_spec_branch_weight_ih_l1.npy",self.pre_spec_branch.weight_ih_l1.data)
        store_tensor_to_npy("data_11_pre_spec_branch_weight_hh_l1.npy",self.pre_spec_branch.weight_hh_l1.data)
        store_tensor_to_npy("data_12_pre_spec_branch_bias_ih_l1.npy",self.pre_spec_branch.bias_ih_l1.data)
        store_tensor_to_npy("data_13_pre_spec_branch_bias_hh_l1.npy",self.pre_spec_branch.bias_hh_l1.data)

        spec, next_pre_spec_hidden = maybe_residual(self.pre_spec_branch)(spec, pre_spec_hidden)
        store_tensor_to_npy('prog_10_spec_pre_rnn.npy', spec)

        store_tensor_to_npy("data_14_pre_waveform_branch_weight_ih_l0.npy",self.pre_waveform_branch.weight_ih_l0.data)
        store_tensor_to_npy("data_15_pre_waveform_branch_weight_hh_l0.npy",self.pre_waveform_branch.weight_hh_l0.data)
        store_tensor_to_npy("data_16_pre_waveform_branch_bias_ih_l0.npy",self.pre_waveform_branch.bias_ih_l0.data)
        store_tensor_to_npy("data_17_pre_waveform_branch_bias_hh_l0.npy",self.pre_waveform_branch.bias_hh_l0.data)

        store_tensor_to_npy("data_18_pre_waveform_branch_weight_ih_l1.npy",self.pre_waveform_branch.weight_ih_l1.data)
        store_tensor_to_npy("data_19_pre_waveform_branch_weight_hh_l1.npy",self.pre_waveform_branch.weight_hh_l1.data)
        store_tensor_to_npy("data_20_pre_waveform_branch_bias_ih_l1.npy",self.pre_waveform_branch.bias_ih_l1.data)
        store_tensor_to_npy("data_21_pre_waveform_branch_bias_hh_l1.npy",self.pre_waveform_branch.bias_hh_l1.data)

        waveform, next_pre_waveform_hidden = maybe_residual(self.pre_waveform_branch)(waveform, pre_waveform_hidden)
        store_tensor_to_npy('prog_11_waveform_pre_rnn.npy', waveform)

        # if small, they just sum the two branches

        if self.small:
            fusion_input = spec + waveform
        else:
            fusion_input = cat((spec, waveform), dim = -1)

        # fusing
        store_tensor_to_npy('prog_12_fusion_input.npy', fusion_input)

        store_tensor_to_npy("data_22_fusion_branch_weight_ih_l0.npy",self.fusion_branch.weight_ih_l0.data)
        store_tensor_to_npy("data_23_fusion_branch_weight_hh_l0.npy",self.fusion_branch.weight_hh_l0.data)
        store_tensor_to_npy("data_24_fusion_branch_bias_ih_l0.npy",self.fusion_branch.bias_ih_l0.data)
        store_tensor_to_npy("data_25_fusion_branch_bias_hh_l0.npy",self.fusion_branch.bias_hh_l0.data)

        store_tensor_to_npy("data_26_fusion_branch_weight_ih_l1.npy",self.fusion_branch.weight_ih_l1.data)
        store_tensor_to_npy("data_27_fusion_branch_weight_hh_l1.npy",self.fusion_branch.weight_hh_l1.data)
        store_tensor_to_npy("data_28_fusion_branch_bias_ih_l1.npy",self.fusion_branch.bias_ih_l1.data)
        store_tensor_to_npy("data_29_fusion_branch_bias_hh_l1.npy",self.fusion_branch.bias_hh_l1.data)

        fused, next_fusion_hidden = maybe_residual(self.fusion_branch)(fusion_input, fusion_hidden)
        store_tensor_to_npy('prog_13_fusion_rnn.npy', fused)

        # split if not small, handle small next week

        if self.small:
            fused_spec, fused_waveform = fused, fused
        else:
            fused_spec, fused_waveform = fused.chunk(2, dim = -1)

        # residual from encoded

        spec = fused_spec + spec_residual

        waveform = fused_waveform + waveform_residual

        store_tensor_to_npy('prog_14_spec_with_residual.npy', spec)
        store_tensor_to_npy('prog_15_waveform_with_residual.npy', waveform)

        # layer for both branches

        store_tensor_to_npy("data_30_post_spec_branch_weight_ih_l0.npy",self.post_spec_branch.weight_ih_l0.data)
        store_tensor_to_npy("data_31_post_spec_branch_weight_hh_l0.npy",self.post_spec_branch.weight_hh_l0.data)
        store_tensor_to_npy("data_32_post_spec_branch_bias_ih_l0.npy",self.post_spec_branch.bias_ih_l0.data)
        store_tensor_to_npy("data_33_post_spec_branch_bias_hh_l0.npy",self.post_spec_branch.bias_hh_l0.data)

        store_tensor_to_npy("data_34_post_spec_branch_weight_ih_l1.npy",self.post_spec_branch.weight_ih_l1.data)
        store_tensor_to_npy("data_35_post_spec_branch_weight_hh_l1.npy",self.post_spec_branch.weight_hh_l1.data)
        store_tensor_to_npy("data_36_post_spec_branch_bias_ih_l1.npy",self.post_spec_branch.bias_ih_l1.data)
        store_tensor_to_npy("data_37_post_spec_branch_bias_hh_l1.npy",self.post_spec_branch.bias_hh_l1.data)

        spec, next_post_spec_hidden = maybe_residual(self.post_spec_branch)(spec, post_spec_hidden)
        store_tensor_to_npy('prog_16_spec_post_rnn.npy', spec)

        store_tensor_to_npy("data_38_post_waveform_branch_weight_ih_l0.npy",self.post_waveform_branch.weight_ih_l0.data)
        store_tensor_to_npy("data_39_post_waveform_branch_weight_hh_l0.npy",self.post_waveform_branch.weight_hh_l0.data)
        store_tensor_to_npy("data_40_post_waveform_branch_bias_ih_l0.npy",self.post_waveform_branch.bias_ih_l0.data)
        store_tensor_to_npy("data_41_post_waveform_branch_bias_hh_l0.npy",self.post_waveform_branch.bias_hh_l0.data)

        store_tensor_to_npy("data_42_post_waveform_branch_weight_ih_l1.npy",self.post_waveform_branch.weight_ih_l1.data)
        store_tensor_to_npy("data_43_post_waveform_branch_weight_hh_l1.npy",self.post_waveform_branch.weight_hh_l1.data)
        store_tensor_to_npy("data_44_post_waveform_branch_bias_ih_l1.npy",self.post_waveform_branch.bias_ih_l1.data)
        store_tensor_to_npy("data_45_post_waveform_branch_bias_hh_l1.npy",self.post_waveform_branch.bias_hh_l1.data)

        waveform, next_post_waveform_hidden = maybe_residual(self.post_waveform_branch)(waveform, post_waveform_hidden)
        store_tensor_to_npy('prog_17_waveform_post_rnn.npy', waveform)

        # spec mask
        store_tensor_to_npy("data_46_to_spec_masks_RMSNorm_weight.npy",self.to_spec_masks._modules['0'].weight.data)
        store_tensor_to_npy("data_47_to_spec_masks_linear_weight.npy",self.to_spec_masks._modules['1'].weight.data)
        store_tensor_to_npy("data_48_to_spec_masks_linear_bias.npy",self.to_spec_masks._modules['1'].bias.data)

        self.to_spec_masks._modules['0'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_18_to_spec_masks_RMSNorm.npy', output))
        self.to_spec_masks._modules['1'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_19_to_spec_masks_linear.npy', output))
        self.to_spec_masks._modules['2'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_20_to_spec_mask.npy', output))
        spec_mask = self.to_spec_masks(spec)

        if self.spec_branch_use_phase:

            scaled_real_imag_spec = multiply('b ..., b ... t -> (b t) ...', real_imag_spec, spec_mask)
        
            complex_spec_per_source = torch.view_as_complex(scaled_real_imag_spec.contiguous())
            store_tensor_to_npy("prog_21_scaled_complex_spec_per_source.npy", complex_spec_per_source)

        else:
            spec_mask = rearrange(spec_mask, '... 1 t -> ... t')

            magnitude, phase = complex_spec.abs(), complex_spec.angle()

            scaled_magnitude = multiply('b ..., b ... t -> (b t) ...', magnitude, spec_mask)

            phase = repeat(phase, 'b ... -> (b t) ...', t = self.num_sources)

            complex_spec_per_source = torch.polar(scaled_magnitude, phase)

        recon_audio_from_spec = self.stft.inverse(complex_spec_per_source)
        store_tensor_to_npy("prog_22_recon_audio_from_spec_pre_reshape.npy", recon_audio_from_spec)

        recon_audio_from_spec = rearrange(recon_audio_from_spec, '(b s t) ... -> b t s ...', b = batch, s = self.audio_channels)
        store_tensor_to_npy("prog_23_recon_audio_from_spec_post_reshape.npy", recon_audio_from_spec)

        # waveform mask

        store_tensor_to_npy("data_49_to_waveform_masks_RMSNorm_weight.npy",self.to_waveform_masks._modules['0'].weight.data)
        store_tensor_to_npy("data_50_to_waveform_masks_linear_weight.npy",self.to_waveform_masks._modules['1'].weight.data)
        store_tensor_to_npy("data_51_to_waveform_masks_linear_bias.npy",self.to_waveform_masks._modules['1'].bias.data)

        self.to_waveform_masks._modules['0'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_24_to_waveform_masks_RMSNorm.npy', output))
        self.to_waveform_masks._modules['1'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_25_to_waveform_masks_linear.npy', output))
        self.to_waveform_masks._modules['2'].register_forward_hook(lambda module, input, output: store_tensor_to_npy('prog_26_to_waveform_mask.npy', output))
        waveform_mask = self.to_waveform_masks(waveform)

        basis_per_source = multiply('b basis n, b n basis t -> (b t) basis n', basis, waveform_mask)
        store_tensor_to_npy("prog_27_basis_per_source.npy", basis_per_source)

         # conv decode

        store_tensor_to_npy("data_52_conv_decode_weight.npy",self.conv_decode.weight.data)
        recon_audio_from_waveform = self.conv_decode(basis_per_source)
        store_tensor_to_npy("prog_28_recon_audio_from_waveform_pre_reshape.npy", recon_audio_from_waveform)

        recon_audio_from_waveform = rearrange(recon_audio_from_waveform, '(b t) ... -> b t ...', b = batch)
        store_tensor_to_npy("prog_29_recon_audio_from_waveform_post_reshape.npy", recon_audio_from_waveform)

        # recon audio

        recon_audio = recon_audio_from_spec + recon_audio_from_waveform
        store_tensor_to_npy("prog_30_recon_audio_combined.npy", recon_audio)

        # take care of l1 loss if target is passed in

        if audio_is_squeezed:
            recon_audio = rearrange(recon_audio, 'b s 1 n -> b s n')

        # excise out the causal padding

        if auto_causal_pad:
            recon_audio = recon_audio[..., self.causal_pad:]

        if exists(targets):
            recon_loss = F.l1_loss(recon_audio, targets, reduction = 'none' if need_audio_mask or return_unreduced_loss else 'mean') # they claim a simple l1 loss is better than all the complicated stuff of past

            if need_audio_mask:
                recon_loss = rearrange(recon_loss, 'b ... n -> b n ...')
                recon_loss = recon_loss[audio_mask].mean()

            if not return_targets_with_loss:
                return recon_loss

            return recon_loss, targets

        # outputs

        lstm_hiddens = (
            next_pre_spec_hidden,
            next_pre_waveform_hidden,
            next_fusion_hidden,
            next_post_spec_hidden,
            next_post_waveform_hidden
        )

        if need_audio_mask:
            recon_audio = einx.where('b n, b t n,', audio_mask, recon_audio, 0.)

        if exists(return_reduced_sources):
            return_reduced_sources = tensor(return_reduced_sources, device = device)

            sources = recon_audio.index_select(1, return_reduced_sources)
            recon_audio = reduce(sources, 'b s ... -> b ...', 'sum')

        return recon_audio, lstm_hiddens

import torch
import torchaudio
import numpy as np
from scipy.io import wavfile

import onnx
import onnxruntime

from hs_tasnet.hs_tasnet import HSTasNet

# Paths
MODEL_PATH = 'path/to/your/trained_hs_tasnet.pth'
AUDIO_PATH = 'path/to/your/stereo_audio.wav'
ONNX_EXPORT_PATH = 'hs_tasnet_export.onnx'
INTERMEDIATE_STATES_PATH = 'intermediate_states.npy'

# 1. Load trained model
model = HSTasNet.init_and_load_from('./checkpoints/hs-tasnet.ckpt.train1.pt')

# 2. Export to ONNX
device = torch.device('cpu') #('cuda' if torch.cuda.is_available() else 'cpu')
# dummy_input = torch.randn(1, 2, 256 * 1024).to(device)  # (batch, channels, samples), adjust shape as needed

samplerate, input = wavfile.read('./data/Tom McKenzie - Directions.wav')
if input.dtype == np.int16:
    input = input.astype(np.float32) / 32768.0
elif input.dtype == np.int32:
    input = input.astype(np.float32) / 2147483648.0
elif input.dtype == np.uint8:
    input = (input.astype(np.float32) - 128) / 128.0
else:
    input = input.astype(np.float32)
input = torch.tensor(input, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)  # (1, samples, channels)

input = input[:, :, 4096:4096+8*1024]  # Trim or pad to desired length
model.forward(input)

onnx_program = torch.onnx.export(model, input)
onnx_program.save(ONNX_EXPORT_PATH)

# torch.onnx.export(
#     model, 
#     dummy_input, 
#     ONNX_EXPORT_PATH, 
#     input_names=['input'], 
#     output_names=['output'],
#     opset_version=13,
#     dynamic_axes={'input': {0: 'batch', 2: 'samples'}, 'output': {0: 'batch', 2: 'samples'}}
# )



# 3. Load stereo audio file
waveform, sr = torchaudio.load(AUDIO_PATH)
if waveform.shape[0] != 2:
    raise ValueError("Audio file is not stereo.")
waveform = waveform.unsqueeze(0).to(device)  # (1, 2, samples)

# 4. Run through model and capture intermediate states
intermediate_states = {}

def save_hook(name):
    def hook(module, input, output):
        intermediate_states[name] = output.detach().cpu().numpy()
    return hook

# Register hooks (example: register on first 2 layers, adjust as needed)
for name, module in list(model.named_modules())[:2]:
    module.register_forward_hook(save_hook(name))

with torch.no_grad():
    _ = model(waveform)

# 5. Save intermediate states
np.save(INTERMEDIATE_STATES_PATH, intermediate_states)

print(f"ONNX model exported to {ONNX_EXPORT_PATH}")
print(f"Intermediate states saved to {INTERMEDIATE_STATES_PATH}")
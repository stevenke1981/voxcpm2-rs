"""Quick script to inspect safetensors metadata without loading bfloat16 data."""
import json

# Load from config
with open('models/VoxCPM2/config.json') as f:
    cfg = json.load(f)

print('=== Config ===')
relevant = {k: cfg.get(k) for k in ['feat_dim','patch_size','residual_lm_num_layers',
                                      'residual_lm_no_rope']}
print(json.dumps(relevant, indent=2))

if 'encoder_config' in cfg:
    print("encoder_config:", json.dumps(cfg['encoder_config'], indent=2))
if 'dit_config' in cfg:
    dit = cfg['dit_config']
    print("dit_config:", json.dumps({k: dit[k] for k in ['hidden_dim','ffn_dim','num_heads','num_layers','kv_channels']}, indent=2))

# Read safetensors header manually (header is JSON at the start)
with open('models/VoxCPM2/model.safetensors', 'rb') as f:
    header_len_bytes = f.read(8)
    header_len = int.from_bytes(header_len_bytes, 'little')
    header_json = f.read(header_len).decode('utf-8')
    header = json.loads(header_json)

print('\n=== Key weight shapes ===')
targets = [
    'fusion_concat_proj.weight', 'fusion_concat_proj.bias',
    'feat_decoder.estimator.cond_proj.weight',
    'feat_decoder.estimator.in_proj.weight', 'feat_decoder.estimator.in_proj.bias',
    'feat_decoder.estimator.out_proj.weight', 'feat_decoder.estimator.out_proj.bias',
    'feat_encoder.in_proj.weight', 'feat_encoder.in_proj.bias',
    'feat_encoder.special_token',
    'feat_encoder.encoder.norm.weight',
    'feat_decoder.estimator.decoder.norm.weight',
    'feat_decoder.estimator.decoder.layers.0.input_layernorm.weight',
    'enc_to_lm_proj.weight', 'enc_to_lm_proj.bias',
    'lm_to_dit_proj.weight', 'lm_to_dit_proj.bias',
    'res_to_dit_proj.weight', 'res_to_dit_proj.bias',
]

for t in targets:
    if t in header:
        info = header[t]
        print(f'  {t}: shape={info["shape"]} dtype={info["dtype"]}')
    else:
        print(f'  {t}: NOT FOUND')

print(f'\nTotal tensors: {len(header)}')
print(f'Tensor names sample (first 10):')
names = list(header.keys())[:10]
for n in names:
    print(f'  {n}: {header[n]["shape"]}')

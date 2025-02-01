#%%
from buffer import AllActivationBuffer
from training import train_scae_suite
from trainers.scae import TrainerConfig
from utils import load_model_with_folded_ln2, load_iterable_dataset

import torch as t
from huggingface_hub import login
import pickle

# Jacob's token but feel free to use
login("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")
device = "cuda:0" if t.cuda.is_available() else "cpu"


#%%
DTYPE = t.bfloat16
MODEL_NAME = "gpt2"

out_batch_size = 64
num_tokens = int(1e7)
remove_bos = True # don't train on BOS activations: they lead to weird loss spikes, especially with bf16.


#%%
model = load_model_with_folded_ln2(MODEL_NAME, device=device, torch_dtype=DTYPE)
data = load_iterable_dataset('Skylion007/openwebtext')

buffer = AllActivationBuffer(
    data=data,
    model=model,
    model_name=MODEL_NAME,
    n_ctxs=256,  # you can set this higher or lower depending on your available memory
    device="cuda",
    out_batch_size=out_batch_size,
    refresh_batch_size=256,
    dtype=DTYPE,
    remove_bos=remove_bos
) 


#%%
trainer_cfg = TrainerConfig(
    steps=num_tokens // out_batch_size,
    n_threshold=0,
    n_random=0,
    random_loss_coeff=0.0
)

# Load connections dict
with open("connections_100.pkl", "rb") as f:
    connections = pickle.load(f)

# If using random connections
# fake_connections = generate_fake_connections(
#     connections,
#     num_features=num_features,
# )

#%%
# Special connections for mlp_1

# for k in connections['mlp_1'].keys():
#     connections['mlp_1'][k] = t.arange(0, num_features).unsqueeze(0).repeat(num_features, 1).cuda()

# for k in connections['mlp_0'].keys():
#     connections['mlp_0'][k] = t.arange(0, num_features).unsqueeze(0).repeat(num_features, 1).cuda()



#%%
trainer = train_scae_suite(
    buffer,
    trainer_config=trainer_cfg,
    connections=connections,
    steps=num_tokens // out_batch_size,
    save_steps = 1000,
    dtype = DTYPE,
    device=device,
    log_steps = 20,
    use_wandb = True,
    repo_id_in='jacobcd52/gpt2_suite_folded_ln_nobos',
    repo_id_out = "jacobcd52/gpt2_scae",
    wandb_project_name="gpt2_scae_finetuning",
)

#%%

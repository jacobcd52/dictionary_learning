import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset
import os
import json
import gc

from dictionary_learning.scae import SCAESuite
import interp.interp_utils as interp_utils
from dictionary_learning.buffer import chunk_and_tokenize

# --- Constants ---
# You can add more repository IDs to this list.
# REPO_LIST = [
#     f"jacobcd52/pythia-70m_warmup0.3_k16_mask{mask_coeff}_fact_fvu0.2_fvu_sparse1.0_fvu1.0_lr0.0005"
#     for mask_coeff in [0, 0.00003, 0.0001, 0.0003, 0.001, 0.01]
# ]
REPO_LIST = ["jacobcd52/pythia-70m_warmup0.3_k16_mask0_fact_fvu0.0_fvu_sparse0.0_fvu1.0_lr0.0005"]

MODEL_NAME = "EleutherAI/pythia-70m"
PATH_TO_PILE = "/root/dictionary_learning/pile-uncopyrighted"
OUTPUT_DIR = "/root/dictionary_learning/interp/feature_data" 

SEQ_LEN = 128
# Number of documents to load from the raw dataset. These will be tokenized and chunked.
RAW_DATA_SAMPLES = 10_000
# The number of final token sequences to process for activation collection.
NUM_SAMPLES_TO_PROCESS = RAW_DATA_SAMPLES
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """
    Main function to process SCAE suites, find non-dead features, and save the results.
    """
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Model and Tokenizer ---
    print(f"Loading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = utils.get_tokenizer_with_bos(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load and Tokenize Dataset ---
    raw_dataset = load_dataset(
        PATH_TO_PILE,
        split=f"train[:1%]",
    ).shuffle(seed=42)

    tokenized_dataset = chunk_and_tokenize(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        text_key="text",
        max_length=SEQ_LEN,
        num_proc=max(1, os.cpu_count() // 2),
        load_from_cache_file=True
    )

    if len(tokenized_dataset) > NUM_SAMPLES_TO_PROCESS:
        dataset_to_process = tokenized_dataset.select(range(NUM_SAMPLES_TO_PROCESS))
    else:
        dataset_to_process = tokenized_dataset


    # --- Process Each Suite ---
    for repo_id in REPO_LIST:
        print(f"\n--- Processing suite: {repo_id} ---")
        
        suite_name = repo_id.split("/")[-1]
        activations_dir = os.path.join(OUTPUT_DIR, suite_name)
        
        # Load suite
        print("Loading suite...")
        suite = SCAESuite.from_pretrained(
            repo_id=repo_id,
            model=model,
            device=DEVICE,
            dtype=torch.bfloat16
        )

        # Collect activations for both sparse and non-sparse modes
        for sparse_mode in [True, False]:
            interp_utils.collect_activations_and_tokens(
                suite=suite,
                model=model,
                tokenizer=tokenizer,
                dataset=dataset_to_process,
                device=DEVICE,
                output_dir=activations_dir,
                run_mode_sparse=sparse_mode,
                batch_size=BATCH_SIZE,
                max_batches_to_process=None # Process the whole selected dataset
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # Find non-dead features by analyzing the collected activations
        print("Finding non-dead features...")
        non_dead_features = interp_utils.find_non_dead_features(activations_dir)

        # Save the results to a JSON file
        output_file_path = os.path.join(OUTPUT_DIR, f"{suite_name}_non_dead_features.json")
        print(f"Saving non-dead feature data to {output_file_path}")
        with open(output_file_path, 'w') as f:
            json.dump(non_dead_features, f, indent=4)
        
        del suite
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n--- All suites processed. ---")

if __name__ == "__main__":
    main() 
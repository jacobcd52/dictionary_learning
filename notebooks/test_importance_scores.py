#%%
#%%
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('..')

import torch as t
import einops
from typing import Dict, List
import torch.sparse as sparse
from tqdm import tqdm
import gc
import pickle
from pathlib import Path

from trainers.scae import SCAESuite
from buffer import AllActivationBuffer
from utils import load_model_with_folded_ln2, load_iterable_dataset
from find_top_connections import get_importance_scores, get_valid_pairs, process_pair

DTYPE = t.bfloat16
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
MODEL_NAME = "gpt2"
t.manual_seed(42)
t.set_grad_enabled(False)
gc.collect()

#%%
import torch as t
import einops
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
import pickle

@dataclass
class MockConfig:
    dict_size: int

@dataclass
class MockAutoEncoder:
    encoder: t.nn.Linear
    decoder: t.nn.Linear

class MockSCAESuite:
    def __init__(
        self,
        submodule_names: List[str],
        dict_sizes: Dict[str, int],
        feature_dim: int,
        device: str = 'cuda'
    ):
        self.submodule_names = submodule_names
        self.configs = {name: MockConfig(size) for name, size in dict_sizes.items()}
        self.aes = {}
        
        for name, size in dict_sizes.items():
            encoder = t.nn.Linear(feature_dim, size, bias=False, device=device)
            decoder = t.nn.Linear(size, feature_dim, bias=False, device=device)
            
            t.nn.init.eye_(encoder.weight)  # Identity for simplicity
            t.nn.init.eye_(decoder.weight)  # Identity for simplicity
            
            self.aes[name] = MockAutoEncoder(encoder, decoder)
    
    def vanilla_forward(self, inputs, return_features=True, return_topk=True):
        return None, None, self.mock_topk_info

    def set_mock_topk_info(self, topk_info):
        self.mock_topk_info = topk_info

@dataclass
class MockInput:
    src: t.Tensor

class MockBuffer:
    def __init__(self, inputs):
        self.inputs = inputs
        self.current = 0
    
    def __next__(self):
        if self.current >= len(self.inputs):
            raise StopIteration
        result = self.inputs[self.current]
        self.current += 1
        return result

def test_valid_pairs():
    print("\n=== Testing get_valid_pairs ===")
    suite = MockSCAESuite(
        submodule_names=['mlp_0', 'mlp_1', 'mlp_2', 'attn_0', 'attn_1'],
        dict_sizes={'mlp_0': 10, 'mlp_1': 10, 'mlp_2': 10, 'attn_0': 10, 'attn_1': 10},
        feature_dim=8
    )
    
    pairs = get_valid_pairs(suite)
    expected_pairs = [
        ('mlp_0', 'mlp_1'),
        ('mlp_0', 'mlp_2'),
        ('mlp_1', 'mlp_2'),
        ('attn_0', 'mlp_1'),
        ('attn_0', 'mlp_2'),
        ('attn_1', 'mlp_2'),
    ]
    
    print(f"Found pairs: {pairs}")
    print(f"Expected pairs: {expected_pairs}")
    print(f"Test passed: {set(pairs) == set(expected_pairs)}")

def test_process_pair_simple():
    print("\n=== Testing process_pair simple case ===")
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    feature_dim = 4
    batch_size = 2
    
    # Create suite with known dictionary sizes
    suite = MockSCAESuite(
        submodule_names=['mlp_0', 'mlp_1'],
        dict_sizes={'mlp_0': 4, 'mlp_1': 4},
        feature_dim=feature_dim,
        device=device
    )
    
    # Create mock activations where features 0,1 active in first batch
    # and features 2,3 active in second batch
    mock_topk_info = {
        'mlp_0': (
            t.tensor([[0, 1], [2, 3]], device=device),  # indices
            t.tensor([[1.0, 0.5], [0.8, 0.4]], device=device)  # values
        ),
        'mlp_1': (
            t.tensor([[0, 1], [2, 3]], device=device),  # indices
            t.tensor([[1.0, 0.5], [0.8, 0.4]], device=device)  # values
        )
    }
    suite.set_mock_topk_info(mock_topk_info)
    
    inputs = [
        MockInput(t.randn(batch_size, feature_dim, device=device))
        for _ in range(2)
    ]
    buffer = MockBuffer(inputs)
    
    scores = process_pair(
        suite=suite,
        buffer=buffer,
        up_name='mlp_0',
        down_name='mlp_1',
        num_batches=2,
        top_c=3,
        save_dir=None
    )
    
    dense_scores = scores.to_dense()
    print(f"Score matrix shape: {dense_scores.shape}")
    print(f"Score matrix:\n{dense_scores}")
    print(f"Non-zero elements: {(dense_scores != 0).sum().item()}")
    print(f"Maximum score: {dense_scores.max().item():.4f}")
    print(f"All scores non-negative: {t.all(dense_scores >= 0).item()}")

def test_numerical_accuracy():
    print("\n=== Testing numerical accuracy ===")
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    feature_dim = 2
    
    # Simple case with identity matrices
    suite = MockSCAESuite(
        submodule_names=['mlp_0', 'mlp_1'],
        dict_sizes={'mlp_0': 2, 'mlp_1': 2},
        feature_dim=feature_dim,
        device=device
    )
    
    # Set identity matrices for weights
    suite.aes['mlp_0'].decoder.weight.data = t.tensor([[1., 0.], [0., 1.]], device=device)
    suite.aes['mlp_1'].encoder.weight.data = t.tensor([[1., 0.], [0., 1.]], device=device)
    
    # Known activation patterns
    mock_topk_info = {
        'mlp_0': (
            t.tensor([[0, 1]], device=device),  # indices
            t.tensor([[1.0, 0.5]], device=device)  # values
        ),
        'mlp_1': (
            t.tensor([[0, 1]], device=device),  # indices
            t.tensor([[1.0, 0.5]], device=device)  # values
        )
    }
    suite.set_mock_topk_info(mock_topk_info)
    
    inputs = [MockInput(t.ones(1, feature_dim, device=device))]
    buffer = MockBuffer(inputs)
    
    scores = process_pair(
        suite=suite,
        buffer=buffer,
        up_name='mlp_0',
        down_name='mlp_1',
        num_batches=1,
        top_c=2,
        save_dir=None
    )
    
    dense_scores = scores.to_dense()
    
    # With identity matrices and these activations, we expect:
    # - Feature 0 upstream should have importance 1.0 to feature 0 downstream
    # - Feature 1 upstream should have importance 0.5 to feature 1 downstream
    expected_scores = t.tensor([
        [1.0, 0.0],
        [0.0, 0.5]
    ], device=device)
    
    print(f"Computed scores:\n{dense_scores}")
    print(f"Expected scores:\n{expected_scores}")
    print(f"Difference:\n{(dense_scores - expected_scores).abs()}")
    print(f"Maximum difference: {(dense_scores - expected_scores).abs().max().item():.8f}")
    
def test_different_dict_sizes():
    print("\n=== Testing different dictionary sizes ===")
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    
    suite = MockSCAESuite(
        submodule_names=['mlp_0', 'mlp_1'],
        dict_sizes={'mlp_0': 3, 'mlp_1': 5},  # Different sizes
        feature_dim=4,
        device=device
    )
    
    mock_topk_info = {
        'mlp_0': (
            t.tensor([[0, 1], [1, 2]], device=device),  # Repeated indices
            t.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)  # Zero values
        ),
        'mlp_1': (
            t.tensor([[0, 1], [2, 3]], device=device),
            t.tensor([[1.0, 1.0], [1.0, 1.0]], device=device)
        )
    }
    suite.set_mock_topk_info(mock_topk_info)
    
    inputs = [MockInput(t.randn(2, 4, device=device))]
    buffer = MockBuffer(inputs)
    
    scores = process_pair(
        suite=suite,
        buffer=buffer,
        up_name='mlp_0',
        down_name='mlp_1',
        num_batches=1,
        top_c=2,
        save_dir=None
    )
    
    dense_scores = scores.to_dense()
    print(f"Score matrix shape: {dense_scores.shape}")
    print(f"Score matrix:\n{dense_scores}")
    print(f"Contains zeros: {t.any(dense_scores == 0).item()}")
    print(f"Number of non-zero elements: {(dense_scores != 0).sum().item()}")

if __name__ == '__main__':
    test_valid_pairs()
    test_process_pair_simple()
    test_numerical_accuracy()
    test_different_dict_sizes()

# %%

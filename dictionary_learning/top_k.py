import einops
import torch as t
import torch.nn as nn
from abc import ABC, abstractmethod

from utils import set_seed
set_seed(42)


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoderTopK(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data /= 10 # TODO: make configurable, or use a better init
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(
        self,
        x: t.Tensor,
        return_topk: bool = False,
        return_preact: bool = False,
        n_threshold: int = 0,
    ):
        """Encode function supporting both [batch, d] and [batch, seq, d] inputs

        Args:
            x: Input tensor of shape [batch, d] or [batch, seq, d]
            return_topk: Whether to return top-k indices and values
            return_preact: Whether to return pre-activation values
            n_threshold: Number of additional threshold features to return

        Returns:
            Same as before, but matching input batch dimensions
        """
        orig_shape = x.shape
        if len(orig_shape) == 3:
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, orig_shape[-1])

        preact_BF = self.encoder(x)

        post_relu_feat_acts_BF = nn.functional.relu(preact_BF)
        post_topk = post_relu_feat_acts_BF.topk(
            self.k + n_threshold, sorted=True, dim=-1
        )

        top_acts_all = post_topk.values
        top_indices_all = post_topk.indices
        tops_acts_BK = post_topk.values[:, : self.k]
        top_indices_BK = post_topk.indices[:, : self.k]

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )
        

        # Reshape back to original batch dimensions if needed
        if len(orig_shape) == 3:
            encoded_acts_BF = encoded_acts_BF.reshape(
                orig_shape[0], orig_shape[1], -1
            )
            if return_topk:
                top_acts_all = top_acts_all.reshape(
                    orig_shape[0], orig_shape[1], -1
                )
                top_indices_all = top_indices_all.reshape(
                    orig_shape[0], orig_shape[1], -1
                )
            if return_preact:
                preact_BF = preact_BF.reshape(orig_shape[0], orig_shape[1], -1)

        if return_topk and return_preact:
            return encoded_acts_BF, preact_BF, top_acts_all, top_indices_all
        elif return_topk:
            return encoded_acts_BF, top_acts_all, top_indices_all
        elif return_preact:
            return encoded_acts_BF, preact_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        """Decode function supporting both [batch, d] and [batch, seq, d] inputs"""
        orig_shape = x.shape
        if len(orig_shape) == 3:
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, orig_shape[-1])

        decoded = self.decoder(x) + self.b_dec

        # Reshape back to original batch dimensions if needed
        if len(orig_shape) == 3:
            decoded = decoded.reshape(orig_shape[0], orig_shape[1], -1)

        return decoded

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    # https://github.com/EleutherAI/sparsify/blob/main/sparsify/sparse_coder.py
    @t.autocast(
        "cuda", dtype=t.bfloat16, enabled=t.cuda.is_bf16_supported()
    )
    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.weight.dtype).eps
        norm = t.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )

    def from_pretrained(path, k: int, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class CrosscoderTopK(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, n_outputs: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.n_outputs = n_outputs

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data /= 10 # TODO: make configurable, or use a better init
        self.encoder.bias.data.zero_()

        # Decoder weights [dict_size, n_outputs, activation_dim]
        # Initialize by repeating encoder weights for each output.
        self.decoder_weight = nn.Parameter(
            einops.repeat(
                self.encoder.weight.data.clone(),
                "d a -> d n a",
                n=self.n_outputs
            ).clone()
        )

        self.b_dec = nn.Parameter(t.zeros(n_outputs, activation_dim))

        self.set_decoder_norm_to_unit_norm()

    def encode(
        self,
        x: t.Tensor,
        return_topk: bool = False,
        return_preact: bool = False,
        n_threshold: int = 0,
    ):
        """Encode function supporting both [batch, d] and [batch, seq, d] inputs

        Args:
            x: Input tensor of shape [batch, d] or [batch, seq, d]
            return_topk: Whether to return top-k indices and values
            return_preact: Whether to return pre-activation values
            n_threshold: Number of additional threshold features to return

        Returns:
            Encoded activations, and optionally pre-activations and top-k info.
        """
        orig_shape = x.shape
        if len(orig_shape) == 3:
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, orig_shape[-1])

        preact_BF = self.encoder(x)
        post_relu_feat_acts_BF = nn.functional.relu(preact_BF)
        post_topk = post_relu_feat_acts_BF.topk(
            self.k + n_threshold, sorted=True, dim=-1
        )

        top_acts_all = post_topk.values
        top_indices_all = post_topk.indices
        tops_acts_BK = post_topk.values[:, : self.k]
        top_indices_BK = post_topk.indices[:, : self.k]

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        # Reshape back to original batch dimensions if needed
        if len(orig_shape) == 3:
            encoded_acts_BF = encoded_acts_BF.reshape(
                orig_shape[0], orig_shape[1], -1
            )
            if return_topk:
                top_acts_all = top_acts_all.reshape(
                    orig_shape[0], orig_shape[1], -1
                )
                top_indices_all = top_indices_all.reshape(
                    orig_shape[0], orig_shape[1], -1
                )
            if return_preact:
                preact_BF = preact_BF.reshape(orig_shape[0], orig_shape[1], -1)

        if return_topk and return_preact:
            return encoded_acts_BF, preact_BF, top_acts_all, top_indices_all
        elif return_topk:
            return encoded_acts_BF, top_acts_all, top_indices_all
        elif return_preact:
            return encoded_acts_BF, preact_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        """
        Decode dictionary features to [..., n_outputs, activation_dim].
        Input x: [batch, dict_size] or [batch, seq, dict_size]
        """
        orig_shape = x.shape
        input_dict_size = x.shape[-1]
        
        if input_dict_size != self.dict_size:
            raise ValueError(f"Input feature dimension {input_dict_size} does not match dictionary size {self.dict_size}")

        is_3d = len(orig_shape) == 3
        if is_3d:
            # Flatten batch and sequence dimensions
            x_flat = x.reshape(-1, input_dict_size)
        else:
            x_flat = x

        # x_flat: [batch_flat, dict_size]
        # self.decoder_weight: [dict_size, n_outputs, activation_dim]
        # self.b_dec: [n_outputs, activation_dim]
        
        decoded_flat = einops.einsum(
            x_flat, self.decoder_weight,
            "b d, d n a -> b n a"
        )
        decoded_flat = decoded_flat + self.b_dec # Broadcasting: [b, n, a] + [n, a]

        if is_3d:
            # Reshape back to [batch, seq, n_outputs, activation_dim]
            decoded = decoded_flat.reshape(
                orig_shape[0], orig_shape[1], self.n_outputs, self.activation_dim
            )
        else:
            # Output shape [batch, n_outputs, activation_dim]
            decoded = decoded_flat
            
        return decoded # [... n_outputs, activation_dim]

    @t.autocast(
        "cuda", dtype=t.bfloat16, enabled=t.cuda.is_bf16_supported()
    )
    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        # x_hat will have shape [..., n_outputs, activation_dim]
        x_hat = self.decode(encoded_acts_BF) 
        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts_BF

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder_weight.dtype).eps
        # Normalize along the activation_dim (last dimension)
        norm = self.decoder_weight.data.pow(2).sum([-1, -2], keepdim=True).sqrt()
        self.decoder_weight.data /= (norm + eps)

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        grad = self.decoder_weight.grad
        if grad is None:
            return

        data = self.decoder_weight.data
        
        # For each (dict_element, output_head), project grad vector onto data vector
        # grad is [dict_size, n_outputs, activation_dim]
        # data is [dict_size, n_outputs, activation_dim]
        
        # parallel_component[d, n] = sum_a (grad[d,n,a] * data[d,n,a])
        # Assumes data vectors are unit norm (due to set_decoder_norm_to_unit_norm)
        parallel_component = einops.einsum(
            grad, data,
            "d n a, d n a -> d n" 
        )
        
        # grad[d,n,a] -= parallel_component[d,n] * data[d,n,a]
        self.decoder_weight.grad -= einops.einsum(
            parallel_component, data,
            "d n, d n a -> d n a"
        )

    @classmethod
    def from_pretrained(cls, path: str, k: int, n_outputs: int, device=None, **kwargs) -> "CrossEncoderTopK":
        """
        Load a pretrained CrossEncoderTopK from a file.
        The state_dict is expected to match the CrossEncoderTopK structure.
        """
        state_dict = t.load(path, map_location=device if device else 'cpu')
        
        # Infer activation_dim and dict_size from encoder weights
        # encoder.weight shape is [dict_size, activation_dim]
        dict_size = state_dict["encoder.weight"].shape[0]
        activation_dim = state_dict["encoder.weight"].shape[1]
        
        model = cls(activation_dim, dict_size, k, n_outputs, **kwargs)
        model.load_state_dict(state_dict)
        
        if device is not None:
            model.to(device)
        return model

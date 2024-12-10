- Write code to obtain top-c most important upstream features
- Currently we just optimize the vanilla-active feature for good approx. It's too expensive to optimize *all* downstream features, but it might make sense to randomly select a small number of non-active features per input and optimize these.
- Deal with embed/unembed
- Make it so that e.g. mlp_3 sees attn_3
- Worry about layernorm
- Decide on 3 losses
- Account for biases!

contributions = down_encoder @ up_decoder @ up_feature_acts
[batch, n_up, n_down]

upstream_features = contributions[:,:,69].mean(0).topk(5).indices  # is a list of 5 upstream features


- Obtain top-c most important upstream features
- Currently we just optimize the vanilla-active feature for good approx. It's too expensive to optimize *all* downstream features, but it might make sense to randomly select a small number of non-active features per input and optimize these.
- Deal with unembed

- Worry about layernorm - layernorm bias argh
- Decide on 3 losses
- Currently we just optimize the vanilla-active feature for good approx. It's too expensive to optimize *all* downstream features, but it might make sense to randomly select a small number of non-active features per input and optimize these.
- Embed/unembed?

- Decide on 3 losses
 -rename inputs/src to be consistent
 - More informative errors (e.g. "if no repo_id_in provided, you must provide submodule_configs" etc)
 - Implement patched CE and add logging step



 Note: to download connections_100.pkl from google drive, run this:
 gdown https://drive.google.com/uc?id=1IW3d-t0EuaLR4rSM7WQQqTPESp2nFGJH
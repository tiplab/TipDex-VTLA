"""Native Orbax checkpoint and the exact 9309 input/output transforms."""
import json
from types import SimpleNamespace


def load_backbone(bundle, inference_config):
    import jax.numpy as jnp
    from openpi.vtla.model import Pi05VTLAConfig
    from openpi.models.model import restore_params
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.octopus_policy import OctopusInputs, OctopusOutputs, ClipNormalizedInputs
    from openpi.shared.normalize import load
    from openpi import transforms as t
    args = json.loads(bundle.path('backbone/config.json').read_text())
    args['loss_action_dims'] = tuple(args['loss_action_dims'])
    args['tactile_fusion'] = inference_config.tactile_fusion
    config = Pi05VTLAConfig(**args)
    if (config.action_horizon,config.action_dim,config.tactile_layout) != (30,48,'f1_364'):
        raise ValueError('Model bundle has an incompatible action or sensor layout')
    model = config.load(restore_params(bundle.path('backbone/params'), dtype=jnp.bfloat16))
    norm = load(bundle.path('backbone/assets/demo01_egg'))
    normalize = t.Normalize(norm, use_quantiles=True)
    tokenizer = PaligemmaTokenizer(config.max_token_len, model_path=bundle.path('tokenizer/paligemma_tokenizer.model'))
    inputs = t.compose([t.InjectDefaultPrompt('Move the egg into the box.'), OctopusInputs(),
                        normalize, ClipNormalizedInputs(), t.ResizeImages(224,224),
                        t.TokenizePrompt(tokenizer, discrete_state_input=True), t.PadStatesAndActions(48)])
    outputs = t.compose([t.Unnormalize(norm, use_quantiles=True), OctopusOutputs(action_dim=13)])
    return SimpleNamespace(_model=model,_input_transform=inputs,_output_transform=outputs), normalize

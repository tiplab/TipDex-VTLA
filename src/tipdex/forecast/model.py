"""Recurrent tactile decoder sharing the policy encoder."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class SharedForecastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048,64), nn.GELU())
        self.history = nn.GRU(256+19+13+2,128,batch_first=True)
        self.initial = nn.Sequential(nn.Linear(192,128),nn.Tanh())
        self.future = nn.GRU(13+2,128,batch_first=True)
        self.output = nn.Linear(128,364)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self,context,touch,state,commands,history_times,future_commands,future_times,
                last_normalized,output_scale):
        dt = torch.diff(history_times,dim=1,prepend=history_times[:,:1])
        values = torch.cat((touch,state,commands,history_times[...,None],dt[...,None]),dim=-1)
        _,hidden = self.history(values)
        hidden = self.initial(torch.cat((hidden[0],self.context(context)),dim=-1))[None]
        fdt = torch.diff(future_times,dim=1,prepend=torch.zeros_like(future_times[:,:1]))
        decoded,_ = self.future(torch.cat((future_commands,future_times[...,None],fdt[...,None]),dim=-1),hidden)
        return torch.clamp(last_normalized[:,None]+self.output(decoded)*output_scale,0.,1.)

    def initialize_from_legacy(self, state):
        """Reuse compatible decoder weights; learn a new encoded-touch input."""
        current = self.state_dict()
        for key,value in state.items():
            if key != 'history.weight_ih_l0':
                if value.shape != current[key].shape:
                    raise ValueError('Incompatible legacy decoder weight')
                current[key] = value.clone()
        # State, command and timestamp columns retain their meanings.
        current['history.weight_ih_l0'][:,256:] = state['history.weight_ih_l0'][:,364:]
        self.load_state_dict(current)


class SharedRuntimeForecaster:
    def __init__(self, directory, encoder, baseline_provider, *, horizon=30):
        self.horizon = horizon
        root = Path(directory)
        self.metadata = json.loads((root/'contract.json').read_text())
        if self.metadata['shared_encoder_parameters_sha256'] != encoder.parameters_sha256:
            raise ValueError('Forecaster was trained with different tactile encoder weights')
        for name in ('weights.pt','normalization.npz'):
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != self.metadata['sha256'][name]:
                raise ValueError('Shared forecaster artifact hash mismatch')
        self.encoder, self.baseline_provider = encoder, baseline_provider
        with np.load(root/'normalization.npz') as data:
            self.norm = {key:data[key] for key in data.files}
        self.model = SharedForecastHead().eval()
        self.model.load_state_dict(torch.load(root/'weights.pt',weights_only=True,map_location='cpu'))

    def predict(self,context,touch_history,state_history,command_history,history_times,
                future_commands,future_times):
        args = [np.asarray(value,np.float32) for value in
                (context,touch_history,state_history,command_history,history_times,future_commands,future_times)]
        for value,shape in zip(args,[(2048,),(16,364),(16,19),(16,13),(16,),(self.horizon,13),(self.horizon,)],strict=True):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Expected finite shared-forecaster input {shape}')
        cx,h,s,a,ht,fa,ft = args
        if not (np.all(np.diff(ht)>0) and abs(float(ht[-1]))<1e-6 and np.all(ft>0) and np.all(np.diff(ft)>0)):
            raise ValueError('Shared forecast timestamps are not causal')
        tokens = self.encoder.encode(h, self.baseline_provider())
        n = self.norm
        values = [cx,(tokens-n['feature_center'])/n['feature_scale'],
            (s-n['state_center'])/n['state_scale'],(a-n['action_center'])/n['action_scale'],
            ht,(fa-n['action_center'])/n['action_scale'],ft,h[-1]/65535.,n['touch_scale']/65535.]
        with torch.inference_mode():
            result = self.model(*(torch.from_numpy(np.array(v,dtype=np.float32,copy=True)[None]) for v in values)).numpy()[0]
        if result.shape != (self.horizon,364) or not np.isfinite(result).all():
            raise ValueError('Invalid shared forecast output')
        return result

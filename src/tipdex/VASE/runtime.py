"""Visual reliability from the touch-independent visual-language prefix."""
import math
from .head import NumpyVASEHead


class VisualReliability:
    def __init__(self, directory, *, config):
        self.head = NumpyVASEHead.load(directory)
        self.config = config
        self.reset()

    def reset(self):
        self.latest = None

    def describe(self):
        return dict(enabled=True, status='ready', affects_actions=True,
                    weights_sha256=self.head.metadata['weights_sha256'],
                    formula='q_v = sigmoid(h(Pool(c_t)))',
                    refresh='slow observation; cached by session and chunk')

    def update(self, features, session_id, chunk_id, observation_time):
        score = self.head.predict(features)
        if score.shape != (1,) or not math.isfinite(float(observation_time)):
            raise ValueError('VASE expects one observation with a finite timestamp')
        self.latest = dict(session_id=session_id, chunk_id=chunk_id,
                           q_v=float(score[0]), vase_observation_time=float(observation_time))
        return self.response(session_id, chunk_id, observation_time)

    def response(self, session_id, chunk_id, at_time):
        value = self.latest
        if value is None or (value['session_id'], value['chunk_id']) != (session_id, chunk_id):
            raise ValueError('VASE has no score for the current session/chunk')
        age = float(at_time)-value['vase_observation_time']
        if not math.isfinite(age) or age < 0:
            raise ValueError('VASE response precedes the observation')
        return {k:v for k,v in value.items() if k not in ('session_id','chunk_id')} | dict(
            q_oper=self.config.visual_offset+self.config.visual_scale*value['q_v'],
            vase_context_age_s=age, vase_status='ok')

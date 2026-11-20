"""Slow/fast updates, executor-owned protection and forecast renewal."""
from collections import deque
import time
import threading
import numpy as np
from .schedule import ActionSchedule, Plan


class ClosedLoop:
    def __init__(self, policy, *, clock=time.monotonic):
        self.policy, self.clock = policy, clock
        self.lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.committed_intervals = []
        self.schedule = None
        self.states = deque(maxlen=1024)
        self.session = policy.session_id
        self.chunk = None
        self.last = {}

    def reset(self):
        self.session = self.policy.reset()
        self.schedule = None
        self.states.clear()
        self.committed_intervals.clear()
        self.chunk = None
        self.last = {}
        return self.session

    def feed_state(self, stamp, state):
        value = np.array(state, np.float32, copy=True)
        with self.state_lock:
            if (value.shape != (19,) or not np.isfinite(value).all() or not np.isfinite(stamp)
                    or stamp > self.clock() or (self.states and stamp <= self.states[-1][0])):
                raise ValueError('State must have 19 finite coordinates and increasing causal timestamps')
            self.states.append((float(stamp),value))

    def feed_frames(self, frames):
        now = self.clock()
        for frame in frames:
            acq, receipt = frame['acquisition_time'], frame['timestamp']
            if not np.isfinite([acq,receipt]).all() or not 0 <= acq <= receipt <= now:
                raise ValueError('Tactile acquisition/receipt times must be causal')
        self.policy.feed_frames(self.session,frames)

    def acknowledge(self, packets):
        return self.policy.acknowledge(self.session,packets)

    def commit(self, indices, *, chunk_id=None):
        # Commit BEFORE placing commands in a nonreplaceable driver queue.
        with self.lock:
            if self.schedule is None or (chunk_id is not None and chunk_id != self.chunk):
                raise ValueError('No matching action plan for commit')
            indices = np.asarray(indices)
            if indices.ndim != 1 or (indices.size and (not np.issubdtype(indices.dtype,np.integer)
                    or np.any(indices<0) or np.any(indices>=30))):
                raise ValueError('Commit indices must be integers in [0,30)')
            self.schedule.commit(indices.astype(int))
            for index in indices.astype(int):
                start=float(self.schedule.plan.times[index])
                self.committed_intervals.append((start,start+1/self.policy.fps))

    def _uncommitted(self, times, now):
        self.committed_intervals = [(a,b) for a,b in self.committed_intervals if b>now]
        mask=np.ones(len(times),bool)
        for start,end in self.committed_intervals:
            mask &= ~((times < end-1.e-9) & (times+1/self.policy.fps > start+1.e-9))
        return mask

    def slow_update(self, images, state, stamp):
        result = self.policy.slow(self.session,images,state,stamp)
        with self.lock:
            self.chunk = result['chunk_id']
            self.schedule = None
        # A new chunk starts a new time grid; old forecasts remain verifiable
        # against ACKs, but can never rewrite the executor's previous plan.
        return result

    def _history(self, anchor):
        times = anchor+np.arange(-15,1)/30.
        with self.state_lock:
            states = list(self.states)
        if not states:
            raise ValueError('State history is missing')
        stamps = np.array([s[0] for s in states])
        indices = np.searchsorted(stamps,times+1.e-9,side='right')-1
        if np.any(indices < 0) or np.any(times-stamps[indices] > .1):
            raise ValueError('State history is incomplete or stale')
        with self.policy.sensor_lock:
            fields,_ = self.policy.stream.snapshot(anchor)
        if not fields['tactile_sample_valid'].all():
            raise ValueError('Forecast requires a complete valid tactile history')
        commands = np.stack([self.policy.forecast.verifier.execution.value_at(t) for t in times])
        return dict(history_times=times.tolist(),touch_history=fields['tactile'][0,:,0,:].tolist(),
                    state_history=np.array([states[i][1] for i in indices]).tolist(),
                    command_history=commands.tolist())

    def issue_forecast(self):
        if self.schedule is None:
            raise ValueError('No accepted plan')
        anchor = self.clock()
        history = self._history(anchor)
        targets = anchor+np.arange(1,self.policy.config.forecast_horizon+1)/30.
        future = self.schedule.plan.times > anchor
        times = np.r_[anchor,self.schedule.plan.times[future]]
        commands = np.vstack((history['command_history'][-1],self.schedule.plan.commands[future]))
        sampled = commands[np.searchsorted(times,targets,side='right')-1]
        request = dict(session_id=self.session,chunk_id=self.chunk,
            plan_id=f'{self.session}:{self.chunk}:{self.schedule.plan.revision}',anchor_time=anchor,
            **history,target_times=targets.tolist(),future_commands=sampled.tolist(),
            command_times=times.tolist(),scheduled_commands=commands.tolist())
        return self.policy.forecast.predict(request)

    def fast_update(self, stamp, state, *, executed_steps=0, allowed=None):
        if self.chunk is None:
            raise ValueError('Run slow_update first')
        allowed = np.ones(30,bool) if allowed is None else np.asarray(allowed)
        if allowed.shape != (30,) or allowed.dtype != bool:
            raise ValueError('allowed must be a boolean mask with 30 entries')
        before = self.clock()
        with self.lock:
            chunk = self.chunk
            times = self.policy.observation_time+np.arange(30)/self.policy.fps
            mask = allowed & self._uncommitted(times,before)
            if self.schedule is not None:
                mask = self.schedule.mask(before,mask)
        result = self.policy.fast(self.session,chunk,stamp,executed_steps,
                                  feedback_state=state,replaceable=mask)
        accepted = self.clock()
        with self.lock:
            if self.chunk != chunk:
                raise ValueError('Chunk changed during fast update')
            mask &= self._uncommitted(times,accepted)
            if self.schedule is None:
                self.schedule = ActionSchedule(Plan(self.session,0,accepted,times,np.asarray(result['actions'])))
                changed = mask & (times > accepted) & (np.arange(30) >= result['valid_from'])
                self.schedule.commit(np.flatnonzero(~changed))
            else:
                changed = self.schedule.replace(result['actions'],accepted,mask & (np.arange(30)>=result['valid_from']))
            accepted_plan = self.schedule.plan
        forecast = None
        reason = None
        if changed.any():
            try:
                forecast = self.issue_forecast()
            except ValueError as error:
                # No evidence is invented; the verifier falls back to e=0.
                reason = str(error)
        self.last = {**result,'replaced':np.flatnonzero(changed).tolist(),
                     'actions':accepted_plan.commands.tolist(),
                     'action_times':accepted_plan.times.tolist(),
                     'forecast':forecast,'forecast_unavailable_reason':reason}
        return self.last

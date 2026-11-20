"""Causal forecast issuance, execution evidence and immutable action timing.

This module has no model, ROS or robot dependencies. Adapters must report actual
accepted command events, not proposed plans or future measured robot states.
"""
from dataclasses import dataclass
from collections import deque
import threading
import math
import numpy as np


def immutable(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite protocol value')
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Plan:
    session: str
    revision: int
    created_at: float
    times: np.ndarray
    commands: np.ndarray

    def __post_init__(self):
        times, commands = immutable(self.times), immutable(self.commands)
        if (times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0)
            or commands.ndim != 2 or len(commands) != len(times)
            or not self.session or not math.isfinite(self.created_at)):
            raise ValueError('Invalid timestamped plan')
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'commands', commands)

    def at(self, stamp):
        index = np.searchsorted(self.times, stamp, side='right')-1
        if index < 0:
            raise ValueError('Plan does not cover requested time')
        return self.commands[index]


class ActionSchedule:
    """Only explicitly replaceable future slots can be changed; times never move."""
    def __init__(self, plan):
        self.plan = plan
        self.lock = threading.RLock()
        self.committed = np.zeros(len(plan.times), bool)

    def commit(self, indices):
        with self.lock:
            indices = np.asarray(indices)
            if indices.ndim != 1 or (indices.size and (not np.issubdtype(indices.dtype,np.integer)
                    or np.any(indices < 0) or np.any(indices >= len(self.committed)))):
                raise ValueError('Invalid commit indices')
            self.committed[indices.astype(int)] = True

    def mask(self, now, allowed):
        with self.lock:
            allowed = np.asarray(allowed, bool)
            if allowed.shape != self.committed.shape:
                raise ValueError('Replacement mask shape mismatch')
            return allowed & ~self.committed & (self.plan.times > now)


    def replace(self, proposed, now, allowed):
        with self.lock:
            proposed = immutable(proposed)
            if proposed.shape != self.plan.commands.shape:
                raise ValueError('Replacement command shape mismatch')
            mask = self.mask(now, allowed)
            commands = np.where(mask[:, None], proposed, self.plan.commands)
            self.plan = Plan(self.plan.session, self.plan.revision+1, now, self.plan.times, commands)
            return mask.copy()

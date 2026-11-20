"""Causal evidence for acknowledged SDK commands, not achieved physical poses.

Missing events, a driver restart, failed writes or gaps fail closed for the
inference session. Commands are compared in radians to the complete conditioned
schedule, including intermediate SDK outputs. Smoothing that changes a command
therefore makes that forecast ineligible; it is not silently ignored.
"""
from collections import deque
from pathlib import Path
import threading
import time
import numpy as np


class ExecutionEvidence:
    def __init__(self, host_boot_id=None):
        self.host_boot_id=host_boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.lock=threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.events={role:deque(maxlen=4096) for role in ('arm','hand')}
            self.last={}
            self.failure=None

    def ingest(self, packet, now=None):
        now=time.monotonic() if now is None else float(now)
        with self.lock:
            if self.failure is not None:
                return
            try:
                role=packet['role'];width={'arm':7,'hand':6}[role]
                values=np.asarray(packet['positions'],np.float64)
                stamp=float(packet['ack_at']);sequence=packet['sequence']
                if (packet['kind']!='sdk_write_ack' or packet['schema']!=1
                    or packet['host_boot_id']!=self.host_boot_id or packet['units']!='radians'
                    or packet['success'] is not True or type(sequence) is not int or sequence<1
                    or values.shape!=(width,) or not np.isfinite(values).all()
                    or not np.isfinite(stamp) or not 0<=stamp<=now):
                    raise ValueError('invalid_or_failed_sdk_ack')
                old=self.last.get(role)
                if old is not None:
                    if packet['driver_boot_id']!=old['driver_boot_id']:
                        raise ValueError('driver_restarted')
                    if sequence!=old['sequence']+1 or packet['lost']!=old['lost']:
                        raise ValueError('execution_ack_event_lost')
                    if not 0<stamp-old['ack_at']<=.05:
                        raise ValueError('execution_ack_clock_or_gap')
                self.last[role]=dict(packet)
                self.events[role].append((stamp,values.copy()))
            except (KeyError,TypeError,ValueError) as error:
                self.failure=str(error)

    def value_at(self, stamp, *, max_age=.05):
        with self.lock:
            if self.failure:
                raise ValueError(self.failure)
            values=[]
            for role in ('arm','hand'):
                row=next((r for r in reversed(self.events[role]) if r[0]<=stamp),None)
                if row is None or stamp-row[0]>max_age:
                    raise ValueError('missing_causal_sdk_command')
                values.extend(row[1])
            return np.asarray(values,np.float64)

    def history_matches(self, times, values):
        try:
            return all(np.allclose(self.value_at(t),v,atol=1e-5,rtol=0)
                       for t,v in zip(times,values,strict=True))
        except (ValueError,TypeError):
            return False

    def schedule_matches(self, start, end, schedule_times, schedule_values):
        """Exact event-time comparison; no allowance for a changed command."""
        times=np.asarray(schedule_times,np.float64)
        values=np.asarray(schedule_values,np.float64)
        if (times.ndim!=1 or not len(times) or values.shape!=(len(times),13)
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or times[0]>start or np.any(np.diff(times)<=0) or end<start):
            return False,'invalid_conditioned_schedule'
        with self.lock:
            if self.failure:
                return False,self.failure
            for role,sl in (('arm',slice(0,7)),('hand',slice(7,13))):
                rows=self.events[role]
                if not rows or rows[0][0]>start or rows[-1][0]<end:
                    return False,'incomplete_execution_watermark'
                before=next((r for r in reversed(rows) if r[0]<=start),None)
                if before is None or start-before[0]>.05:
                    return False,'missing_execution_anchor'
                # Compare both actual changes and expected transitions. Checking
                # only the endpoint would miss a changed-and-restored command.
                probes=[start,end]+[t for t,_ in rows if start<t<end]
                probes += [float(t) for t in times if start<t<end]
                for at in probes:
                    actual=next(r[1] for r in reversed(rows) if r[0]<=at)
                    expected=values[np.searchsorted(times,at,side='right')-1,sl]
                    if not np.allclose(actual,expected,atol=1e-5,rtol=0):
                        return False,'conditioned_command_differs_from_sdk_output'
            return True,'acknowledged_command_schedule_matched'

    def describe(self):
        with self.lock:
            return dict(failure=self.failure,
                received={k:len(v) for k,v in self.events.items()},
                semantics='SDK accepted/written commands; not proof of achieved physical pose')

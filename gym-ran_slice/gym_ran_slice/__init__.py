from gym.envs.registration import register

from .ran_slice import RanSlice

register(
    id='RanSlice-v1',
    entry_point='gym-ran_slice:RanSlice'
)
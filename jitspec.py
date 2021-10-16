from numba.experimental import jitclass
from numba import int32, float32, typeof
import numpy as np

# print out truth table and outputs
spec = [
    ('nInputs', int32),
    ('nHidden', int32),
    ('nOutputs', int32),

    ('inputs', float32[:]),
    ('thetaj', float32[:]),
    ('hj', float32[:]),
    ('thetai', float32[:]),
    ('Fi', float32[:]),
    ('Ti', float32[:]),

    ('Esum', float32), 

    ('trainingInputs', float32[:,:]),
    ('trainingOutputs', float32[:,:]),
    ('trainingPos', int32),

    ('omegaj', float32[:]),
    ('psij', float32[:]),

    ('omegai', float32[:]),
    ('psii', float32[:]),

    ('partialEwkj', float32[:,:]),
    ('partialEwji', float32[:,:]),

    ('delwkj', float32[:,:]),
    ('delwji', float32[:,:]),

    ('weights', float32[:,:,:])
]


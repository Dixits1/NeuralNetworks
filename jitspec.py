from numba.experimental import jitclass
from numba import int32, float32, typeof

# print out truth table and outputs
spec = [
    ('nInputs', int32),
    ('nHidden', int32),

    ('inputs', float32[:]),
    ('thetaj', float32[:]),
    ('hj', float32[:]),
    ('theta0', float32),
    ('F0', float32),
    ('output', float32),

    ('trainingInputs', float32[:]),
    ('trainingOutputs', float32[:,:]),
    ('trainingPos', int32),

    ('weights', float32[:,:,:])

]


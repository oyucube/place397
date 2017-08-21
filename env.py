import numpy as np
from chainer import cuda

import socket
xp = np
test = "cpu"
if socket.gethostname() == "naruto":
    xp = cuda.cupy
    test = "gpu"

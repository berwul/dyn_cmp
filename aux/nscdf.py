# Copyright (c) 2026, ABB Schweiz AG
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import pathlib

import numpy as np
from numba import jit


@jit(nopython=True)
def sdf_compute(X, W_0, W_1, W_2, W_3, b_0, b_1, b_2, b_3):
    h = X
    h = np.maximum(W_0 @ h + b_0, 0.0)
    h = np.maximum(W_1 @ h + b_1, 0.0)
    h = np.maximum(W_2 @ h + b_2, 0.0)
    return W_3 @ h + b_3


class nSCDF:

    def __init__(self, coordinate_offset, Ws, bs, Ws_sc=None, bs_sc=None):
        self.dims = np.empty((0, ))
        self.Ws = Ws
        self.bs = bs
        self.include_sc = Ws_sc is not None
        self.Ws_sc = Ws_sc
        self.bs_sc = bs_sc
        self.margin = 0.0
        self.coord_offset = coordinate_offset

    @classmethod
    def from_saved(cls):
        data = np.load(pathlib.Path(__file__).parent / "data" / "model_params.npz")
        Ws = [data[f"W_{i}"] for i in range(4)]
        bs = [data[f"b_{i}"] for i in range(4)]
        coordinate_offest = np.zeros(3, )
        return cls(coordinate_offest, Ws, bs, Ws_sc=None, bs_sc=None)

    def set_dims(self, dims):
        self.dims = dims.copy()
        self.dims[:, :-1] = self.dims[:, :-1] - self.coord_offset

    def set_margin(self, margin):
        self.margin = margin

    def sdf(self, q):
        m, k = self.dims.shape
        c = q.size
        X = np.full((k + c, m), 0.0)
        for i in range(m):
            X[:k, i] = self.dims[i]
            X[k:, i] = q
        sd = sdf_compute(X, *self.Ws, *self.bs).min()
        if self.include_sc:
            sd_sc, = sdf_compute(q[:, None], *self.Ws_sc, *self.bs_sc).ravel()
            sd = min(sd, sd_sc)
        return sd - self.margin

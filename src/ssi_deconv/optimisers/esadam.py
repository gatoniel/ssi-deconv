"""
Copied from https://github.com/royerlab/ssi-code/blob/master/ssi/optimisers/esadam.py

Copyright 2020 Hirofumi Kobayashi, Ahmet Can Solak, Joshua Batson, Loic A. Royer

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
"""

import torch
from torch.optim import Adam


class ESAdam(Adam):
    r"""Implements a modifified version of the Adam algorithm that adds noise to the
    the .

    """

    def __init__(self, params, start_noise_level=0.001, **kwargs):
        super().__init__(params, **kwargs)

        self.start_noise_level = start_noise_level
        self.step_counter = 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = super().step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad.data
                if grad.is_sparse:
                    continue

                step_size = group["lr"]

                p.data += (
                    step_size
                    * (self.start_noise_level / (1 + self.step_counter))
                    * (torch.randn_like(p.data))
                )

        self.step_counter += 1

        return loss

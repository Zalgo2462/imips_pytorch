from typing import List

import torch


class MSIndexProposal(torch.nn.Module):
    def __init__(self):
        super(MSIndexProposal, self).__init__()
        self._ksizes = [8 * (i + 1) for i in range(0, 5, 1)]
        self._scale_weights = [4 ** i for i in range(4, -1, -1)]
        self._one_kernels = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(1, 1, ksize, ksize, dtype=torch.float), False) for ksize in self._ksizes
        ])
        self._idx_kernels = torch.nn.ParameterList([
            torch.nn.Parameter(x, False) for x in MSIndexProposal.gen_idx_kernels(self._ksizes)
        ])

    @staticmethod
    def gen_idx_kernels(ksizes: List[int]) -> List[torch.Tensor]:
        kernels = []
        for sz in ksizes:
            grid = torch.stack(torch.meshgrid([
                torch.arange(sz, dtype=torch.float),
                torch.arange(sz, dtype=torch.float)
            ]))
            kernels.append(grid.unsqueeze_(1))
        return kernels

    def forward(self, scores: torch.Tensor):
        scores = torch.nn.functional.relu(scores)
        indices = []
        soft_scores = []
        for i in range(len(self._ksizes)):
            maxes = torch.nn.functional.max_pool2d(
                scores,
                kernel_size=(self._ksizes[i], self._ksizes[i]),
                stride=(self._ksizes[i], self._ksizes[i]),
                ceil_mode=True,
            )
            maxes = torch.nn.functional.conv_transpose2d(
                maxes,
                self._one_kernels[i].expand(scores.shape[1], -1, -1, -1),
                stride=self._ksizes[i],
                groups=scores.shape[1]
            )[:, :, :scores.shape[2], :scores.shape[3]].clamp_min_(1e-6)

            exp = torch.zeros(
                scores.shape[0],
                scores.shape[1],
                scores.shape[2] + (self._ksizes[i] - scores.shape[2] % self._ksizes[i]),
                scores.shape[3] + (self._ksizes[i] - scores.shape[3] % self._ksizes[i]),
                dtype=scores.dtype,
                device=scores.device
            )

            exp[:, :, :scores.shape[2], :scores.shape[3]] = torch.exp(3 * scores / maxes)

            sum_exp = torch.nn.functional.conv2d(
                exp,
                self._one_kernels[i].expand(exp.shape[1], -1, -1, -1),
                stride=self._ksizes[i],
                groups=exp.shape[1]
            ).clamp_min_(1e-6)

            indices_i = torch.nn.functional.conv2d(
                exp,
                self._idx_kernels[i].repeat(exp.shape[1], 1, 1, 1),
                stride=self._ksizes[i],
                groups=exp.shape[1]
            )
            indices_i = indices_i / sum_exp.repeat_interleave(2, dim=1)

            grid_idx = torch.stack(torch.meshgrid([
                torch.arange(indices_i.shape[2], dtype=indices_i.dtype, device=indices_i.device),
                torch.arange(indices_i.shape[3], dtype=indices_i.dtype, device=indices_i.device)
            ]))
            grid_idx *= self._ksizes[i]

            grid_idx = grid_idx.unsqueeze_(0).repeat(
                scores.shape[1], 1, 1, 1
            ).view(
                1, scores.shape[1] * 2, indices_i.shape[2], indices_i.shape[3]
            ).squeeze_()

            indices_i += grid_idx

            # multiply in scores against exp now that we are done using it for indices and sum_exp
            exp[:, :, :scores.shape[2], :scores.shape[3]] *= scores
            soft_scores_i = torch.nn.functional.conv2d(
                exp,
                self._one_kernels[i].expand(scores.shape[1], -1, -1, -1),
                stride=self._ksizes[i],
                groups=scores.shape[1]
            )
            soft_scores_i = soft_scores_i / sum_exp

            indices.append(indices_i)
            soft_scores.append(soft_scores_i)
        return indices, soft_scores

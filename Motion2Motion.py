import torch
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F

from utils.base import logger
from utils.joint_names import projection_masking


class Motion2Motion:
    def __init__(
        self,
        mode="retargeting",
        noise_sigma=1.0,
        coarse_ratio=0.2,
        coarse_ratio_factor=6,
        pyr_factor=0.75,
        num_stages_limit=-1,
        device="cuda:0",
        silent=False,
    ):
        """
        Motion2Motion main constructor
        Args:
            device : str = 'cuda:0', default device.
            silent : bool = False, whether to mute the output.
        """
        self.device = torch.device(device)
        self.silent = silent

        self.mode = mode

    def _get_pyramid_lengths(self, final_len, coarse_ratio, pyr_factor):
        """
        Get a list of pyramid lengths using given target length and ratio
        """
        lengths = [int(np.round(final_len * coarse_ratio))]
        while lengths[-1] < final_len:
            lengths.append(int(np.round(lengths[-1] / pyr_factor)))
            if lengths[-1] == lengths[-2]:
                lengths[-1] += 1
        lengths[-1] = final_len

        return lengths

    def _get_target_pyramid(
        self, target, coarse_ratio, pyr_factor, num_stages_limit=-1
    ):
        """
        Reads a target motion(s) and create a pyraimd out of it. Ordered in increatorch.sing size
        """
        lengths = []
        min_len = 10000

        for i in range(len(target)):
            new_length = self._get_pyramid_lengths(
                len(target[i].motion_data), coarse_ratio, pyr_factor
            )
            min_len = min(min_len, len(new_length))

            if num_stages_limit != -1:
                new_length = new_length[:num_stages_limit]
            lengths.append(new_length)

        # from
        min_len = len(
            self._get_pyramid_lengths(self.syn_length, coarse_ratio, pyr_factor)
        )
        self.clear_index = {}
        for i in range(min_len):
            self.clear_index[i + 1] = []
        for i in range(len(target)):
            if len(lengths[i]) < min_len:
                self.clear_index[len(lengths[i])].append(i)

        # to
        for i in range(len(target)):
            lengths[i] = lengths[i][-min_len:]

        self.num_target = len(target)
        self.pyraimd_lengths_target = lengths
        target_pyramid = [[] for _ in range(len(lengths[0]))]

        for step in range(len(lengths[0])):
            for i in range(len(target)):
                try:
                    length = lengths[i][step]
                    target_pyramid[step].append(
                        target[i].sample(size=length).to(self.device)
                    )
                except:
                    print(f"{i}-th sequence does not with step {step + 1}")

        if not self.silent:
            for i in range(len(target_pyramid)):
                print(
                    f"Number of clips in target pyramid {i} is {len(target_pyramid[i])}, ranging {[[tgt.min(), tgt.max()] for tgt in target_pyramid[i]]}"
                )

        return target_pyramid

    def _get_source_pyramid(
        self, target, coarse_ratio, pyr_factor, num_stages_limit=-1
    ):
        """
        Reads a target motion(s) and create a pyraimd out of it. Ordered in increatorch.sing size
        """
        self.num_source = len(target)
        lengths = []
        min_len = 10000

        for i in range(len(target)):
            self.syn_length = len(target[i].motion_data)
            new_length = self._get_pyramid_lengths(
                self.syn_length, coarse_ratio, pyr_factor
            )
            min_len = min(min_len, len(new_length))
            if num_stages_limit != -1:
                new_length = new_length[:num_stages_limit]
            lengths.append(new_length)

        for i in range(len(target)):
            lengths[i] = lengths[i][-min_len:]
        self.pyraimd_lengths_source = lengths

        target_pyramid = [[] for _ in range(len(lengths[0]))]

        for step in range(len(lengths[0])):
            for i in range(len(target)):
                length = lengths[i][step]
                target_pyramid[step].append(
                    target[i].sample(size=length).to(self.device)
                )

        if not self.silent:
            for i in range(len(target_pyramid)):
                print(
                    f"Number of clips in target pyramid {i} is {len(target_pyramid[i])}, ranging {[[tgt.min(), tgt.max()] for tgt in target_pyramid[i]]}"
                )

        return target_pyramid

    def _get_initial_motion(self, init_length, noise_sigma):
        """
        Prepare the initial motion for optimization
        """
        initial_motion = F.interpolate(
            torch.cat(
                [self.target_pyramid[0][i] for i in range(self.num_target)], dim=-1
            ),
            size=init_length,
            mode="linear",
            align_corners=True,
        )

        if noise_sigma > 0:
            initial_motion_w_noise = (
                initial_motion + torch.randn_like(initial_motion) * noise_sigma
            )
            initial_motion_w_noise = torch.fmod(initial_motion_w_noise, 1.0)
        else:
            initial_motion_w_noise = initial_motion

        if not self.silent:
            print("Initial motion:", initial_motion.min(), initial_motion.max())
            print(
                "Initial motion with noise:",
                initial_motion_w_noise.min(),
                initial_motion_w_noise.max(),
            )

        return initial_motion_w_noise

    def run(
        self,
        target,
        source_motion,
        criteria,
        num_frames,
        num_steps,
        noise_sigma,
        patch_size,
        coarse_ratio,
        pyr_factor,
        matching_alpha,
        ext=None,
        debug_dir=None,
    ):
        """
        generation function
        Args:
            mode             : - string = 'x?', generate x times longer frames results
                             : - int, specifying the number of times to generate
            noise_sigma      : float = 1.0, random noise.
            coarse_ratio     : float = 0.2, ratio at the coarse level.
            pyr_factor       : float = 0.75, pyramid factor.
            num_stages_limit : int = -1, no limit.

            <new feature>:
                source_motion: source motion data
        """
        if target[0].motion_data.matching_mask is not None:
            self.matching_mask_target = target[0].motion_data.matching_mask
            self.matching_mask_source = (
                source_motion[0].motion_data.matching_mask
                if source_motion is not None
                else None
            )
            self.masking_map_target = target[0].motion_data.masking_map
            self.masking_map_source = (
                source_motion[0].motion_data.masking_map
                if source_motion is not None
                else None
            )
        else:
            self.matching_mask = None

        projection_mask = projection_masking(
            self.masking_map_target,
            self.masking_map_source,
            len(self.matching_mask_target),
            len(self.matching_mask_source),
        )

        if debug_dir is not None:
            from tensorboardX import SummaryWriter

            writer = SummaryWriter(log_dir=debug_dir)

        # build target pyramid
        if "patchsize" in coarse_ratio:
            coarse_ratio = (
                patch_size
                * float(coarse_ratio.split("x_")[0])
                / max([len(t.motion_data) for t in target])
            )
        elif "nframes" in coarse_ratio:
            coarse_ratio = float(coarse_ratio.split("x_")[0])
        else:
            raise ValueError("Unsupported coarse ratio specified")

        # get the source pyramid
        self.source_pyramid = self._get_source_pyramid(
            source_motion, coarse_ratio, pyr_factor
        )

        self.target_pyramid = self._get_target_pyramid(target, coarse_ratio, pyr_factor)

        # get the initial motion data
        if "nframes" in num_frames:
            """
            # change the sum() of the lengths as a random one in the list
            syn_length = int(sum([i[-1] for i in self.pyraimd_lengths]) * float(num_frames.split('x_')[0]))
            """
            syn_length = int(
                sum([i[-1] for i in self.pyraimd_lengths_source])
                * float(num_frames.split("x_")[0])
            )
        elif num_frames.isdigit():
            syn_length = int(num_frames)
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

        self.synthesized_lengths = self._get_pyramid_lengths(
            syn_length, coarse_ratio, pyr_factor
        )
        if not self.silent:
            print("Synthesized lengths:", self.synthesized_lengths)
        self.synthesized = self._get_initial_motion(
            self.synthesized_lengths[0], noise_sigma
        )

        # perform the optimization
        self.synthesized.requires_grad_(False)
        self.pbar = logger(num_steps, len(self.target_pyramid))

        lvl_source_iter = (
            iter(self.source_pyramid) if self.source_pyramid is not None else None
        )

        for lvl, lvl_target in enumerate(self.target_pyramid):
            # enmuerate the target pyramid is not right
            lvl_source = next(lvl_source_iter) if lvl_source_iter is not None else None

            self.pbar.new_lvl()
            if lvl > 0:
                with torch.no_grad():
                    self.synthesized = F.interpolate(
                        self.synthesized.detach(),
                        size=self.synthesized_lengths[lvl],
                        mode="linear",
                    )

            # if using matching mask, blend the synthesized with target
            if self.matching_mask_target is not None:
                # print("lvl")
                self.synthesized = (
                    torch.matmul(projection_mask, lvl_source[0].squeeze(0)).unsqueeze(0)
                    + (1 - self.matching_mask_target.unsqueeze(0).unsqueeze(-1))
                    * self.synthesized
                )

            self.synthesized, losses = Motion2Motion.match_and_blend(
                self.synthesized,
                lvl_target,
                criteria,
                num_steps,
                self.pbar,
                ext=ext,
                mask=self.matching_mask_target,
                matching_alpha=matching_alpha,
            )

            criteria.clean_cache()
            if debug_dir is not None:
                for itr in range(len(losses)):
                    writer.add_scalar(f"optimize/losses_lvl{lvl}", losses[itr], itr)

        # Default mode in Motion2Motion: retargeting
        if self.mode == "retargeting_copy":
            self.synthesized = (
                torch.matmul(projection_mask, lvl_source[0].squeeze(0)).unsqueeze(0)
                + (1 - self.matching_mask_target.unsqueeze(0).unsqueeze(-1))
                * self.synthesized
            )
        else:
            pass

        self.pbar.pbar.close()

        return self.synthesized.detach()

    @staticmethod
    @torch.no_grad()
    def match_and_blend(
        synthesized,
        targets,
        criteria,
        n_steps,
        pbar,
        ext=None,
        mask=None,
        matching_alpha=1,
    ):
        """
        Minimizes criteria bewteen synthesized and target
        Args:
            synthesized    : torch.Tensor, optimized motion data
            targets        : torch.Tensor, target motion data
            criteria       : optimmize target function
            n_steps        : int, number of steps to optimize
            pbar           : logger
            ext            : extra configurations or constraints (optional)
        """
        losses = []
        for _i in range(n_steps):
            synthesized, loss = criteria(
                synthesized,
                targets,
                ext=ext,
                return_blended_results=True,
                mask=mask,
                matching_alpha=matching_alpha,
            )

            # Update staus
            losses.append(loss.item())
            pbar.step()
            pbar.print()

        return synthesized, losses

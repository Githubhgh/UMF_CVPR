from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

from .utils import *


class TM2TMetrics_HM(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "matching, fid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("HM_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("HM_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []
        # Matching scores
        self.add_state("HM_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("HM_gt_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.Matching_metrics = ["HM_Matching_score", "HM_gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"HM_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"HM_R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"HM_gt_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"HM_gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.Matching_metrics)

        # Fid
        self.add_state("HM_FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("HM_FID")

        # Diversity
        self.add_state("HM_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("HM_gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["HM_Diversity", "HM_gt_Diversity"])

        # chached batches
        self.add_state("HM_text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("HM_recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("HM_gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.HM_count.item()
        count_seq = self.HM_count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag or count_seq < 500:
            return metrics

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = torch.cat(self.HM_text_embeddings,
                              axis=0).cpu()[shuffle_idx, :]
        all_genmotions = torch.cat(self.HM_recmotion_embeddings,
                                   axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.HM_gtmotion_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]

        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            group_motions = all_genmotions[i * self.R_size:(i + 1) *
                                           self.R_size]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            self.HM_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        metrics["HM_Matching_score"] = self.HM_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"HM_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            group_motions = all_gtmotions[i * self.R_size:(i + 1) *
                                          self.R_size]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            self.HM_gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        metrics["HM_gt_Matching_score"] = self.HM_gt_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"HM_gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions, 1)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions, 1)
        metrics["HM_FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["HM_Diversity"] = calculate_diversity_np(all_genmotions,
                                                      self.diversity_times)
        metrics["HM_gt_Diversity"] = calculate_diversity_np(
            all_gtmotions, self.diversity_times)

        return {**metrics}

    def update(
        self,
        text_embeddings: Tensor,
        recmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.HM_count += sum(lengths)
        self.HM_count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        text_embeddings = torch.flatten(text_embeddings, start_dim=1).detach()
        recmotion_embeddings = torch.flatten(recmotion_embeddings,
                                             start_dim=1).detach()
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings,
                                            start_dim=1).detach()

        # store all texts and motions
        self.HM_text_embeddings.append(text_embeddings)
        self.HM_recmotion_embeddings.append(recmotion_embeddings)
        self.HM_gtmotion_embeddings.append(gtmotion_embeddings)
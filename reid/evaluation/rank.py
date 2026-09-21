
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReIDMetrics:
    """
    Final person ReID evaluation metrics.

    Values are stored in [0, 1].
    """

    cmc: np.ndarray

    mean_ap: float
    mean_inp: float

    valid_queries: int
    skipped_queries: int

    @property
    def total_queries(self) -> int:

        return (
            self.valid_queries
            + self.skipped_queries
        )

    def rank(
        self,
        k: int,
    ) -> float:

        if k <= 0:
            raise ValueError(
                "Rank k must be greater than 0."
            )

        if k > len(self.cmc):
            raise ValueError(
                f"Rank-{k} unavailable. "
                f"CMC contains only "
                f"{len(self.cmc)} ranks."
            )

        return float(
            self.cmc[
                k - 1
            ]
        )


class ReIDMetricAccumulator:
    """
    Streaming CMC / mAP / mINP evaluator.

    Standard ReID filtering:

        same PID
        +
        same camera
            ↓
        remove gallery sample

    Queries with no remaining valid positive
    are skipped.
    """

    def __init__(
        self,
        max_rank: int = 50,
    ) -> None:

        if max_rank <= 0:
            raise ValueError(
                "max_rank must be greater than 0."
            )

        self.max_rank = (
            max_rank
        )

        self._cmc_sum = np.zeros(
            max_rank,
            dtype=np.float64,
        )

        self._ap_sum = 0.0
        self._inp_sum = 0.0

        self._valid_queries = 0
        self._skipped_queries = 0

    @property
    def valid_queries(
        self,
    ) -> int:

        return self._valid_queries

    @property
    def skipped_queries(
        self,
    ) -> int:

        return self._skipped_queries

    def update(
        self,
        *,
        ranked_gallery_indices: np.ndarray,
        query_pid: int,
        query_camera_id: int,
        gallery_pids: np.ndarray,
        gallery_camera_ids: np.ndarray,
    ) -> bool:

        ranking = np.asarray(
            ranked_gallery_indices,
            dtype=np.int64,
        ).reshape(-1)

        gallery_pids = np.asarray(
            gallery_pids,
            dtype=np.int64,
        ).reshape(-1)

        gallery_camera_ids = np.asarray(
            gallery_camera_ids,
            dtype=np.int64,
        ).reshape(-1)

        if (
            len(gallery_pids)
            != len(gallery_camera_ids)
        ):
            raise ValueError(
                "gallery_pids and "
                "gallery_camera_ids must "
                "have equal length."
            )

        if (
            len(ranking)
            != len(gallery_pids)
        ):
            raise ValueError(
                "Ranking length must equal "
                "gallery size."
            )

        if ranking.size == 0:
            raise ValueError(
                "Gallery ranking cannot be empty."
            )

        if (
            ranking.min() < 0
            or ranking.max()
            >= len(gallery_pids)
        ):
            raise ValueError(
                "Ranking contains an invalid "
                "gallery index."
            )

        ordered_pids = (
            gallery_pids[
                ranking
            ]
        )

        ordered_cameras = (
            gallery_camera_ids[
                ranking
            ]
        )

        # ====================================================
        # Standard ReID filtering
        #
        # Same PID + same camera is excluded.
        # ====================================================

        remove = (
            (
                ordered_pids
                == query_pid
            )
            &
            (
                ordered_cameras
                == query_camera_id
            )
        )

        keep = ~remove

        filtered_pids = (
            ordered_pids[
                keep
            ]
        )

        matches = (
            filtered_pids
            == query_pid
        ).astype(
            np.int32
        )

        num_relevant = int(
            matches.sum()
        )

        # ====================================================
        # Query has no valid positive
        # ====================================================

        if num_relevant == 0:

            self._skipped_queries += 1

            return False

        # ====================================================
        # CMC
        # ====================================================

        cumulative_matches = (
            np.cumsum(
                matches
            )
        )

        cmc = np.minimum(
            cumulative_matches,
            1,
        ).astype(
            np.float64
        )

        cmc_for_query = np.empty(
            self.max_rank,
            dtype=np.float64,
        )

        available_ranks = min(
            self.max_rank,
            len(cmc),
        )

        cmc_for_query[
            :available_ranks
        ] = cmc[
            :available_ranks
        ]

        if (
            available_ranks
            < self.max_rank
        ):
            cmc_for_query[
                available_ranks:
            ] = cmc[-1]

        self._cmc_sum += (
            cmc_for_query
        )

        # ====================================================
        # Average Precision
        # ====================================================

        relevant_positions = (
            np.flatnonzero(
                matches
            )
        )

        precision_at_each_rank = (
            cumulative_matches
            /
            (
                np.arange(
                    len(matches),
                    dtype=np.float64,
                )
                + 1.0
            )
        )

        average_precision = (
            (
                precision_at_each_rank
                * matches
            ).sum()
            / num_relevant
        )

        self._ap_sum += float(
            average_precision
        )

        # ====================================================
        # INP
        #
        # num positives
        # -------------------
        # rank final positive
        # ====================================================

        last_positive_index = int(
            relevant_positions[
                -1
            ]
        )

        inp = (
            num_relevant
            /
            (
                last_positive_index
                + 1
            )
        )

        self._inp_sum += float(
            inp
        )

        self._valid_queries += 1

        return True

    def compute(
        self,
    ) -> ReIDMetrics:

        if self._valid_queries == 0:

            raise RuntimeError(
                "Cannot compute ReID metrics: "
                "zero valid queries."
            )

        cmc = (
            self._cmc_sum
            / self._valid_queries
        )

        mean_ap = (
            self._ap_sum
            / self._valid_queries
        )

        mean_inp = (
            self._inp_sum
            / self._valid_queries
        )

        return ReIDMetrics(
            cmc=cmc,
            mean_ap=float(
                mean_ap
            ),
            mean_inp=float(
                mean_inp
            ),
            valid_queries=(
                self._valid_queries
            ),
            skipped_queries=(
                self._skipped_queries
            ),
        )


def evaluate_rankings(
    *,
    rankings: np.ndarray,
    query_pids: np.ndarray,
    query_camera_ids: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_camera_ids: np.ndarray,
    max_rank: int = 50,
) -> ReIDMetrics:

    rankings = np.asarray(
        rankings,
        dtype=np.int64,
    )

    query_pids = np.asarray(
        query_pids,
        dtype=np.int64,
    ).reshape(-1)

    query_camera_ids = np.asarray(
        query_camera_ids,
        dtype=np.int64,
    ).reshape(-1)

    gallery_pids = np.asarray(
        gallery_pids,
        dtype=np.int64,
    ).reshape(-1)

    gallery_camera_ids = np.asarray(
        gallery_camera_ids,
        dtype=np.int64,
    ).reshape(-1)

    if rankings.ndim != 2:
        raise ValueError(
            "rankings must have shape "
            "[num_queries, num_gallery]."
        )

    if (
        rankings.shape[0]
        != len(query_pids)
    ):
        raise ValueError(
            "Number of ranking rows does not "
            "match query PIDs."
        )

    if (
        len(query_pids)
        != len(query_camera_ids)
    ):
        raise ValueError(
            "query_pids and "
            "query_camera_ids must have "
            "equal length."
        )

    if (
        rankings.shape[1]
        != len(gallery_pids)
    ):
        raise ValueError(
            "Ranking width does not match "
            "gallery size."
        )

    accumulator = (
        ReIDMetricAccumulator(
            max_rank=max_rank,
        )
    )

    for query_index in range(
        len(query_pids)
    ):

        accumulator.update(
            ranked_gallery_indices=(
                rankings[
                    query_index
                ]
            ),

            query_pid=int(
                query_pids[
                    query_index
                ]
            ),

            query_camera_id=int(
                query_camera_ids[
                    query_index
                ]
            ),

            gallery_pids=(
                gallery_pids
            ),

            gallery_camera_ids=(
                gallery_camera_ids
            ),
        )

    return accumulator.compute()

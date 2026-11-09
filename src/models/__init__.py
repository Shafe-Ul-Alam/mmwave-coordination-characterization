"""Model architectures used in the paper.

Extracted verbatim from the research notebook cell 220L / 220J / 220M and reformatted only
by moving it into a module. No functional change: the parameter counts asserted
in tests match those reported in the paper exactly, which is what makes the
port verifiable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NB = 64
M = 3
HIDDEN_DIMENSION = 80
MESSAGE_ROUNDS = 4
BEAM_FEATURE_COUNT = 7
BEAM_ATTENTION_LAYERS = 3
EXPECTED_U = 181
SD_CLEAN = 1.263322


class BeamAttn2(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.q = nn.Linear(
            dimension,
            dimension,
        )

        self.k = nn.Linear(
            dimension,
            dimension,
        )

        self.v = nn.Linear(
            dimension,
            dimension,
        )

        self.o = nn.Linear(
            dimension,
            dimension,
        )

        self.n1 = nn.LayerNorm(
            dimension
        )

        self.n2 = nn.LayerNorm(
            dimension
        )

        self.ff = nn.Sequential(
            nn.Linear(
                dimension,
                2 * dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                2 * dimension,
                dimension,
            ),
        )

        self.wov = nn.Parameter(
            torch.tensor(2.0)
        )

    def forward(
        self,
        hidden,
        overlap,
    ):
        attention_logits = (
            self.q(hidden)
            @ self.k(hidden).transpose(-2, -1)
        ) / math.sqrt(hidden.shape[-1])

        attention_logits = (
            attention_logits
            - self.wov * overlap
        )

        attended = (
            F.softmax(
                attention_logits,
                dim=-1,
            )
            @ self.v(hidden)
        )

        hidden = self.n1(
            hidden
            + self.o(attended)
        )

        return self.n2(
            hidden
            + self.ff(hidden)
        )


class JointDecGATv14Batch(nn.Module):
    def __init__(
        self,
        number_of_beams=NB,
        dimension=HIDDEN_DIMENSION,
        rounds=MESSAGE_ROUNDS,
        number_of_features=BEAM_FEATURE_COUNT,
        number_of_layers=BEAM_ATTENTION_LAYERS,
        normalization_sd=SD_CLEAN,
    ):
        super().__init__()

        self.nb = number_of_beams
        self.d = dimension
        self.R = rounds

        self.register_buffer(
            "SDv",
            torch.tensor(
                float(normalization_sd)
            ),
        )

        self.enc = nn.Sequential(
            nn.Linear(
                number_of_beams,
                dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                dimension,
                dimension,
            ),
        )

        self.gru = nn.GRUCell(
            dimension + 2,
            dimension,
        )

        self.ln = nn.LayerNorm(
            dimension
        )

        self.assoc_head = nn.Sequential(
            nn.Linear(
                dimension,
                dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                dimension,
                1,
            ),
        )

        self.beam_in = nn.Linear(
            number_of_features + dimension,
            dimension,
        )

        self.beam_lay = nn.ModuleList(
            [
                BeamAttn2(dimension)
                for _ in range(
                    number_of_layers
                )
            ]
        )

        self.beam_out = nn.Linear(
            dimension,
            1,
        )

    def _coverage(
        self,
        normalized_input,
    ):
        relative = (
            normalized_input
            - normalized_input.amax(
                dim=-1,
                keepdim=True,
            )
        ) * self.SDv

        power = torch.pow(
            10.0,
            relative,
        )

        # B x M x U x NB
        profiles = power.permute(
            0,
            2,
            1,
            3,
        )

        normalized_profiles = (
            profiles
            / (
                profiles.norm(
                    dim=2,
                    keepdim=True,
                )
                + 1e-12
            )
        )

        overlap = (
            normalized_profiles.transpose(
                -2,
                -1,
            )
            @ normalized_profiles
        )

        return power, overlap

    def _beam_features(
        self,
        normalized_input,
        power,
        overlap,
    ):
        batch, users, base_stations, beams = (
            normalized_input.shape
        )

        best_beam = normalized_input.argmax(
            dim=-1
        )

        top_three = normalized_input.topk(
            3,
            dim=-1,
        ).indices

        popularity = torch.zeros(
            batch,
            base_stations,
            beams,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )

        popularity_indices = (
            best_beam
            .permute(0, 2, 1)
        )

        popularity_source = torch.full(
            popularity_indices.shape,
            1.0 / users,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )

        popularity.scatter_add_(
            2,
            popularity_indices,
            popularity_source,
        )

        in_top_three = torch.zeros_like(
            popularity
        )

        top_three_indices = (
            top_three
            .permute(0, 2, 1, 3)
            .reshape(
                batch,
                base_stations,
                users * 3,
            )
        )

        top_three_source = torch.full(
            top_three_indices.shape,
            1.0 / users,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )

        in_top_three.scatter_add_(
            2,
            top_three_indices,
            top_three_source,
        )

        top_ten_percent = max(
            1,
            users // 10,
        )

        redundancy = (
            overlap.sum(dim=-1)
            - 1.0
        ) / (beams - 1)

        return torch.stack(
            [
                popularity,
                in_top_three,
                normalized_input.mean(dim=1),
                normalized_input.amax(dim=1),
                normalized_input.topk(
                    top_ten_percent,
                    dim=1,
                ).values.mean(dim=1),
                power.mean(dim=1),
                redundancy,
            ],
            dim=-1,
        )

    def _beam_forward(
        self,
        beam_features,
        overlap,
        context,
    ):
        hidden = self.beam_in(
            torch.cat(
                [
                    beam_features,
                    context,
                ],
                dim=-1,
            )
        )

        for layer in self.beam_lay:
            hidden = layer(
                hidden,
                overlap,
            )

        return self.beam_out(
            hidden
        ).squeeze(-1)

    def forward(
        self,
        normalized_input,
    ):
        batch, users, base_stations, beams = (
            normalized_input.shape
        )

        power, overlap = self._coverage(
            normalized_input
        )

        beam_features = self._beam_features(
            normalized_input,
            power,
            overlap,
        )

        hidden = self.enc(
            normalized_input.reshape(
                batch
                * users
                * base_stations,
                beams,
            )
        ).reshape(
            batch,
            users,
            base_stations,
            self.d,
        )

        initial_context = (
            hidden.mean(dim=1)
            .unsqueeze(2)
            .expand(
                batch,
                base_stations,
                beams,
                self.d,
            )
        )

        concentration = F.softmax(
            self._beam_forward(
                beam_features,
                overlap,
                initial_context,
            ),
            dim=-1,
        ).amax(dim=-1)

        for _ in range(self.R):
            other = (
                hidden.sum(
                    dim=2,
                    keepdim=True,
                )
                - hidden
            ) / max(
                base_stations - 1,
                1,
            )

            association_probability = F.softmax(
                self.assoc_head(
                    hidden
                ).squeeze(-1),
                dim=2,
            )

            load_message = (
                association_probability
                .sum(dim=1)
                / users
            ).unsqueeze(1).expand(
                batch,
                users,
                base_stations,
            ).unsqueeze(-1)

            beam_use_message = (
                concentration
                .unsqueeze(1)
                .expand(
                    batch,
                    users,
                    base_stations,
                )
                .unsqueeze(-1)
            )

            message = torch.cat(
                [
                    other,
                    load_message,
                    beam_use_message,
                ],
                dim=-1,
            )

            hidden = self.gru(
                message.reshape(
                    batch
                    * users
                    * base_stations,
                    self.d + 2,
                ),
                hidden.reshape(
                    batch
                    * users
                    * base_stations,
                    self.d,
                ),
            ).reshape(
                batch,
                users,
                base_stations,
                self.d,
            )

            hidden = self.ln(
                hidden
            )

        final_context = (
            hidden.mean(dim=1)
            .unsqueeze(2)
            .expand(
                batch,
                base_stations,
                beams,
                self.d,
            )
        )

        beam_logits = self._beam_forward(
            beam_features,
            overlap,
            final_context,
        )

        association_logits = (
            self.assoc_head(
                hidden
            ).squeeze(-1)
        )

        return (
            beam_logits,
            association_logits,
        )


class AblationBeamAttn2(nn.Module):
    """
    Exact BeamAttn2 parameterization with an optional overlap-logit term.
    """

    def __init__(
        self,
        dimension,
        use_overlap_attention=True,
    ):
        super().__init__()

        self.use_overlap_attention = bool(
            use_overlap_attention
        )

        self.q = nn.Linear(
            dimension,
            dimension,
        )

        self.k = nn.Linear(
            dimension,
            dimension,
        )

        self.v = nn.Linear(
            dimension,
            dimension,
        )

        self.o = nn.Linear(
            dimension,
            dimension,
        )

        self.n1 = nn.LayerNorm(
            dimension
        )

        self.n2 = nn.LayerNorm(
            dimension
        )

        self.ff = nn.Sequential(
            nn.Linear(
                dimension,
                2 * dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                2 * dimension,
                dimension,
            ),
        )

        self.wov = nn.Parameter(
            torch.tensor(
                2.0
            )
        )

    def forward(
        self,
        hidden,
        overlap,
    ):
        attention_logits = (
            self.q(
                hidden
            )
            @ self.k(
                hidden
            ).transpose(
                -2,
                -1,
            )
        ) / math.sqrt(
            hidden.shape[
                -1
            ]
        )

        if self.use_overlap_attention:
            attention_logits = (
                attention_logits
                - self.wov
                * overlap
            )

        attended = (
            F.softmax(
                attention_logits,
                dim=-1,
            )
            @ self.v(
                hidden
            )
        )

        hidden = self.n1(
            hidden
            + self.o(
                attended
            )
        )

        return self.n2(
            hidden
            + self.ff(
                hidden
            )
        )


class JointDecGATv14Ablation(nn.Module):
    """
    Parameter-count-preserving operational ablation of JointDecGATv14.

    All modules of the complete model are instantiated. A removed operation
    is bypassed or zeroed only in forward propagation.
    """

    def __init__(
        self,
        *,
        use_overlap_attention=True,
        use_redundancy_feature=True,
        use_message_passing=True,
        use_beam_attention=True,
        number_of_beams=NB,
        dimension=80,
        rounds=4,
        number_of_features=7,
        number_of_layers=3,
        normalization_sd=SD_CLEAN,
    ):
        super().__init__()

        self.nb = number_of_beams
        self.d = dimension
        self.R = rounds

        self.use_overlap_attention = bool(
            use_overlap_attention
        )

        self.use_redundancy_feature = bool(
            use_redundancy_feature
        )

        self.use_message_passing = bool(
            use_message_passing
        )

        self.use_beam_attention = bool(
            use_beam_attention
        )

        self.register_buffer(
            "SDv",
            torch.tensor(
                float(
                    normalization_sd
                )
            ),
        )

        self.enc = nn.Sequential(
            nn.Linear(
                number_of_beams,
                dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                dimension,
                dimension,
            ),
        )

        self.gru = nn.GRUCell(
            dimension + 2,
            dimension,
        )

        self.ln = nn.LayerNorm(
            dimension
        )

        self.assoc_head = nn.Sequential(
            nn.Linear(
                dimension,
                dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                dimension,
                1,
            ),
        )

        self.beam_in = nn.Linear(
            number_of_features
            + dimension,
            dimension,
        )

        self.beam_lay = nn.ModuleList(
            [
                AblationBeamAttn2(
                    dimension,
                    use_overlap_attention=(
                        self.use_overlap_attention
                    ),
                )
                for _
                in range(
                    number_of_layers
                )
            ]
        )

        self.beam_out = nn.Linear(
            dimension,
            1,
        )

    def _coverage(
        self,
        normalized_input,
    ):
        relative = (
            normalized_input
            - normalized_input.amax(
                dim=-1,
                keepdim=True,
            )
        ) * self.SDv

        power = torch.pow(
            10.0,
            relative,
        )

        profiles = power.permute(
            0,
            2,
            1,
            3,
        )

        normalized_profiles = (
            profiles
            / (
                profiles.norm(
                    dim=2,
                    keepdim=True,
                )
                + 1e-12
            )
        )

        overlap = (
            normalized_profiles.transpose(
                -2,
                -1,
            )
            @ normalized_profiles
        )

        return (
            power,
            overlap,
        )

    def _beam_features(
        self,
        normalized_input,
        power,
        overlap,
    ):
        batch, users, base_stations, beams = (
            normalized_input.shape
        )

        best_beam = normalized_input.argmax(
            dim=-1
        )

        top_three = normalized_input.topk(
            3,
            dim=-1,
        ).indices

        popularity = torch.zeros(
            batch,
            base_stations,
            beams,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )

        popularity_indices = best_beam.permute(
            0,
            2,
            1,
        )

        popularity_source = torch.full(
            popularity_indices.shape,
            1.0 / users,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )

        popularity.scatter_add_(
            2,
            popularity_indices,
            popularity_source,
        )

        in_top_three = torch.zeros_like(
            popularity
        )

        top_three_indices = (
            top_three
            .permute(
                0,
                2,
                1,
                3,
            )
            .reshape(
                batch,
                base_stations,
                users * 3,
            )
        )

        top_three_source = torch.full(
            top_three_indices.shape,
            1.0 / users,
            device=normalized_input.device,
            dtype=normalized_input.dtype,
        )

        in_top_three.scatter_add_(
            2,
            top_three_indices,
            top_three_source,
        )

        top_ten_percent = max(
            1,
            users // 10,
        )

        redundancy = (
            overlap.sum(
                dim=-1
            )
            - 1.0
        ) / (
            beams - 1
        )

        if not self.use_redundancy_feature:
            redundancy = torch.zeros_like(
                redundancy
            )

        return torch.stack(
            [
                popularity,
                in_top_three,
                normalized_input.mean(
                    dim=1
                ),
                normalized_input.amax(
                    dim=1
                ),
                normalized_input.topk(
                    top_ten_percent,
                    dim=1,
                ).values.mean(
                    dim=1
                ),
                power.mean(
                    dim=1
                ),
                redundancy,
            ],
            dim=-1,
        )

    def _beam_forward(
        self,
        beam_features,
        overlap,
        context,
    ):
        hidden = self.beam_in(
            torch.cat(
                [
                    beam_features,
                    context,
                ],
                dim=-1,
            )
        )

        if self.use_beam_attention:
            for layer in self.beam_lay:
                hidden = layer(
                    hidden,
                    overlap,
                )

        return self.beam_out(
            hidden
        ).squeeze(
            -1
        )

    def forward(
        self,
        normalized_input,
    ):
        batch, users, base_stations, beams = (
            normalized_input.shape
        )

        (
            power,
            overlap,
        ) = self._coverage(
            normalized_input
        )

        beam_features = self._beam_features(
            normalized_input,
            power,
            overlap,
        )

        hidden = self.enc(
            normalized_input.reshape(
                batch
                * users
                * base_stations,
                beams,
            )
        ).reshape(
            batch,
            users,
            base_stations,
            self.d,
        )

        initial_context = (
            hidden.mean(
                dim=1
            )
            .unsqueeze(
                2
            )
            .expand(
                batch,
                base_stations,
                beams,
                self.d,
            )
        )

        concentration = F.softmax(
            self._beam_forward(
                beam_features,
                overlap,
                initial_context,
            ),
            dim=-1,
        ).amax(
            dim=-1
        )

        if self.use_message_passing:
            for _ in range(
                self.R
            ):
                other = (
                    hidden.sum(
                        dim=2,
                        keepdim=True,
                    )
                    - hidden
                ) / max(
                    base_stations
                    - 1,
                    1,
                )

                association_probability = F.softmax(
                    self.assoc_head(
                        hidden
                    ).squeeze(
                        -1
                    ),
                    dim=2,
                )

                load_message = (
                    association_probability
                    .sum(
                        dim=1
                    )
                    / users
                ).unsqueeze(
                    1
                ).expand(
                    batch,
                    users,
                    base_stations,
                ).unsqueeze(
                    -1
                )

                beam_use_message = (
                    concentration
                    .unsqueeze(
                        1
                    )
                    .expand(
                        batch,
                        users,
                        base_stations,
                    )
                    .unsqueeze(
                        -1
                    )
                )

                message = torch.cat(
                    [
                        other,
                        load_message,
                        beam_use_message,
                    ],
                    dim=-1,
                )

                hidden = self.gru(
                    message.reshape(
                        batch
                        * users
                        * base_stations,
                        self.d
                        + 2,
                    ),
                    hidden.reshape(
                        batch
                        * users
                        * base_stations,
                        self.d,
                    ),
                ).reshape(
                    batch,
                    users,
                    base_stations,
                    self.d,
                )

                hidden = self.ln(
                    hidden
                )

        final_context = (
            hidden.mean(
                dim=1
            )
            .unsqueeze(
                2
            )
            .expand(
                batch,
                base_stations,
                beams,
                self.d,
            )
        )

        beam_logits = self._beam_forward(
            beam_features,
            overlap,
            final_context,
        )

        association_logits = self.assoc_head(
            hidden
        ).squeeze(
            -1
        )

        return (
            beam_logits,
            association_logits,
        )


class MatchedPooledMLP(nn.Module):
    """
    Shared per-user MLP + permutation-invariant mean/max pooling.

    No graph edges, message passing, beam-overlap matrix, or attention.
    """

    def __init__(
        self,
        users=EXPECTED_U,
        base_stations=M,
        beams=NB,
        user_dimension=112,
        hidden_dimension=256,
    ):
        super().__init__()

        self.users = users
        self.base_stations = base_stations
        self.beams = beams
        self.user_dimension = user_dimension

        input_dimension = (
            base_stations
            * beams
        )

        self.user_encoder = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dimension,
                user_dimension,
            ),
            nn.ReLU(),
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(
                2
                * user_dimension,
                user_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                user_dimension,
                user_dimension,
            ),
            nn.ReLU(),
        )

        self.association_head = nn.Sequential(
            nn.Linear(
                2
                * user_dimension,
                user_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                user_dimension,
                base_stations,
            ),
        )

        self.beam_head = nn.Sequential(
            nn.Linear(
                user_dimension,
                hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dimension,
                base_stations
                * beams,
            ),
        )

    def forward(
        self,
        normalized_input,
    ):
        batch, users, base_stations, beams = (
            normalized_input.shape
        )

        user_hidden = self.user_encoder(
            normalized_input.reshape(
                batch,
                users,
                base_stations
                * beams,
            )
        )

        global_context = (
            self.context_encoder(
                torch.cat(
                    [
                        user_hidden.mean(
                            dim=1
                        ),
                        user_hidden.amax(
                            dim=1
                        ),
                    ],
                    dim=-1,
                )
            )
        )

        expanded_context = (
            global_context
            .unsqueeze(1)
            .expand(
                batch,
                users,
                self.user_dimension,
            )
        )

        association_logits = (
            self.association_head(
                torch.cat(
                    [
                        user_hidden,
                        expanded_context,
                    ],
                    dim=-1,
                )
            )
        )

        beam_logits = (
            self.beam_head(
                global_context
            ).reshape(
                batch,
                base_stations,
                beams,
            )
        )

        return (
            beam_logits,
            association_logits,
        )


class MatchedCNN2D(nn.Module):
    """
    Fully convolutional baseline over the user-by-beam tensor.

    It has no explicit graph construction, decentralized message passing,
    overlap-conditioned attention, or learned inter-BS message exchange.
    """

    def __init__(
        self,
        base_stations=M,
        channels=(
            64,
            80,
            104,
            104,
        ),
    ):
        super().__init__()

        c1, c2, c3, c4 = channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                base_stations,
                c1,
                kernel_size=(
                    3,
                    5,
                ),
                padding=(
                    1,
                    2,
                ),
            ),
            nn.ReLU(),
            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                c2,
                c3,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                c3,
                c4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.association_head = nn.Conv1d(
            c4,
            base_stations,
            kernel_size=1,
        )

        self.beam_head = nn.Conv1d(
            c4,
            base_stations,
            kernel_size=1,
        )

    def forward(
        self,
        normalized_input,
    ):
        # B x M x U x NB
        feature_map = self.feature_extractor(
            normalized_input.permute(
                0,
                2,
                1,
                3,
            )
        )

        # Average over beams, then classify each user.
        association_logits = (
            self.association_head(
                feature_map.mean(
                    dim=-1
                )
            )
            .permute(
                0,
                2,
                1,
            )
        )

        # Average over users, then score each beam at each BS.
        beam_logits = self.beam_head(
            feature_map.mean(
                dim=2
            )
        )

        return (
            beam_logits,
            association_logits,
        )

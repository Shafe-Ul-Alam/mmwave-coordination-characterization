"""The imitation teacher: popularity initialisation + swap refinement.

This is Algorithm 1 of the paper. It is a heuristic: the (1-1/e) guarantee of
Theorem 1 applies to greedy maximisation of the submodular surrogate F_m, NOT
to this procedure. Cell 220T measures its optimality gap on reduced instances
(99.17 per cent of the exact optimum, exactly optimal in 85.5 per cent).

Extracted verbatim from the research notebook cell 220A and reformatted only
by moving it into a module. No functional change: the parameter counts asserted
in tests match those reported in the paper exactly, which is what makes the
port verifiable.
"""

import numpy as np

NOISE_VAR = 1e-9
M = 3
B_RF = 8
SWAP_PASSES = 2
CANDIDATE_BEAMS = 16
SCORING_PF_ITERATIONS = 2
naive_fill_count = 0


def spectral_efficiency(gain):
    gain = np.asarray(
        gain,
        dtype=np.float64,
    )

    return np.log2(
        1.0
        + np.maximum(gain, 0.0)
        / NOISE_VAR
    )


def effective_spectral_efficiency(
    spectral_efficiency_array,
    active_beams,
):
    users = spectral_efficiency_array.shape[0]

    output = np.zeros(
        (users, M),
        dtype=np.float64,
    )

    for base_station in range(M):
        selected = sorted(
            int(beam)
            for beam in active_beams[base_station]
        )

        if selected:
            output[:, base_station] = (
                spectral_efficiency_array[
                    :,
                    base_station,
                    selected,
                ].max(axis=1)
            )

    return output


def pf_assoc_fast(
    effective_rates,
    iterations=10,
    initial_association=None,
):
    return pf_assoc_with_info(
        effective_rates,
        iterations=iterations,
        initial_association=initial_association,
    )[0]


def rates_from_assignment(
    effective_rates,
    association,
):
    users = effective_rates.shape[0]

    load = np.bincount(
        association,
        minlength=M,
    ).astype(np.float64)

    safe_load = np.maximum(
        load,
        1.0,
    )

    rates = (
        effective_rates[
            np.arange(users),
            association,
        ]
        / safe_load[association]
    )

    return (
        np.maximum(
            rates,
            1e-12,
        ),
        load,
    )


def pf_assoc_with_info(
    effective_rates,
    iterations=10,
    initial_association=None,
):
    users, base_stations = effective_rates.shape

    log_effective_rates = np.log(
        np.maximum(
            effective_rates,
            1e-300,
        )
    )

    if initial_association is None:
        association = (
            effective_rates
            .argmax(axis=1)
            .astype(np.int64)
        )
    else:
        association = np.asarray(
            initial_association,
            dtype=np.int64,
        ).copy()

    assert association.shape == (users,)
    assert np.all(
        (association >= 0)
        & (association < base_stations)
    )

    load = np.bincount(
        association,
        minlength=base_stations,
    ).astype(np.float64)

    converged = False
    iterations_used = 0

    for iteration in range(iterations):
        changed = False

        for user in range(users):
            current = int(
                association[user]
            )

            candidate_utility = (
                log_effective_rates[user]
                - np.log(load + 1.0)
            )

            candidate_utility[current] = (
                log_effective_rates[
                    user,
                    current,
                ]
                - np.log(
                    max(
                        load[current],
                        1.0,
                    )
                )
            )

            best_base_station = int(
                np.argmax(
                    candidate_utility
                )
            )

            if best_base_station != current:
                load[current] -= 1.0
                load[best_base_station] += 1.0

                association[user] = (
                    best_base_station
                )

                changed = True

        iterations_used = iteration + 1

        if not changed:
            converged = True
            break

    return (
        association,
        iterations_used,
        converged,
    )


def score_geometric_mean(
    spectral_efficiency_array,
    active_beams,
    pf_iterations=SCORING_PF_ITERATIONS,
):
    effective_rates = (
        effective_spectral_efficiency(
            spectral_efficiency_array,
            active_beams,
        )
    )

    association = pf_assoc_fast(
        effective_rates,
        iterations=pf_iterations,
    )

    rates, _ = rates_from_assignment(
        effective_rates,
        association,
    )

    return float(
        np.exp(
            np.mean(
                np.log(rates)
            )
        )
    )


def naive_top_np(
    spectral_efficiency_array,
    budget=B_RF,
):
    global naive_fill_count

    active = []

    for base_station in range(M):
        strongest_beam = (
            spectral_efficiency_array[
                :,
                base_station,
                :,
            ].argmax(axis=1)
        )

        values, counts = np.unique(
            strongest_beam,
            return_counts=True,
        )

        order = np.argsort(
            counts
        )[::-1]

        chosen = [
            int(values[index])
            for index in order[:budget]
        ]

        if len(chosen) < budget:
            naive_fill_count += 1

            mean_quality = (
                spectral_efficiency_array[
                    :,
                    base_station,
                    :,
                ].mean(axis=0)
            )

            for beam in np.argsort(
                mean_quality
            )[::-1]:
                beam = int(beam)

                if beam not in chosen:
                    chosen.append(beam)

                if len(chosen) == budget:
                    break

        active.append(
            set(chosen[:budget])
        )

    return active


def swap_refine_fast(
    spectral_efficiency_array,
    active_beams,
    passes=SWAP_PASSES,
    candidate_count=CANDIDATE_BEAMS,
):
    active = [
        set(group)
        for group in active_beams
    ]

    best_score = score_geometric_mean(
        spectral_efficiency_array,
        active,
        pf_iterations=SCORING_PF_ITERATIONS,
    )

    for _ in range(passes):
        improved = False

        for base_station in range(M):
            max_quality = (
                spectral_efficiency_array[
                    :,
                    base_station,
                    :,
                ].max(axis=0)
            )

            unchosen = [
                int(beam)
                for beam in np.argsort(
                    max_quality
                )[::-1]
                if int(beam)
                not in active[base_station]
            ][:candidate_count]

            for beam_out in list(
                active[base_station]
            ):
                for beam_in in unchosen:
                    trial = [
                        set(group)
                        for group in active
                    ]

                    trial[
                        base_station
                    ].discard(
                        int(beam_out)
                    )

                    trial[
                        base_station
                    ].add(
                        int(beam_in)
                    )

                    score = score_geometric_mean(
                        spectral_efficiency_array,
                        trial,
                        pf_iterations=(
                            SCORING_PF_ITERATIONS
                        ),
                    )

                    if score > (
                        best_score
                        * (1.0 + 1e-9)
                    ):
                        best_score = score
                        active = trial
                        improved = True

                        unchosen = [
                            int(beam)
                            for beam in np.argsort(
                                max_quality
                            )[::-1]
                            if int(beam)
                            not in active[
                                base_station
                            ]
                        ][:candidate_count]

                        break

        if not improved:
            break

    return active

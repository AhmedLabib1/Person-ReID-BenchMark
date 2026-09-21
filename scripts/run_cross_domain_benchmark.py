
from __future__ import annotations


import argparse
import json
import subprocess
import sys
import time

from pathlib import Path



PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


BUILD_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "build_embedding_cache.py"
)


EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "evaluate_embedding_cache.py"
)



# ============================================================
# DukeMTMC FastReID models
# ============================================================

MODELS = [

    "bot_r50",
    "bot_r50_ibn",
    "bot_s50",
    "bot_r101_ibn",

    "agw_r50",
    "agw_r50_ibn",
    "agw_s50",
    "agw_r101_ibn",

    "sbs_r50",
    "sbs_r50_ibn",
    "sbs_s50",
    "sbs_r101_ibn",

    "mgn_r50_ibn",

]



DATASETS = [

    "market1501",
    "msmt17",
    "mars",

]



PROTOCOLS = {

    "market1501":
        "single_image_l2",

    "msmt17":
        "single_image_l2",

    "mars":
        "uniform8_mean_raw_l2",

}



# ============================================================
# Commands
# ============================================================


def run_command(
    command,
):

    print()

    print(
        "RUNNING:"
    )

    print(
        " ".join(command)
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )


    if result.returncode != 0:

        raise RuntimeError(
            "Command failed."
        )



# ============================================================
# Status
# ============================================================


def metrics_exist(
    model,
    dataset,
):

    path = (

        PROJECT_ROOT

        /

        "benchmarks"

        /

        "results"

        /

        model

        /

        dataset

        /

        PROTOCOLS[dataset]

        /

        "metrics.json"

    )


    return path.exists()



def cache_exists(
    model,
    dataset,
):

    root = (

        PROJECT_ROOT

        /

        "benchmarks"

        /

        "embeddings"

        /

        model

        /

        dataset

        /

        PROTOCOLS[dataset]

    )


    return (

        (root / "query.npz").exists()

        and

        (root / "gallery.npz").exists()

    )



# ============================================================
# Benchmark
# ============================================================


def run_single(
    model,
    dataset,
):


    print()
    print("=" * 90)

    print(
        f"{model} × {dataset}"
    )

    print("=" * 90)



    # --------------------------------
    # Cache
    # --------------------------------

    if cache_exists(
        model,
        dataset,
    ):

        print(
            "Cache: EXISTS - skipping"
        )

    else:

        print(
            "Cache: BUILDING"
        )


        command = [

            sys.executable,

            str(BUILD_SCRIPT),

            "--model",

            model,

            "--dataset",

            dataset,

            "--split",

            "both",

        ]


        if dataset in [
            "market1501",
            "msmt17",
        ]:

            command += [

                "--batch-size",

                "16",

                "--image-chunk-size",

                "1024",

            ]


        else:

            command += [

                "--frame-batch-size",

                "16",

                "--tracklet-chunk-size",

                "64",

            ]


        run_command(
            command
        )



    # --------------------------------
    # Evaluation
    # --------------------------------

    if metrics_exist(
        model,
        dataset,
    ):

        print(
            "Evaluation: EXISTS - skipping"
        )

        return



    print(
        "Evaluation: RUNNING"
    )


    command = [

        sys.executable,

        str(EVAL_SCRIPT),

        "--model",

        model,

        "--dataset",

        dataset,

        "--query-batch-size",

        "128",

        "--max-rank",

        "50",

        "--device",

        "cuda",

    ]


    run_command(
        command
    )



# ============================================================
# Main
# ============================================================


def main():


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute everything."
        ),
    )


    args = parser.parse_args()



    total = (
        len(MODELS)
        *
        len(DATASETS)
    )


    completed = 0



    start = time.time()



    print("=" * 90)

    print(
        "SHAWAF DUKE FULL CROSS DOMAIN BENCHMARK"
    )

    print("=" * 90)


    print()

    print(
        "Models :",
        len(MODELS)
    )

    print(
        "Datasets :",
        len(DATASETS)
    )

    print(
        "Total runs :",
        total
    )



    for model in MODELS:

        for dataset in DATASETS:


            if args.force:

                pass


            run_single(
                model,
                dataset,
            )


            completed += 1


            print()

            print(
                f"Progress "
                f"{completed}/{total}"
            )



    elapsed = (
        time.time()
        -
        start
    )


    print()

    print("=" * 90)

    print(
        "BENCHMARK FINISHED"
    )

    print("=" * 90)


    print(
        "Total time:",
        f"{elapsed/3600:.2f} hours"
    )



if __name__ == "__main__":

    main()


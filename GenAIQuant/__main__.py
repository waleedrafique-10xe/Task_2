from __future__ import annotations

import multiprocessing as mp
from multiprocessing import get_context

mp.set_start_method("spawn", force=True)

from .config import Config
from .evaluation import evaluate_full_precision_model
from .logger import logger
from .processes import (
    evaluation_pipeline_process,
    quantization_pipeline_process,
)
from .utils import normalize_model_id, print_config
from .utils.arg_parser import parse_args


def main():
    args = parse_args()

    skip_evaluation = args.skip_evaluation
    skip_quantization = args.skip_quantization
    evaluate_fp = args.evaluate_fp

    config = Config.from_file(args.config)
    model_id = args.model

    normalized_model_id = normalize_model_id(model_id)

    config.output_dir = config.output_dir / normalized_model_id

    print_config(config)

    if evaluate_fp:
        if config.evaluation is None:
            logger.info(
                "No evaluation config found -- Skipping"
                "Evaluation on Full Precision Model"
            )
            return

        evaluate_fp_process = get_context("spawn").Process(
            target=evaluate_full_precision_model, args=(model_id, config)
        )
        logger.info(f"Starting Evaluation on Full Precision model: {model_id}")
        evaluate_fp_process.start()
        evaluate_fp_process.join()

        if evaluate_fp_process.exitcode != 0:
            logger.error("Full Precision evaluation  failed")
            return

        logger.info("Full Precision evaluation pipeline completed")
        return

    if not skip_quantization:
        quant_process = get_context("spawn").Process(
            target=quantization_pipeline_process, args=(model_id, config)
        )
        logger.info("Starting quantization pipeline")
        quant_process.start()
        quant_process.join()

        if quant_process.exitcode != 0:
            logger.error("Quantization pipeline failed")
            return

        logger.info("Quantization pipeline completed")
    else:
        logger.info(
            "`skip-quantization` arguement supplied -- Skipping quantization pipeline"
        )

    if config.evaluation is None:
        logger.info("No Evaluation Config found -- Skipping Evaluation")
        return

    if skip_evaluation:
        logger.info("`skip-evaluation` arguement supplied -- Skipping Evaluation")
        return

    eval_process = get_context("spawn").Process(
        target=evaluation_pipeline_process, args=(model_id, config)
    )

    logger.info("Starting evaluation pipeline")
    eval_process.start()
    eval_process.join()

    if eval_process.exitcode != 0:
        logger.error("Evaluation pipeline failed")
        return

    logger.info("Evaluation pipeline completed")


if __name__ == "__main__":
    main()

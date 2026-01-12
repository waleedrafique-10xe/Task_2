import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Quantization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or model name (from hugging face hub)",
    )
    required.add_argument(
        "--config",
        type=str,
        help="Path to custom config file",
    )

    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skips the QDQ and Evaluation flow",
    )
    optional.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skips the entire Optimization pipeline -- Useful"
        " for running just evaluation on optimized model",
    )
    optional.add_argument(
        "--evaluate-fp",
        action="store_true",
        help="Evaluates the provided model without running any quantization or QDQ hooks"
        "Should be used to obtain baseline results",
    )

    return parser.parse_args()

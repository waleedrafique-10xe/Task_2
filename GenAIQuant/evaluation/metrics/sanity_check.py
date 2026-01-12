from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from GenAIQuant.logger import logger

from .base import EvaluationAPI

if TYPE_CHECKING:
    from pathlib import Path

    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

_VLM_PROMPTS = [
    "Describe what is happening in this image.",
    "List three objects visible in the image.",
    "Where is the person (or main subject) located within the image?",
    "What colors are most noticeable?",
    "Is the scene indoors or outdoors?",
    "Is there any text in the image? If yes, what does it say?",
]

_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

_LLM_PROMPTS: list[str] = [
    "What is 17 multiplied by 23?",
    "If Sara has 5 apples and gives 2 to Amir, how many does she have left?",
    "Summarize this sentence in one line: 'The cat slept on the warm windowsill during the storm.'",
    "A train leaves at 3 PM and arrives at 5:30 PM. How long is the journey?",
    "Convert the following into minutes: 2 hours and 45 minutes.",
    "List the steps for boiling an egg.",
    "My name for this prompt is Liora. What is my name?",
    "I will tell you a code: K93-LM. Repeat it back to me after giving one unrelated fact about birds.",
    "Which number is larger: 0.5 or 0.05?",
    "If all bloops are razzles and some razzles are flims, can we conclude that some bloops are flims? Explain briefly.",
    "Write a Python function that returns the square of a number.",
    "Fix this code: for i in range(5) print(i)",
    "Who is the current president of the United States? If uncertain, state that you are uncertain.",
    "Give a definition of quantization-aware training without mentioning any specific papers.",
    "Output a numbered list of three ways to improve model inference speed.",
    "Write a two-sentence paragraph about renewable energy.",
]

_GENERATION_KWARGS = {"max_new_tokens": 1024}


class SanityCheck(EvaluationAPI):
    _json_file_name = "sanity_check.json"

    def __init__(self):
        """
        Initialization of the Sanity Check class
        """

        super().__init__({})

    def evaluate_vlm(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        results = {}

        data_device = next(model.parameters()).device

        for prompt in tqdm(_VLM_PROMPTS):  # type: ignore
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "resized_height": 360,
                            "resized_width": 360,
                            "image": _IMAGE_URL,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(device=data_device)
            generated_ids = model.generate(**inputs, **_GENERATION_KWARGS)  # type: ignore

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            generated_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            print("=" * 50)
            print("PROMPT: ", prompt)
            print(generated_text)
            print("=" * 50)

            if output_dir is not None:
                self._save_results(results, output_dir)

            results[prompt] = generated_text
        results["image"] = _IMAGE_URL
        return results

    def evaluate_llm(
        self,
        model: PreTrainedModel,
        processor: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        results = {}

        data_device = next(model.parameters()).device

        for prompt in tqdm(_LLM_PROMPTS):  # type: ignore
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            inputs = processor.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )

            input_tokens = processor(inputs, return_tensors="pt")
            input_tokens.to(device=data_device)
            outputs = model.generate(**input_tokens, **_GENERATION_KWARGS)  # type: ignore
            generated_text = processor.decode(outputs[0])

            print("=" * 50)
            print(generated_text)
            print("=" * 50)

            if output_dir is not None:
                self._save_results(results, output_dir)

            results[prompt] = generated_text
        return results

    def evaluate(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizer | PreTrainedTokenizerFast,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        if isinstance(processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            results = self.evaluate_llm(model, processor, output_dir)

        elif isinstance(processor, ProcessorMixin):
            results = self.evaluate_vlm(model, processor, output_dir)

        else:
            logger.error(f"Unkown processor type received: {type(processor)}")
            raise ValueError(f"Unknown processor type received {type(processor)}")

        return results

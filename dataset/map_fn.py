"""CoQA dataset map function for prompt formatting."""

from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional
import yaml


def _required_template_fields(template):
    fields = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is not None and field_name != "":
            fields.add(field_name)
    return fields


def _render_user_prompt(user_prompt_template, values):
    required_fields = _required_template_fields(user_prompt_template)
    missing_fields = [field for field in required_fields if field not in values]
    if missing_fields:
        raise ValueError(f"Missing fields in prompt values: {missing_fields}")
    return user_prompt_template.format(**values)


def _normalize_system_prompt(system_prompt):
    if system_prompt is None:
        return None
    if isinstance(system_prompt, dict) and "content" in system_prompt:
        return system_prompt["content"]
    return system_prompt


def _resolve_prompt_bundle(prompt_config, default_user_prompt, system_prompt):
    if prompt_config is None:
        return {
            "system_prompt": _normalize_system_prompt(system_prompt),
            "user_prompt_template": default_user_prompt,
        }
    bundle = {
        "system_prompt": prompt_config.get("system_prompt"),
        "user_prompt_template": prompt_config["user_prompt_template"],
    }
    if system_prompt is not None:
        bundle["system_prompt"] = _normalize_system_prompt(system_prompt)
    else:
        bundle["system_prompt"] = _normalize_system_prompt(bundle["system_prompt"])
    return bundle


def _ensure_system_message(messages, system_prompt):
    if system_prompt is None:
        return messages
    return [{"role": "system", "content": str(system_prompt)}] + messages


def coqa_map_batch_fn(
    examples,
    llm_tokenizer=None,
    max_input_length: int = 4096,
    instruction: str = "",
    system_prompt: str = None,
    prompt_config=None,
):
    """Format CoQA examples into LLM chat prompts.

    Each story may contain multiple question-answer pairs; each pair becomes
    an independent sample.
    """
    stories = examples["story"]
    questions_batch = examples["questions"]
    answers_batch = examples["answers"]

    prompt_bundle = _resolve_prompt_bundle(
        prompt_config=prompt_config,
        default_user_prompt=(
            "Story:\n{story}\n\n"
            "Q: {question}\n"
            "Answer in a few words based on the story. "
            "If the answer is not in the story, answer \"unknown\".{instruction_suffix}"
        ),
        system_prompt=system_prompt,
    )

    prompts, labels, data_sources, ground_truths, reward_models = [], [], [], [], []
    for row_idx, (story, questions, answers_struct) in enumerate(
        zip(stories, questions_batch, answers_batch)
    ):
        story_text = "" if story is None else str(story)
        question_list = questions if isinstance(questions, list) else []
        answer_texts = (
            answers_struct.get("input_text", [])
            if isinstance(answers_struct, dict)
            else []
        )
        pair_count = min(len(question_list), len(answer_texts))
        for qa_idx in range(pair_count):
            question_text = str(question_list[qa_idx] or "")
            answer_text = str(answer_texts[qa_idx] or "")
            prompt_text = _render_user_prompt(
                prompt_bundle["user_prompt_template"],
                {"story": story_text, "question": question_text, "instruction_suffix": instruction},
            )
            messages = _ensure_system_message(
                [{"role": "user", "content": prompt_text}], prompt_bundle["system_prompt"]
            )
            prompts.append(messages)
            labels.append(answer_text)
            ground_truths.append(answer_text)
            reward_models.append({"style": "gt", "value": answer_text})
            data_sources.append(f"coqa_row{row_idx}_qa{qa_idx}")

    return {
        "prompt": prompts,
        "label": labels,
        "data_source": data_sources,
        "ground_truth": ground_truths,
        "reward_model": reward_models,
    }

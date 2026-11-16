
from transformers import AutoTokenizer
from src.data_loader import load_squad_dataset

MODEL_NAME = "distilbert/distilbert-base-uncased-distilled-squad"
MAX_LENGTH = 384
DOC_STRIDE = 128


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def validate_qa_inputs(question, context, tokenizer):
    """Validate inputs before sliding-window tokenization."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context cannot be empty.")

    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required.")

    if tokenizer.model_max_length < MAX_LENGTH:
        raise ValueError(
            "Tokenizer model maximum length is smaller than "
            f"{MAX_LENGTH} tokens."
        )

    question_ids = tokenizer(
        question,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    available_context_tokens = (
        MAX_LENGTH - len(question_ids) - special_tokens
    )

    if available_context_tokens <= DOC_STRIDE:
        max_question_tokens = (
            MAX_LENGTH - special_tokens - DOC_STRIDE - 1
        )

        raise ValueError(
            "Question is too long. "
            f"Maximum supported question length is "
            f"{max_question_tokens} tokens for this tokenizer. "
            f"Your question contains {len(question_ids)} tokens. "
            "Please shorten your question."
        )


def tokenize_example(question, context, tokenizer):
    """Tokenize a QA pair with overlapping context windows."""

    validate_qa_inputs(question, context, tokenizer)

    return tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )


def align_answer_positions(tokenized_inputs, answers, tokenizer):
    """Map reference answer character offsets to token positions."""

    if (
        not answers
        or not answers.get("answer_start")
        or not answers.get("text")
    ):
        raise ValueError(
            "Answer alignment requires at least one reference answer."
        )

    answer_start = answers["answer_start"][0]
    answer_end = answer_start + len(answers["text"][0])

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        sequence_ids = tokenized_inputs.sequence_ids(i)
        input_ids = tokenized_inputs["input_ids"][i]

        cls_token_id = tokenizer.cls_token_id

        if cls_token_id is None or cls_token_id not in input_ids:
            raise ValueError("CLS token not found in tokenized input.")

        cls_index = input_ids.index(cls_token_id)

        context_indices = [
            j
            for j, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
        ]

        if not context_indices:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        first = context_indices[0]
        last = context_indices[-1]

        if (
            offsets[first][0] > answer_start
            or offsets[last][1] < answer_end
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start = first

        while (
            start <= last
            and offsets[start][1] <= answer_start
        ):
            start += 1

        end = last

        while (
            end >= first
            and offsets[end][0] >= answer_end
        ):
            end -= 1

        start_positions.append(start)
        end_positions.append(end)

    return start_positions, end_positions


def validate_preprocessing(sample, tokenized_inputs, tokenizer):
    """Display answer-alignment diagnostics."""

    answers = sample["answers"]

    start_positions, end_positions = align_answer_positions(
        tokenized_inputs,
        answers,
        tokenizer,
    )

    print("Question:", sample["question"])
    print("Expected Answer:", answers["text"][0])
    print("Total Windows:", len(tokenized_inputs["input_ids"]))

    for i, (start, end) in enumerate(
        zip(start_positions, end_positions)
    ):
        input_ids = tokenized_inputs["input_ids"][i]

        recovered_answer = tokenizer.decode(
            input_ids[start:end + 1],
            skip_special_tokens=True,
        )

        print(f"\nWindow {i + 1}")
        print("Token Count:", len(input_ids))
        print("Start Position:", start)
        print("End Position:", end)
        print("Recovered Answer:", recovered_answer)


def main():
    dataset = load_squad_dataset()
    sample = dataset["train"][0]
    tokenizer = load_tokenizer()

    long_context = sample["context"] * 5

    tokenized_inputs = tokenize_example(
        sample["question"],
        long_context,
        tokenizer,
    )

    print("Total Windows:", len(tokenized_inputs["input_ids"]))

    for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
        print(f"Window {i + 1}: {len(input_ids)} tokens")

    for i in range(len(tokenized_inputs["input_ids"]) - 1):
        current_ids = tokenized_inputs["input_ids"][i]
        next_ids = tokenized_inputs["input_ids"][i + 1]

        current_context = [
            token_id
            for token_id, sid in zip(
                current_ids,
                tokenized_inputs.sequence_ids(i),
            )
            if sid == 1
        ]

        next_context = [
            token_id
            for token_id, sid in zip(
                next_ids,
                tokenized_inputs.sequence_ids(i + 1),
            )
            if sid == 1
        ]

        overlap_matches = (
            current_context[-DOC_STRIDE:]
            == next_context[:DOC_STRIDE]
        )

        print(
            f"Window {i + 1} → {i + 2} overlap:",
            overlap_matches,
        )


if __name__ == "__main__":
    main()

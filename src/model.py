import torch
from transformers import AutoModelForQuestionAnswering
from src.data_loader import load_squad_dataset
from src.preprocessing import load_tokenizer,tokenize_example

MODEL_NAME = "distilbert/distilbert-base-uncased-distilled-squad"

def load_model():
    device = torch.device("cpu")

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model, device


def prepare_inputs(tokenized_inputs, window_index, device):
    model_inputs = {}

    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key in tokenized_inputs:
            model_inputs[key] = torch.tensor(
                [tokenized_inputs[key][window_index]],
                dtype=torch.long,
                device=device
            )

    return model_inputs

def predict_logits(model, model_inputs):
    with torch.inference_mode():
        outputs = model(**model_inputs)

    return outputs

def extract_answer(context, tokenized_inputs, outputs, window_index=0):
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    offsets = tokenized_inputs["offset_mapping"][window_index]
    sequence_ids = tokenized_inputs.sequence_ids(window_index)

    context_indices = [
        i for i, sid in enumerate(sequence_ids)
        if sid == 1
    ]

    if not context_indices:
        return "", float("-inf")

    max_answer_length = 30
    best_score = float("-inf")
    best_answer = ""

    for start in context_indices:
        for end in range(start, min(start + max_answer_length, len(offsets))):

            if sequence_ids[end] != 1:
                break

            start_char = offsets[start][0]
            end_char = offsets[end][1]

            if end_char <= start_char:
                continue

            score = (
                start_logits[start].item()
                + end_logits[end].item()
            )

            if score > best_score:
                best_score = score
                best_answer = context[start_char:end_char]

    return best_answer, best_score

def predict_answer(question, context, model, tokenizer, device):
    tokenized_inputs = tokenize_example(question, context, tokenizer)

    best_answer = ""
    best_score = float("-inf")

    for window_index in range(len(tokenized_inputs["input_ids"])):
        model_inputs = prepare_inputs(tokenized_inputs, window_index, device)

        outputs = predict_logits(model, model_inputs)

        answer, score = extract_answer(
            context, tokenized_inputs, outputs, window_index
        )

        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer, best_score


def main():
    dataset = load_squad_dataset()
    sample = dataset["train"][0]

    tokenizer = load_tokenizer()
    model, device = load_model()

    long_context = sample["context"] * 5

    tokenized_inputs = tokenize_example(
        sample["question"], long_context, tokenizer
    )

    answer, score = predict_answer(
        sample["question"],
        long_context,
        model,
        tokenizer,
        device
    )

    print("Total Windows:", len(tokenized_inputs["input_ids"]))
    print("Expected Answer:", sample["answers"]["text"][0])
    print("Predicted Answer:", answer)
    print("Answer Score:", score)


if __name__ == "__main__":
    main()

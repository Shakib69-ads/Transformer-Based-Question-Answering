from src.data_loader import load_squad_dataset

def analyze_dataset():
    dataset = load_squad_dataset()

    sample = dataset['train'][0]

    print("Title :", sample['title'])
    print("Context :", sample['context'])
    print("Question :", sample['question'])
    print("Answers :", sample['answers'])

    # Answer varification
    answer_text = sample["answers"]["text"][0]
    answer_start = sample["answers"]["answer_start"][0]

    extracted_answer = sample["context"][
        answer_start : answer_start + len(answer_text)
    ]

    print("Expected Answer:", answer_text)
    print("Extracted Answer:", extracted_answer)
    print("Match:", answer_text == extracted_answer)



if __name__ == "__main__":
    analyze_dataset()
from datasets import load_dataset

def load_squad_dataset():
    """
    Load the SQuAD v1.1 dataset from Hugging Face.

    Returns:
        DatasetDict containing train and validation splits.
    """

    dataset = load_dataset("rajpurkar/squad")
    return dataset
import random
import numpy as np
import torch
import os


def fix_random_seed(seed):
    """
    Fix the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def show_dataset_info(dataset, max_classes_to_show=10):
    """
    Print basic information about the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to inspect.
        max_classes_to_show (int): Maximum number of class names to display
                                   if there are many classes.
    """
    print(f"Number of samples: {len(dataset)}")
    if hasattr(dataset, "classes"):
        if len(dataset.classes) > max_classes_to_show:
            print(
                f"Classes: {dataset.classes[:max_classes_to_show]} ... (and {len(dataset.classes) - max_classes_to_show} more)"
            )
        else:
            print(f"Classes: {dataset.classes}")
    if hasattr(dataset, "class_to_idx"):
        print(f"Class to index mapping: {dataset.class_to_idx}")
    if hasattr(dataset, "data"):
        try:
            # Attempt to show first 5 samples, handling potential errors if data is not directly sliceable or printable
            sample_data_repr = repr(dataset.data[:5])
            if len(sample_data_repr) > 200:  # Limit length of representation
                sample_data_repr = sample_data_repr[:200] + "..."
            print(f"Sample data (first 5): {sample_data_repr}")
        except TypeError:
            print("Sample data is not directly sliceable or representable in this way.")
        except AttributeError:
            # For datasets like ImageFolder, .data might not exist, but .samples or .imgs might
            if hasattr(dataset, "samples"):
                print(f"Sample paths (first 5): {dataset.samples[:5]}")
            elif hasattr(dataset, "imgs"):
                print(f"Sample paths (first 5): {dataset.imgs[:5]}")
            else:
                print(
                    "No standard 'data', 'samples', or 'imgs' attribute found for sample preview."
                )
    else:
        # Fallback if .data attribute doesn't exist
        try:
            # Try to get a few samples using __getitem__
            samples_to_show = min(5, len(dataset))
            if samples_to_show > 0:
                print(f"First {samples_to_show} sample(s) (obtained via __getitem__):")
                for i in range(samples_to_show):
                    sample_repr = repr(dataset[i])
                    if (
                        len(sample_repr) > 100
                    ):  # Limit length of representation for each sample
                        sample_repr = sample_repr[:100] + "..."
                    print(f"  Sample {i}: {sample_repr}")
            else:
                print("Dataset is empty.")
        except Exception as e:
            print(f"Could not retrieve sample data using __getitem__: {e}")

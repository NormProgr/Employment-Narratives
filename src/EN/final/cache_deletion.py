"""Function that frees up memory by deleting cache files."""

import os

# def delete_caches_in_directory(directory):
#    for root, _, files in os.walk(directory):
#        for file in files:
#            if file.endswith(".cache"):


def delete_caches_in_directory(folder_path):
    """Delete old scrapes to save memory and for overview.

    Args:
        folder_path (directory): Directory of the folder containing scrapes.

    """
    folder_contents = os.listdir(folder_path)
    for item in folder_contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item_path.startswith("cache-"):
            os.remove(item_path)

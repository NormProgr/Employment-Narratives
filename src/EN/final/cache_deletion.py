"""Function that frees up memory by deleting cache files."""

import os


def delete_caches_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cache"):
                cache_file_path = os.path.join(root, file)
                os.remove(cache_file_path)

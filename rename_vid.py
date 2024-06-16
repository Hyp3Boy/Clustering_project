import os


def rename_videos(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            # Find the index of the first underscore before the .mp4 extension
            underscore_index = filename.find("_")
            extension_index = filename.rfind(".mp4")

            if underscore_index != -1 and underscore_index < extension_index:
                # Create the new filename by taking the part before the first underscore
                new_filename = filename[:underscore_index] + ".mp4"
                # Construct full file paths
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed: {filename} to {new_filename}")
            else:
                print(f"Skipping: {filename} (no underscore found before .mp4)")


# Example usage
directory = "./videos/train_subset/"
rename_videos(directory)
directory = "./videos/val_subset/"
rename_videos(directory)
directory = "./videos/test_subset/"
rename_videos(directory)

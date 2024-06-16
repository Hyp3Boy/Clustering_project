import csv


def generate_paths_from_csv(csv_file_path, output_txt_path):
    with open(csv_file_path, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_txt_path, mode="w") as txt_file:
            for row in csv_reader:
                youtube_id = row["youtube_id"]
                # csv file path without the _10.csv
                index_last_ = csv_file_path.rfind("_")
                path = f"../videos/{csv_file_path[7:index_last_]}/{youtube_id}.mp4\n"
                # path = f"../videos/train_subset/{youtube_id}.mp4\n"
                txt_file.write(path)


# Especifica la ruta a tu archivo CSV y la ruta de salida del archivo TXT
csv_file_path = "./data/train_subset_10.csv"
output_txt_path = "./file_paths_train.txt"

generate_paths_from_csv(csv_file_path, output_txt_path)

csv_file_path = "./data/test_subset_10.csv"
output_txt_path = "./file_paths_test.txt"

generate_paths_from_csv(csv_file_path, output_txt_path)

csv_file_path = "./data/val_subset_10.csv"
output_txt_path = "./file_paths_val.txt"

generate_paths_from_csv(csv_file_path, output_txt_path)

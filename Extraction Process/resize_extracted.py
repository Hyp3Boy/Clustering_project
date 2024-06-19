import os
import glob
import numpy as np
import pandas as pd


def extract_video_data(video_features_path, labels_df):
    video_features = []
    video_labels = []
    video_empty_frames = 0

    for each_video in glob.glob(os.path.join(video_features_path, "*.npy")):
        youtube_id = os.path.basename(each_video).split("_")[0]
        current_video_feature = np.load(each_video)

        if current_video_feature.size == 0:
            video_empty_frames += 1
            continue

        # aggregated_features = aggregate_features(current_video_feature)
        current_video_feature_avg = np.mean(current_video_feature, axis=0)
        # current_video_feature_avg = np.concatenate([current_video_feature_avg, aggregated_features])
        video_features.append(current_video_feature_avg)

        current_video_label = labels_df[labels_df["youtube_id"] == youtube_id][
            "label"
        ].values[0]
        video_labels.append(current_video_label)

    print(
        f"Número de videos vacíos no agregados ({video_features_path}): {video_empty_frames}"
    )

    video_features_stacked = np.vstack(video_features)
    video_labels_stacked = np.vstack(video_labels)

    df_features = pd.DataFrame(video_features_stacked)
    df_labels = pd.DataFrame(video_labels_stacked)

    return df_features, df_labels


df_train_labels = pd.read_csv(
    "./data/train_subset.csv", header=None, names=["youtube_id", "label"]
)
df_val_labels = pd.read_csv(
    "./data/val_subset.csv", header=None, names=["youtube_id", "label"]
)

X_train, Y_train = extract_video_data(
    "./data/train_output/r21d/r2plus1d_34_32_ig65m_ft_kinetics", df_train_labels
)
X_train.dropna(inplace=True)
Y_train.dropna(inplace=True)

print(f"\nNúmero de videos cargados para entrenamiento: {len(X_train)}")
print(f"Shape de nuestra matriz de videos para entrenamiento: {X_train.shape}")

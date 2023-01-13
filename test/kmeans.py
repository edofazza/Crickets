from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os


def create_dataset(control_path, sugar_path, ammonia_path):
    data, labels = [], []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        data.append(tmp_npy)
        labels.append(0)

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        data.append(tmp_npy)
        labels.append(1)

    tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(ammonia_path, i))
        data.append(tmp_npy)
        labels.append(2)

    return np.array(data), np.array(labels)


if __name__=='__main__':
    train_set, train_labels = create_dataset(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled/control/train",
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled/sugar/train",
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled/ammonia/train/",
    )
    print(train_set.shape, train_labels.shape)
    train_set = train_set.reshape((42, 34800))
    print(train_set.shape, train_labels.shape)
    scaler = StandardScaler()
    scaler.fit(train_set)
    train_set = scaler.transform(train_set)
    """train_set = scaler.transform(train_set)
    clusters = KMeans(n_clusters=3).fit_predict(train_set)
    print(f'Accuracy: {accuracy_score(train_labels, clusters)}')
    print(f'ARI: {adjusted_rand_score(train_labels, clusters)}')
    Accuracy: 0.35714285714285715 (0.30952380952380953 using scaler)
    ARI: -0.0072407551073183345 random (-0.023539614693675277)
    """
    rf = RandomForestClassifier(n_estimators=100000, random_state=42)
    rf.fit(train_set, train_labels)
    pred = rf.predict(train_set)
    print("Accuracy:", accuracy_score(train_labels, pred))

    train_set, train_labels = create_dataset(
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled/control/val",
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled/sugar/val",
        "/Users/edoardo/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/crickets/predictions/predictions_filled/ammonia/val/",
    )
    train_set = train_set.reshape((12, 34800))
    train_set = scaler.transform(train_set)
    pred = rf.predict(train_set)
    print("Accuracy:", accuracy_score(train_labels, pred))

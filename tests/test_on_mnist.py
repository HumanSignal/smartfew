import pandas as pd

from protonet_classifier import train_model, load_model, apply_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def main():
    train_set = pd.read_csv('../data/mnist-in-csv/mnist_train.csv')
    train_classes = train_set.values[:, 0]
    train_samples = train_set.values[:, 1:]
    print(f'Train samples shape: {train_samples.shape}')

    test_set = pd.read_csv('../data/mnist-in-csv/mnist_test.csv')
    test_classes = test_set.values[:, 0]
    test_samples = test_set.values[:, 1:]
    print(f'Test samples shape: {test_samples.shape}')

    pca = PCA(n_components=100)
    train_samples = pca.fit_transform(train_samples)
    test_samples = pca.transform(test_samples)

    # train_model(
    #     train_samples, train_classes,
    #     layers=(100, 100),
    #     warm_start=False,
    #     num_episodes=20000,
    #     dropout=0.2,
    #     num_support_per_class=100,
    #     num_query_per_class=100,
    # )
    # model = load_model()
    # train_samples, train_classes = apply_model(model, train_samples, train_classes)
    # test_samples = apply_model(model, test_samples)
    # print(f'Model applied, new shapes: {train_samples.shape}, {test_samples.shape}')
    # print(f'{train_samples[0, :3]}')
    # print(f'{test_samples[1, :3]}')

    clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    clf.fit(train_samples, train_classes)
    test_classes_pred = clf.predict(test_samples)

    print(classification_report(test_classes, test_classes_pred))


if __name__ == "__main__":
    main()
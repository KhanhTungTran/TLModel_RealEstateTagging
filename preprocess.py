import numpy as np
import os
# import tensornets as nets

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

def load_batch(preprocess_batch_path, n_batches, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    while True:
        for batch_id in range(1, n_batches + 1):
            batch = np.load(preprocess_batch_path + '/preprocess_batch_' + str(batch_id) + '.npz')
            features = batch["features"]
            labels = batch["labels"]
            for start in range(0, len(features), batch_size):
                end = min(start + batch_size, len(features))
                yield features[start:end], labels[start:end]
            # return batch_features_labels(batch["features"], batch["labels"], batch_size)

def load_validation(preprocess_batch_path, batch_size):
    valid = np.load(preprocess_batch_path + '/preprocess_validation.npz')
    return batch_features_labels(valid["features"], valid["labels"], batch_size)

def save_batch(preprocess_batch_path, n_batches, features, labels):
    """
    Save the Preprocessed Training data
    """
    # find index to be the point as validation data in the whole dataset of the batch (10%)
    index_of_validation = int(len(features) * 0.1)

    # list_batch_features = np.split(features[:-index_of_validation], n_batches)
    # list_batch_labels = np.split(labels[:-index_of_validation], n_batches)

    # for i in range(1, n_batches + 1):
    #     np.savez(preprocess_batch_path + '/preprocess_batch_' + str(i) + '.npz',\
    #          features=list_batch_features[i-1], labels=list_batch_labels[i-1])
    #     print("Batch number " + str(i) + ": saved")

    # np.savez(preprocess_batch_path + '/preprocess_validation.npz',\
    #         features=features[-index_of_validation:], \
    #         labels=labels[-index_of_validation:])
    # print("Validation: saved")
    list_batch_features = np.split(features, n_batches)
    list_batch_labels = np.split(labels, n_batches)

    for i in range(1, n_batches + 1):
        np.savez(preprocess_batch_path + '/preprocess_batch_' + str(i+5) + '.npz',\
             features=list_batch_features[i-1], labels=list_batch_labels[i-1])
        print("Batch number " + str(i+5) + ": saved")


def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 7))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded

def load_raw_data(path, cats):
    features = []
    labels = []
    cat_input_length = 4500
    for idx, cat in enumerate(cats):
        cat_path = os.listdir(path + '/' + cat)
        i = 0
        for img_path in cat_path[5000:]:
            try:
                img = nets.utils.load_img(path + '/' + cat + '/' + img_path,\
                        crop_size=224)
            except Exception as _:
                print(path + '/' + cat + '/' + img_path)
                raise(ValueError())
            img = img.reshape(224, 224, 3)
            features.append(img)
            labels.append(idx)
            i += 1
            if i == cat_input_length:
                break
        print(path + '/' + cat + ": done")
    features = np.array(features)
    labels = one_hot_encode(labels)
    rng_state = np.random.get_state()
    np.random.shuffle(features)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return features, labels

if __name__ == "__main__":
    categories = ['bathroom', 'bedroom', 'dining_room', 'exterior', \
                    'interior', 'kitchen', 'living_room']
    dataset_path = 'Dataset'
    features, labels = load_raw_data(dataset_path, categories)
    
    save_batch('Preprocess_batch', 5, features, labels)
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import coo_matrix
import numpy as np
import sys

def main():
    folds_base_path = sys.argv[1]
    num_folds = int(sys.argv[2])
    sparse = True if sys.argv[3]=="sparse" else False
    accuracies = []
    for i in range(1, num_folds+1):
        clf = DecisionTreeClassifier()
        test_lines = [line for line in read_file(f'{folds_base_path}{i}-test.data') if len(line) > 0]
        training_lines = [line for line in read_file(f'{folds_base_path}{i}-train.data') if len(line) > 0]
        test_data, test_labels = parse_data_labels(test_lines)
        training_data, training_labels = parse_data_labels(training_lines)
        if sparse:
            training_data, test_data = create_sparse_matrices(training_data, test_data)
        clf.fit(training_data, training_labels)
        accuracies.append(clf.score(test_data, test_labels))
        print(f'Fold {i}\'s Accuracy: {accuracies[i-1]:.5}')
    print(f'Accuracy: mean = {np.average(accuracies):.5}, std.dev = {np.std(accuracies):.5}')


 # Extracts labels and feature vectors from the specified lines
def parse_data_labels(lines):
    labels = []
    data = []
    for line in lines:
        values = [float(v) for v in line.split()]
        labels.append(values[-1])
        data.append(values[:len(values)-1])
    return data, labels

# Creates a sparse matrices from the specified data sets
def create_sparse_matrices(training_data, test_data):
    split_point = len(training_data)
    lines = training_data
    lines.extend(test_data)
    rows = []
    cols = []
    data = []
    for i in range(0, len(lines)):
        cols.extend([int(lines[i][j]) for j in range(0, len(lines[i]), 2)])
        data.extend([lines[i][j] for j in range(1, len(lines[i]), 2)])
        rows.extend(i for j in range(0, len(lines[i]), 2))
        i += 1
    matrix = coo_matrix((data, (rows, cols))).tocsr()
    return matrix[:split_point], matrix[split_point:]

# Returns the contents of the specified file.
def read_file(filename):
    with open(filename) as file:
        return file.read().split('\n')

if __name__ == '__main__':
    main()

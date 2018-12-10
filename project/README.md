# Genetic Algorithm Oblique Decision Tree (GA-ODT) and Compound Decision Tree (C-DT)
This project contains two new methods for constructing decision trees that are capable of splitting the feature space along decision boundaries that consider multiple features. The first method, C-DT, creates axis-parallel split boundaries using compound predicates through multiple creation and purity testing phases. The second method, GA-ODT, creates oblique split boundaries along hyperplanes expressed as linear combinations of the features. These linear combinations are created using a genetic algorithm.

### Prerequisites
* Java SE Development Kit (JDK) 8+

### Compilation
* To compile the methods run: javac -Xlint:unchecked -d project/target project/src/*.java

### Formatting Data
* Label files must have one class label per line in the same order as the corresponding training file.
* Class labels are treated as strings so "1" is not the same as "1 ".
* Feature vector files can either be in a sparse or a dense format, with one instance/vector per line.
* Dense feature vectors are represented on a single line with each feature in order and separated by whitespace from the next.
* Sparse feature vectors are represented on a single line. Features can be represented in any order by specifying the feature number followed by whitespace and then the value of the feature. Each feature number, value pair is separated from the next by whitespace. Any feature not specified for a vector is assumed to be 0-valued.
* All features must be numeric.
* Missing features can be used only in dense files and should be denoted with a single "?".

### Running a Test Set Classification
* run: java -cp project/target ClassificationDriver [sparse|dense] [test_feature_vectors_file] [training_feature_vectors_file] [training_labels_file] [number_of_folds] [output_labels_file] [GA-ODT|C-DT|DT]

### Running a Cross Validation Test
* run: java -cp project/target CVDriver [sparse|dense] [feature_vectors_file] [labels_file] [number_of_folds] [random_seed] [GA-ODT|C-DT|DT]

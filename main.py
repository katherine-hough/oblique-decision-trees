import sys, subprocess, re, time
import cart.classify
import numpy as np

def main():
    # List of tuples of the form (dataset_name, is_sparse?, num_instances, num_attributes, num_classes)
    datasets = [('arcene', False), ('breast-cancer', False, 286, 15, 2), ('dermatology', False),
        ('dorothea', True), ('farm-ads', True), ('iris', False), ('multiple-features', False),
        ('wine', False, 178, 13)]
    num_folds = 5 # Number of folds made for cross-validation
    random_seed = 484 # Seed used for the random number generator
    dataset = datasets[-1]

    # Compile the java code for the project
    compile_java = ['javac', '-Xlint:unchecked', '-d', 'project/target', 'project/src/*.java']
    ret_code = subprocess.call(compile_java, stdout=subprocess.DEVNULL)
    assert (ret_code==0),'Java code failed to compile.'

    # Build OC1
    build_oc1 = ['make', '-C', 'OC1', 'mktree']
    ret_code = subprocess.call(build_oc1, stdout=subprocess.DEVNULL)
    assert (ret_code==0),'Failed to build OC1.'

    # Create folds for the dataset
    # make_folds = create_folds_cmd(num_folds, random_seed, dataset)
    # ret_code = subprocess.call(make_folds, stdout=subprocess.DEVNULL)
    # assert (ret_code==0), f'Failed to create folds for {dataset[0]}.'

    # Run the CART implementation
    accuracies, elapsed_time = run_cart(num_folds, random_seed, dataset)
    print(f'CART --- {dataset[0]}')
    print(f'Accuracy: mean = {np.average(accuracies):.5}, std.dev = {np.std(accuracies):.5}')
    print(f'Elapsed Time (seconds): {elapsed_time}')

    # Run the GA-ODT implementation
    accuracies, elapsed_time = run_project_DT(num_folds, random_seed, dataset, 'GA-ODT')
    print(f'GA-ODT --- {dataset[0]}')
    print(f'Accuracy: mean = {np.average(accuracies):.5}, std.dev = {np.std(accuracies):.5}')
    print(f'Elapsed Time (seconds): {elapsed_time}')

    # Run the OC1 implementation
    accuracies, elapsed_time = run_oc1(num_folds, random_seed, dataset)
    print(f'OC1 --- {dataset[0]}')
    print(f'Accuracy: mean = {np.average(accuracies):.5}, std.dev = {np.std(accuracies):.5}')
    print(f'Elapsed Time (seconds): {elapsed_time}')

# Runs cross validation for the java implementations
def run_project_DT(num_folds, random_seed, dataset, method):
    cmd = create_project_DT_cmd(num_folds, random_seed, dataset, method)
    start_time = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, errs = p.communicate()
    elapsed_time = time.perf_counter()-start_time
    accuracies = []
    for line in output.decode('utf-8').split('\n'):
        if 'Accuracies' in line:
            accuracies = [float(x) for x in re.match('Accuracies: \\[(.*)\\]', line).group(1).split(",")]
            break
    return accuracies, elapsed_time

# Runs cross validation for the OC1 implementation
def run_oc1(num_folds, random_seed, dataset):
    start_time = time.perf_counter()
    outputs = []
    for cur_fold in range(1, num_folds+1):
        cmd = create_OC1_cmd(num_folds, random_seed, dataset, cur_fold)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output, errs = p.communicate()
        outputs.append(output.decode('utf-8').split('\n')[0])
    elapsed_time = time.perf_counter()-start_time
    accuracies = []
    for output in outputs:
        accuracies.append(float(re.match('accuracy = ([0-9]*\\.[0-9]*)', output).group(1)))
    return accuracies, elapsed_time

# Runs cross validation for the CART implementation
def run_cart(num_folds, random_seed, dataset):
    path = f'data/{dataset[0]}/folds/{num_folds}-folds/{dataset[0]}'
    start_time = time.perf_counter()
    accuracies = cart.classify.cross_validate(path, num_folds, dataset[1], random_seed)
    elapsed_time = time.perf_counter()-start_time
    return accuracies, elapsed_time

# Creates the list of arguments used to generate folds
def create_folds_cmd(num_folds, random_seed, dataset):
    data_file = f'data/{dataset[0]}/{dataset[0]}.data'
    label_file = f'data/{dataset[0]}/{dataset[0]}.labels'
    sparse_str = 'sparse' if dataset[1] else 'dense'
    return ['java', '-cp', 'project/target', 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), 'F']

# Creates the list of arguments used to run CVDriver
def create_project_DT_cmd(num_folds, random_seed, dataset, method):
    data_file = f'data/{dataset[0]}/{dataset[0]}.data'
    label_file = f'data/{dataset[0]}/{dataset[0]}.labels'
    sparse_str = 'sparse' if dataset[1] else 'dense'
    return ['java', '-cp', 'project/target', 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), method]

# Creates the list of arguments used to run OC1 on a single fold
def create_OC1_cmd(num_folds, random_seed, dataset, cur_fold):
    training_file = f'-tdata/{dataset[0]}/folds/{num_folds}-folds/{dataset[0]}{cur_fold}-train.data'
    test_file = f'-Tdata/{dataset[0]}/folds/{num_folds}-folds/{dataset[0]}{cur_fold}-test.data'
    return ['OC1/mktree', training_file, test_file, f'-s{random_seed}', '-z']

if __name__ == '__main__':
    main()

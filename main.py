import sys, subprocess, re, time
import cart.classify
import numpy as np

def main():
    # List of tuples of the form (dataset_name, is_sparse?, num_instances, num_attributes, num_classes)
    datasets = [('arcene', False, 200, 9961, 2), ('breast-cancer', False, 286, 15, 2),
        ('dermatology', False, 366, 34, 6), ('dorothea', True, 1150, 91598, 2),
        ('farm-ads', True, 4143, 54877, 2), ('iris', False, 150, 4, 3),
        ('multiple-features', False, 2000, 649, 10), ('wine', False, 178, 13, 3)]
    num_folds = 5 # Number of folds made for cross-validation
    random_seed = 484 # Seed used for the random number generator
    dataset = datasets[5]

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

    for dataset in datasets:
        print(f'-------+-------------------------{center_string(dataset[0], 13, "-")}-------+--------------------------')
        # Run the CART implementation
        accuracies, elapsed_time = run_cart(num_folds, random_seed, dataset)
        print(f'CART   | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

        # Run the GA-ODT implementation
        accuracies, elapsed_time = run_project_dt(num_folds, random_seed, dataset, 'GA-ODT')
        print(f'GA-ODT | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

        # Run the OC1 implementation
        if not dataset[1]:
            accuracies, elapsed_time = run_oc1(num_folds, random_seed, dataset)
            print(f'OC1    | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

def center_string(str, width, symbol):
    if len(str) >= width:
        return str
    left_padding = (width-len(str)+1)//2
    right_padding = (width-len(str))//2
    chars = [symbol for i in range(0, left_padding)]
    chars.extend([c for c in str])
    chars.extend([symbol for i in range(0, right_padding)])
    return ''.join(chars)
# Runs cross validation for the java implementations
def run_project_dt(num_folds, random_seed, dataset, method):
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
        accuracies.append(float(re.match('accuracy = ([0-9]*\\.[0-9]*)', output).group(1))/100.0)
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
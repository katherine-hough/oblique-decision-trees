import sys, subprocess, re, time, os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp
import cart.classify
import numpy as np

def main():
    # List of tuples of the form (dataset_name, is_sparse?, num_instances, num_attributes, num_classes)
    datasets = [('iris', False, 150, 4, 3), ('wine', False, 178, 13, 3),
        ('breast-cancer', False, 286, 15, 2), ('dermatology', False, 366, 34, 6),
        ('multiple-features', False, 2000, 649, 10), ('arcene', False, 200, 9961, 2),
        ('farm-ads', True, 4143, 54877, 2), ('dorothea', True, 1150, 91598, 2)]
    num_folds = 10 # Number of folds made for cross-validation
    random_seed = 484 # Seed used for the random number generator

    # datasets = datasets[:5]

    # Compile the java code for the project
    if(os.name == 'nt'):
        compile_java = ['javac', '-Xlint:unchecked', '-d', os.path.join('project', 'target'), os.path.join('project', 'src', '*.java')]
    else:
        srcs_file = open("sources.txt", "w")
        find_java_srcs = ['find', '-name', '*.java']
        subprocess.call(find_java_srcs, stdout=srcs_file)
        compile_java = ['javac', '-Xlint:unchecked', '-d', os.path.join('project', 'target'), '@sources.txt']
    ret_code = subprocess.call(compile_java, stdout=subprocess.DEVNULL)
    assert (ret_code==0),'Java code failed to compile.'

    # Build OC1
    build_oc1 = ['make', '-C', 'OC1', 'mktree']
    ret_code = subprocess.call(build_oc1, stdout=subprocess.DEVNULL)
    assert (ret_code==0),'Failed to build OC1.'

    # Create folds for the datasets
    # for dataset in datasets:
    #     make_folds = create_folds_cmd(num_folds, random_seed, dataset)
    #     ret_code = subprocess.call(make_folds, stdout=subprocess.DEVNULL)
    #     assert (ret_code==0), f'Failed to create folds for {dataset[0]}.'

    for dataset in datasets:
        print(f'-------+---------------------{center_string(dataset[0], 17, "-")}-------+--------------------------')
        # Run the CART implementation
        accuracies, elapsed_time = run_cart(num_folds, random_seed, dataset)
        print(accuracies)
        print(f'CART   | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

        # Run the OC1 implementation
        # if not dataset[1]:
        #     accuracies, elapsed_time = run_oc1(num_folds, random_seed, dataset)
        #     print(accuracies)
        #     print(f'OC1    | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

        # Run the C-DT implementation
        accuracies, elapsed_time = run_project_dt(num_folds, random_seed, dataset, 'C-DT')
        print(accuracies)
        print(f'C-DT   | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

        # Run the GA-ODT implementation
        accuracies, elapsed_time = run_project_dt(num_folds, random_seed, dataset, 'GA-ODT')
        print(accuracies)
        print(f'GA-ODT | Accuracy: mean = {np.average(accuracies):5.5f}, std.dev = {np.std(accuracies):5.5f} | Elapsed Time (s): {elapsed_time:5.5f}')

# Centers the specified string to the specified width by padding it with the specified
# symbol
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
    for cur_fold in range(1, num_folds+1):
        cmd = create_OC1_cmd(num_folds, random_seed, dataset, cur_fold)
        ret_code = subprocess.call(cmd, stdout=subprocess.DEVNULL)
        assert (ret_code==0),'Failed to run OC1.'
    elapsed_time = time.perf_counter()-start_time
    accuracies = []
    test_file = os.path.join('data', dataset[0], 'folds', f'{num_folds}-folds', f'{dataset[0]}{cur_fold}-test.data')
    for cur_fold in range(1, num_folds+1):
        accuracies.append(calc_balanced_accuracy(os.path.join('data', 'temp', f'oc1-{cur_fold}.txt'), test_file))
    return accuracies, elapsed_time

# Calculated the balanced accuracy for the specfied misclassification file and testing file
def calc_balanced_accuracy(misclas_file, test_file):
    misclas_lines = [line for line in cart.classify.read_file(misclas_file) if len(line) > 0]
    test_lines = [line for line in cart.classify.read_file(test_file) if len(line) > 0]
    positives = {}
    misses = {}
    for line in test_lines:
        label = line.split()[-1]
        if label in positives:
            positives[label]+=1
        else:
            positives[label]=1
    for line in misclas_lines:
        label = line.split()[-1]
        if label in misses:
            misses[label]+=1
        else:
            misses[label]=1
    bal_acc = 0
    for positive in positives:
        total = positives[positive]
        miss = misses[positive] if positive in misses else 0
        bal_acc += (total-miss)/total
    return bal_acc/len(positives)

# Runs cross validation for the CART implementation
def run_cart(num_folds, random_seed, dataset):
    path = os.path.join('data', dataset[0], 'folds', f'{num_folds}-folds', dataset[0])
    start_time = time.perf_counter()
    accuracies = cart.classify.cross_validate(path, num_folds, dataset[1], random_seed)
    elapsed_time = time.perf_counter()-start_time
    return accuracies, elapsed_time

# Creates the list of arguments used to generate folds
def create_folds_cmd(num_folds, random_seed, dataset):
    data_file = os.path.join('data', dataset[0], f'{dataset[0]}.data')
    label_file = os.path.join('data', dataset[0], f'{dataset[0]}.labels')
    sparse_str = 'sparse' if dataset[1] else 'dense'
    return ['java', '-cp', os.path.join('project', 'target'), 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), 'F']

# Creates the list of arguments used to run CVDriver
def create_project_DT_cmd(num_folds, random_seed, dataset, method):
    data_file = os.path.join('data', dataset[0], f'{dataset[0]}.data')
    label_file = os.path.join('data', dataset[0], f'{dataset[0]}.labels')
    sparse_str = 'sparse' if dataset[1] else 'dense'
    return ['java', '-cp', os.path.join('project', 'target'), 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), method]

# Creates the list of arguments used to run OC1 on a single fold
def create_OC1_cmd(num_folds, random_seed, dataset, cur_fold):
    training_file = os.path.join('-tdata', dataset[0], 'folds', f'{num_folds}-folds', f'{dataset[0]}{cur_fold}-train.data')
    test_file = os.path.join('-Tdata', dataset[0], 'folds', f'{num_folds}-folds', f'{dataset[0]}{cur_fold}-test.data')
    temp_file = os.path.join('-Mdata', 'temp', f'oc1-{cur_fold}.txt')
    return [os.path.join('.', 'OC1', 'mktree'), training_file, test_file, temp_file, f'-s{random_seed}', '-z']

if __name__ == '__main__':
    main()

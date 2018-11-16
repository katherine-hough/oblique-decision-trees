import sys, subprocess, os

def main():
    # List of tuples of the form (dataset_name, is_sparse?, num_dattributes, num_instances)
    datasets = [('arcene', False), ('breast-cancer', False), ('dermatology', False),
        ('dorothea', True), ('farm-ads', True), ('iris', False), ('multiple-features', False),
        ('wine', False)]
    FNULL = open(os.devnull, 'w') # Used to suppress output
    num_folds = 5 # Number of folds made for cross-validation
    random_seed = 484 # Seed used for the random number generator

    # Compile the java code for the project
    compile_java = ['javac', '-Xlint:unchecked', '-d', 'project/target', 'project/src/*.java']
    ret_code = subprocess.call(compile_java, stdout=FNULL, stderr=subprocess.STDOUT)
    assert (ret_code==0),'Java code failed to compile.'

    # Build OC1
    build_oc1 = ['make', '-C', 'OC1', 'mktree']
    ret_code = subprocess.call(build_oc1, stdout=FNULL, stderr=subprocess.STDOUT)
    assert (ret_code==0),'Failed to build OC1.'

    # Create folds for the datasets
    for dataset in datasets:
        make_folds = create_folds_cmd(num_folds, random_seed, dataset[0], dataset[1])
        ret_code = subprocess.call(make_folds, stdout=FNULL, stderr=subprocess.STDOUT)
        assert (ret_code==0), f'Failed to create folds for {dataset[0]}.'

def create_folds_cmd(num_folds, random_seed, dataset, sparse):
    data_file = f'data/{dataset}/{dataset}.data'
    label_file = f'data/{dataset}/{dataset}.labels'
    sparse_str = 'sparse' if sparse else 'dense'
    return ['java', '-cp', 'project/target', 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), 'F']

def create_project_DT_cmd(num_folds, random_seed, dataset, sparse, method):
    data_file = f'data/{dataset}/{dataset}.data'
    label_file = f'data/{dataset}/{dataset}.labels'
    sparse_str = 'sparse' if sparse else 'dense'
    return ['java', '-cp', 'project/target', 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), method]


if __name__ == '__main__':
    main()

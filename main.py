import sys, subprocess, os

def main():
    # List of tuples of the dataset names and whether they are sparsely represented
    datasets = [('arcene', False), ('breast-cancer', False), ('dermatology', False),
        ('dorothea', True), ('farm-ads', True), ('iris', False), ('multiple-features', False),
        ('wine', False)]
    FNULL = open(os.devnull, 'w') # Used to suppress output
    num_folds = 5 # Number of folds made for cross-validation
    random_seed = 484 # Seed used for the random number generator
    compile_java = ['javac', '-Xlint:unchecked', '-d', 'project/target', 'project/src/*.java']
    build_oc1 = ['make', '-C', 'OC1', 'mktree']

    # Compile the java code for the project
    ret_code = subprocess.call(compile_java, stdout=FNULL, stderr=subprocess.STDOUT)
    assert (ret_code==0),'Java code failed to compile.'
    # Build OC1
    ret_code = subprocess.call(build_oc1, stdout=FNULL, stderr=subprocess.STDOUT)
    assert (ret_code==0),'Failed to build OC1.'

def create_fold(num_folds, dataset, sparse):
    data_file = f'data/{dataset}/{dataset}.data'
    label_file = f'data/{dataset}/{dataset}.labels'
    sparse_str = 'sparse' if sparse else 'dense'
    create_folds = ['java', '-cp', 'project/target', 'CVDriver', sparse_str, data_file, label_file, str(num_folds), str(random_seed), '-f']
)

if __name__ == '__main__':
    main()

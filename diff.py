import sys

def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    data1 = read_file(filename1)
    data2 = read_file(filename2)
    dif = 0
    for i in range(0, len(data1)):
        s = data1[i]
        if(data1[i] != data2[i]):
            dif += 1
    print(filename1 + " & " + filename2 + " line difference: " + str(dif) + "/" + str(len(data1)) + " = " + str(1.0*dif/len(data1)))

# Returns the contents of the specified file.
def read_file(filename):
    with open(filename) as file:
        return file.read().split('\n')

if __name__ == '__main__':
    main()

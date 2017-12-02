import config


def compare_reference_set(unsupervised_res):
    correct = 0
    for i in range(config.data_size):
        if (i + 1) in config.reference_set:
            if unsupervised_res[i] == 1:
                correct += 1
    if correct > len(config.reference_set) / 2:
        return 0
    return 1


def read_first_lines(filename, limit):
    result = []
    with open(filename, 'r') as input_file:
        for line_number, line in enumerate(input_file):
            if line_number > limit:  # line_number starts at 0.
                break
            result.append(line)
    return result


def compare_results():
    size = config.data_size
    unsupervised_res = read_first_lines('unsupervised.txt', size)
    expected = read_first_lines('train_labels.csv', size)

    inverse = compare_reference_set(unsupervised_res)

    correct = 0
    for i in range(size):
        if inverse == 0 and unsupervised_res[i] == expected[i]:
            correct += 1
        if inverse == 1 and unsupervised_res[i] != expected[i]:
            correct += 1

    acc = correct / size
    print("Accuracy : " + str(acc))
import config


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

    correct = 0
    for i in range(size):
        if unsupervised_res[i] == expected[i]:
            correct += 1

    acc = correct / size
    if acc < 0.5:
        acc = 1 - acc
        print(acc)
        print('Using inverted labels')
    else:
        print(acc)
        print('Using same labels')

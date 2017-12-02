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

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    correct = 0
    for i in range(size):
        if inverse == 0:
            res = int(unsupervised_res[i])
            exp = int(expected[i])
            if res == exp:
                correct += 1
            if exp == 1 and res == 1:
                true_positives += 1
            if exp == 0 and res == 0:
                true_negatives += 1
            if exp == 1 and res == 0:
                false_negatives += 1
            if exp == 0 and res == 1:
                false_positives += 1
        if inverse == 1:
            res = int(unsupervised_res[i])
            exp = int(expected[i])
            if res != exp:
                correct += 1
            if exp == 1 and res == 1:
                true_negatives += 1
            if exp == 0 and res == 0:
                true_positives += 1
            if exp == 1 and res == 0:
                false_positives += 1
            if exp == 0 and res == 1:
                false_negatives += 1

    acc = correct / size
    print("Accuracy : " + str(acc))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))

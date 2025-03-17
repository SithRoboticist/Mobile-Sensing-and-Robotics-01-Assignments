import numpy as np

def sum_nested_list(nested_list):
    total_sum = 0

    for sublist in nested_list:
        for item in sublist:
            total_sum += item
    
    print("Sum of all entries:", total_sum)
    
    return total_sum


def mean_std(dataset):
    mean = np.mean(dataset)
    std = np.std(dataset)
    
    print("Distribution:")
    print("Mean:", mean)
    print("Standard Deviation:", std)
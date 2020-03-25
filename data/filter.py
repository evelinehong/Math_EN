# remove data with parentheses and sort by template length

import json
from scipy import stats

files = ["test23k_processed", "train23k_processed", "valid23k_processed"]

for file in files:
    with open(file + ".json", "r") as read_file:
        data = json.load(read_file)
        filtered = filter(lambda item: item["id"] != "22279", data)
        filtered = sorted(filtered, key=lambda item: len(item["target_template"]))
        print(f"{file} length: {len(data)} -> {len(filtered)}")
        with open(file + "_sorted.json", "w") as write_file:
            json.dump(filtered, write_file, indent=4)

        lengths = list(map(lambda item: len(item["target_template"]), filtered))
        print(stats.describe(lengths))

'''
REUSLT:
    test23k_processed length: 1000 -> 457
    DescribeResult(nobs=457, minmax=(3, 21), mean=6.571115973741795, variance=2.7981112518714726, skewness=2.3329025698369232, kurtosis=13.727172273854961)
    train23k_processed length: 21162 -> 10653
    DescribeResult(nobs=10653, minmax=(3, 43), mean=6.526893832723177, variance=3.0596643744766054, skewness=3.2394047110250583, kurtosis=33.20115430147103)
    valid23k_processed length: 1000 -> 484
    DescribeResult(nobs=484, minmax=(3, 19), mean=6.487603305785124, variance=2.7017093578358407, skewness=2.2509410258607687, kurtosis=13.395730085615956)
'''
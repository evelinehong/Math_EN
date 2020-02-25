# remove data with parentheses and sort by template length

import json

files = ["test23k_processed", "train23k_processed", "valid23k_processed"]

for file in files:
    with open(file + ".json", "r") as read_file:
        data = json.load(read_file)
        filtered = filter(lambda item: "(" not in item["target_template"], data)
        filtered = sorted(filtered, key=lambda item: len(item["target_template"]))
        print(f"{file} length: {len(data)} -> {len(filtered)}")
        with open(file + "_filtered.json", "w") as write_file:
            json.dump(filtered, write_file, indent=4) 
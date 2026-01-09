import json
import argparse
from pathlib import Path


def parser_args():
    parser = argparse.ArgumentParser(
        description="Combine caption and generated text into a single JSONL file."
    )
    parser.add_argument(
        "--caption_file",
        type=Path,
        nargs='+',
        required=True,
        help="Path to the JSONL file containing captions.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output JSONL file.",
    )
    return parser.parse_args()

def load_json(file_path):
    if "backup" not in str(file_path):
        data = json.load(open(file_path, 'r'))
    else:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
                
    dict_data = {}
    for item in data:
        if item["image_id"] in dict_data and item["question"] in dict_data:
            continue
        else:
            dict_data[item["image_id"]] = item
            dict_data[item["question"]] = item

    for i in data:
        if i["image_id"] in dict_data and i["question"] in dict_data:
            continue
        else:
            print(i["image_id"], i["question"])
            data.remove(i)
    
    return list(data)

def main():
    args = parser_args()
    combined_data = []
    
    for caption_file in args.caption_file:
        data_captions = load_json(caption_file)
        combined_data.extend(data_captions)
    
    json.dump(combined_data, open(args.output_file, 'w'), indent=4)
    
if __name__ == "__main__":
    main()

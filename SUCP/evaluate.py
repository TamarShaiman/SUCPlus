import os
import re

import numpy as np


def extract_metrics(filename, top_x):
    metric_names = ["precision_{}", "recall_{}", "nDCG_{}", "MAP_{}"]
    metrics = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None  # Return None if file is empty
        
        last_line = lines[-1].strip().split('\t')
        if len(last_line) < 6:
            return None  # Return None if last line is malformed
        
        extracted_metrics = {}
        for i, metric_template in enumerate(metric_names, start=2):
            if i < len(last_line):
                match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", last_line[i])
                if match:
                    metric_key = metric_template.format(top_x)
                    extracted_metrics[metric_key] = list(map(float, match))
        
        return extracted_metrics

def process_all_results(directory):
    results = {}
    pattern = re.compile(r"result_top_(\d+)\.txt")
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            top_x = int(match.group(1))
            file_path = os.path.join(directory, filename)
            extracted = extract_metrics(file_path, top_x)
            if extracted:
                results.update({key: np.mean(val) if val else None for key, val in extracted.items()})
    
    return results

if __name__ == "__main__":
    directory = "/Users/tamar/Desktop/Uni/Rec Sys/project_RS/result2"  # Replace with actual directory path
    # directory = "path_to_results_directory"  # Replace with actual directory path
    all_averages = process_all_results(directory)
    
    for metric, avg in sorted(all_averages.items()):
        print("{}: {:.4f}".format(metric, avg) if avg is not None else "{}: No data".format(metric))
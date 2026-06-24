import csv

def log_metrics(file, row):
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
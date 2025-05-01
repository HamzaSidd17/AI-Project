
import csv

def count_rows_in_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip header row if exists
        next(csvreader, None)  
        row_count = sum(1 for row in csvreader)
    return row_count

print(count_rows_in_csv('carState.csv'))
import os

log_path = 'incremental_log.txt'
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-16le') as f:
        print(f.read())
else:
    print("Log not found")

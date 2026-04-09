import os

def fix_bible(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace literal \n with real newlines
    fixed = content.replace('\\n', '\n')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(fixed)

if __name__ == "__main__":
    fix_bible(r'c:\Users\reddy\OneDrive\Documents\TSA-PROJECT\EXPERT_BIBLE.md')
    print("Fixed Expert Bible line endings.")

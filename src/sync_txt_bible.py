import re
import os

def md_to_txt(md_path, txt_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple stripping of some markdown syntax
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)
    content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'>\s+', '', content)
    content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
    content = re.sub(r'>\s+\[\!.*?\]', '', content)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    md_to_txt(r'c:\Users\reddy\OneDrive\Documents\TSA-PROJECT\EXPERT_BIBLE.md', r'c:\Users\reddy\OneDrive\Documents\TSA-PROJECT\EXPERT_BIBLE.txt')
    print("Synchronized EXPERT_BIBLE.txt with EXPERT_BIBLE.md")

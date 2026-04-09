import re
import os

def md_to_txt(md_path, txt_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple stripping of some markdown syntax
    # Remove bold/italic
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)
    # Remove headers symbols but keep text
    content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
    # Remove blockquotes
    content = re.sub(r'>\s+', '', content)
    # Remove link syntax, keep text [text](url) -> text
    content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
    # Remove alerts > [!NOTE]
    content = re.sub(r'>\s+\[\!.*?\]', '', content)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    md_to_txt(r'c:\Users\reddy\OneDrive\Documents\TSA-PROJECT\Project_Report.md', r'c:\Users\reddy\OneDrive\Documents\TSA-PROJECT\Project_Report.txt')
    print("Synchronized Project_Report.txt with Project_Report.md")

# msc/tools/filesystem.py
import os
from typing import Dict, List
from rich.console import Console

console = Console()

class FilesystemTool:
    @staticmethod
    def _extract_clean_code(content: str, file_path: str) -> str:
        """Extract clean code by removing markdown code fences and artifacts"""
        if not content.strip():
            return content
            
        # Detect file type
        is_python = file_path.endswith('.py')
        is_dockerfile = 'Dockerfile' in file_path or file_path.endswith('.dockerfile')
        
        lines = content.split('\n')
        cleaned_lines = []
        in_code_block = False
        skip_fences = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Handle code fence markers
            if stripped.startswith('```python') or stripped.startswith('````python'):
                if is_python:
                    skip_fences = True
                    continue
            elif stripped.startswith('```dockerfile') or stripped.startswith('````dockerfile'):
                if is_dockerfile:
                    skip_fences = True
                    continue
            elif stripped.startswith('```') or stripped.startswith('````'):
                if skip_fences:
                    break  # End of code block
                continue
            
            # If no fences detected, assume all content is valid
            if not any('```' in l for l in lines):
                cleaned_lines.append(line)
            elif skip_fences or not any('```' in l for l in lines):
                cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines).strip()
        
        # Validate content makes sense
        if is_python and cleaned_content:
            # Should have valid Python syntax indicators
            if not any(keyword in cleaned_content for keyword in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'try:', 'print(', '=']):
                console.log(f"⚠️ [FileSystem] Warning: Python file may have extraction issues")
        elif is_dockerfile and cleaned_content:
            # Should start with FROM
            if not cleaned_content.startswith('FROM'):
                console.log(f"⚠️ [FileSystem] Warning: Dockerfile doesn't start with FROM")
        
        return cleaned_content

    @staticmethod
    def write_file(path: str, content: str) -> bool:
        try:
            # Automatically clean code content
            cleaned_content = FilesystemTool._extract_clean_code(content, path)
            
            if cleaned_content != content:
                console.log(f"🧹 [FileSystem] Cleaned markdown artifacts from {os.path.basename(path)}")
            
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            console.log(f"✅ [FileSystem] Wrote file: {path}")
            return True
        except IOError as e:
            console.log(f"❌ [FileSystem] Error writing file {path}: {e}")
            return False

    @staticmethod
    def read_directory_contents(directory_path: str, include_extensions: List[str] = ['.py', '.json', '.txt', '.md']) -> Dict[str, str]:
        context = {}
        for root, _, files in os.walk(directory_path):
            if 'venv' in root or '__pycache__' in root or '.git' in root:
                continue
            for file in files:
                if any(file.endswith(ext) for ext in include_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            context[file_path] = f.read()
                    except (IOError, UnicodeDecodeError):
                        context[file_path] = "[Error reading file content]"
        return context
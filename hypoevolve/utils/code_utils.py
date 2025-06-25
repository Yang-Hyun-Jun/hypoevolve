"""
Code processing utilities for HypoEvolve
"""

import re
from typing import List, Tuple


def apply_diff(original_code: str, diff_text: str) -> str:
    """Apply a SEARCH/REPLACE diff to code"""

    # Extract diff blocks
    diff_blocks = extract_diff_blocks(diff_text)

    if not diff_blocks:
        return original_code

    result = original_code

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        # Clean up the search and replace text
        search_text = search_text.strip()
        replace_text = replace_text.strip()

        if search_text in result:
            result = result.replace(search_text, replace_text)

    return result


def extract_diff_blocks(diff_text: str) -> List[Tuple[str, str]]:
    """Extract SEARCH/REPLACE blocks from diff text"""

    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, diff_text, re.DOTALL)

    return [(search.strip(), replace.strip()) for search, replace in matches]


def extract_code_from_response(response: str) -> str:
    """Extract code from LLM response, handling markdown code blocks"""

    # First try to extract from markdown code blocks
    code_block_pattern = r"```(?:python|py)?\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # If no code blocks, return the response as-is
    return response.strip()


def detect_language(code: str) -> str:
    """Simple language detection"""

    if re.search(r"^\s*(def|class|import|from)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^\s*(function|var|let|const)\s", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^\s*(public|private|class).*\{", code, re.MULTILINE):
        return "java"
    else:
        return "python"  # Default to python


def clean_code(code: str) -> str:
    """Clean and normalize code"""

    # Remove excessive whitespace
    lines = code.split("\n")
    cleaned_lines = []

    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        cleaned_lines.append(line)

    # Remove excessive empty lines
    result = []
    prev_empty = False

    for line in cleaned_lines:
        if line.strip() == "":
            if not prev_empty:
                result.append(line)
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False

    return "\n".join(result).strip()


def calculate_similarity(code1: str, code2: str) -> float:
    """Calculate simple similarity between two code snippets"""

    if code1 == code2:
        return 1.0

    # Normalize both codes by removing extra whitespace
    normalized1 = re.sub(r"\s+", " ", code1.strip())
    normalized2 = re.sub(r"\s+", " ", code2.strip())

    if normalized1 == normalized2:
        return 0.95  # Very similar but not identical

    # Simple line-based similarity
    lines1 = set(line.strip() for line in code1.split("\n") if line.strip())
    lines2 = set(line.strip() for line in code2.split("\n") if line.strip())

    if not lines1 and not lines2:
        return 1.0

    if not lines1 or not lines2:
        return 0.0

    intersection = len(lines1 & lines2)
    union = len(lines1 | lines2)

    # Calculate Jaccard similarity
    jaccard = intersection / union if union > 0 else 0.0

    # Also calculate character-level similarity as a fallback
    if jaccard == 0.0:
        # Use simple character overlap
        chars1 = set(c for c in normalized1 if c.isalnum())
        chars2 = set(c for c in normalized2 if c.isalnum())

        if chars1 and chars2:
            char_intersection = len(chars1 & chars2)
            char_union = len(chars1 | chars2)
            char_similarity = char_intersection / char_union if char_union > 0 else 0.0
            return max(
                jaccard, char_similarity * 0.5
            )  # Weight character similarity less

    return jaccard


def validate_code_syntax(code: str, language: str = "python") -> bool:
    """Basic syntax validation"""

    if language == "python":
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    # For other languages, just check basic structure
    return len(code.strip()) > 0

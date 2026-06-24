import pandas as pd
import json
import sys
import os
from typing import Set, Dict, List, Optional
import re

# ==================== HARDCODED OUTPUT PATH ====================
OUTPUT_FILE = "/raid/home/rizwank/Normalization/data_generation/PreProcessed_Data_tagged/OP_21_MONEY_only_1.txt"
# ===============================================================

def extract_sentences_by_category(
    csv_path: str,
    target_category: str,
    max_sentences: int = 30000,
    chunk_size: int = 200_000,
    pure_only: bool = True  # True = only target, False = target + at least one other
) -> List[Dict[str, str]]:
    """
    Extract sentences containing a specific target category.
    
    Args:
        csv_path: Path to CSV file
        target_category: Category to filter by (e.g., "DATE", "MONEY", "TIME")
        max_sentences: Maximum number of sentences to extract
        chunk_size: Chunk size for reading CSV
        pure_only: If True, only sentences with ONLY this category
                   If False, sentences must have this category PLUS at least one other
    
    Returns:
        List of dictionaries with "original" and "tagged" keys
    """
    TARGET_CATS = {
        "DATE", "MEASURE", "CARDINAL", "ORDINAL",
        "FRACTION", "TIME", "TELEPHONE", "MONEY", "DECIMAL"
    }
    
    # Validate target category
    target_category = target_category.upper()
    if target_category not in TARGET_CATS:
        raise ValueError(f"Invalid category: {target_category}. Must be one of: {TARGET_CATS}")
    
    # Special case: treat ORDINAL as CARDINAL
    effective_category = "CARDINAL" if target_category == "ORDINAL" else target_category
    
    sentences = []
    current_sentence_tokens = []
    current_ops_tokens = []
    sentence_count = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            token = str(row["Input Token"]).strip()
            cat = str(row["Semiotic Class"]).strip().upper()
            
            # Convert ORDINAL to CARDINAL
            if cat == "ORDINAL":
                cat = "CARDINAL"
            
            # Store real sentence tokens
            if token != "<eos>":
                current_sentence_tokens.append(token)
            
            # Build tagged sentence tokens
            if token == "<eos>":
                # Collect all unique categories in this sentence
                categories_in_sentence = set()
                for tok in current_ops_tokens:
                    if tok.startswith("<") and tok.endswith(">"):
                        # Extract category from tag like "<token><CATEGORY>"
                        parts = tok.split("><")
                        if len(parts) == 2:
                            category = parts[1].rstrip(">")
                            categories_in_sentence.add(category)
                
                # Check if target category is present
                has_target = effective_category in categories_in_sentence
                
                # Apply filtering logic based on mode
                include_sentence = False
                
                if has_target:
                    if pure_only:
                        # PURE mode: sentence should have ONLY the target category
                        if categories_in_sentence == {effective_category}:
                            include_sentence = True
                    else:
                        # MIXED mode: sentence must have target category + at least one other
                        if len(categories_in_sentence) >= 2:
                            include_sentence = True
                
                if include_sentence:
                    # Keep ALL tags from TARGET_CATS in the output
                    filtered_ops = []
                    for tok in current_ops_tokens:
                        # Check if this is a tagged token
                        if tok.startswith("<") and tok.endswith(">"):
                            # Extract category from tag
                            parts = tok.split("><")
                            if len(parts) == 2:
                                category = parts[1].rstrip(">")
                                # Only include if it's in our TARGET_CATS
                                if category in TARGET_CATS:
                                    filtered_ops.append(tok)
                                else:
                                    # Extract just the token part for non-target categories
                                    token_part = parts[0].lstrip("<")
                                    filtered_ops.append(token_part)
                            else:
                                filtered_ops.append(tok)
                        else:
                            # Include non-tagged tokens as-is
                            filtered_ops.append(tok)
                    
                    sentences.append({
                        "original": " ".join(current_sentence_tokens),
                        "tagged": " ".join(filtered_ops)
                    })
                    
                    sentence_count += 1
                    
                    if sentence_count >= max_sentences:
                        return sentences
                
                # Reset for next sentence
                current_sentence_tokens = []
                current_ops_tokens = []
                continue
            
            # If category is one of the target categories → tag it
            if cat in TARGET_CATS:
                tagged = f"<{token}><{cat}>"
                current_ops_tokens.append(tagged)
            else:
                # keep plain token in ops
                current_ops_tokens.append(token)
    
    return sentences

def save_to_txt(data: List[Dict[str, str]], output_path: str) -> None:
    """
    Save data to TXT file where each line is a dictionary string.
    Creates directories if they don't exist.
    
    Args:
        data: List of dictionaries to save
        output_path: Output file path
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Convert dictionary to string representation
            line_str = str(item)
            f.write(line_str + '\n')
    print(f"Saved {len(data)} sentences to {output_path}")

def main():
    """
    Main function to run from command line with arguments.
    Usage: python script.py <category> <max_sentences> [--mixed]
    
    PURE mode: Only sentences with ONLY the target tag (no other tags)
    MIXED mode: Sentences with the target tag PLUS at least one other tag
    
    Note: ORDINAL tags are automatically converted to CARDINAL tags
    """
    # Default values
    csv_path = "/raid/home/rizwank/Normalization_Data_Unprocessed/output_21.csv"    ########## input file path
    
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <category> <max_sentences> [--mixed]")
        print("Example 1: python script.py MEASURE 10000                      # Pure MEASURE only")
        print("Example 2: python script.py MEASURE 10000 --mixed              # MEASURE + at least one other tag")
        print(f"Available categories: DATE, MEASURE, CARDINAL, ORDINAL, FRACTION, TIME, TELEPHONE, MONEY, DECIMAL")
        print("\nMode:")
        print("  PURE (default): Only sentences with ONLY the specified tag")
        print("  MIXED (--mixed): Sentences with the target tag PLUS at least one other tag")
        print("\nNote: ORDINAL is automatically converted to CARDINAL")
        print(f"\nOutput will be saved to hardcoded path: {OUTPUT_FILE}")
        sys.exit(1)
    
    target_category = sys.argv[1].upper()
    try:
        max_sentences = int(sys.argv[2])
    except ValueError:
        print(f"Error: max_sentences must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    # Check for --mixed flag
    pure_only = True  # Default: pure-only mode
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--mixed":
            pure_only = False
    
    try:
        print(f"Extracting up to {max_sentences} sentences with category: {target_category}")
        print(f"Mode: {'PURE-ONLY (only target tag)' if pure_only else 'MIXED (target + at least one other tag)'}")
        print(f"Reading from: {csv_path}")
        print(f"Output will be saved to hardcoded path: {OUTPUT_FILE}")
        
        # Check if output directory exists, create if it doesn't
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory created/verified: {output_dir}")
        
        sentences = extract_sentences_by_category(
            csv_path=csv_path,
            target_category=target_category,
            max_sentences=max_sentences,
            pure_only=pure_only
        )
        
        print(f"Found {len(sentences)} sentences with category '{target_category}'")
        
        if sentences:
            save_to_txt(sentences, OUTPUT_FILE)
            
            # Show first 3 examples
            print("\nFirst 3 examples (as they appear in the file):")
            for i, s in enumerate(sentences[:3]):
                print(f"\nLine {i+1}:")
                print(f"  {str(s)}")
                
            # Also show formatted versions
            print("\n\nFormatted view of first 3 sentences:")
            for i, s in enumerate(sentences[:3]):
                print(f"\nExample {i+1}:")
                print(f"  Original: {s['original']}")
                print(f"  Tagged: {s['tagged']}")
            
            # Analyze what categories appear in the output
            all_categories = set()
            for s in sentences:
                tagged = s['tagged']
                # Extract categories from tagged text
                tags = re.findall(r'<[^>]+><([A-Z]+)>', tagged)
                all_categories.update(tags)
            
            # Show statistics about the filtered data
            print(f"\n{'='*60}")
            print("FILTERING STATISTICS:")
            print(f"  Total sentences found: {len(sentences)}")
            print(f"  Target category: {target_category}")
            if pure_only:
                print(f"  Mode: PURE-ONLY (only sentences with ONLY '{target_category}' tags)")
            else:
                print(f"  Mode: MIXED (sentences with '{target_category}' + at least 1 other tag)")
                print(f"  Categories found in output: {sorted(all_categories)}")
            print(f"  Output file: {OUTPUT_FILE}")
            
            # Check if file was created successfully
            if os.path.exists(OUTPUT_FILE):
                file_size = os.path.getsize(OUTPUT_FILE)
                print(f"  File size: {file_size:,} bytes")
            print(f"{'='*60}")
        else:
            print(f"No sentences found with category '{target_category}'")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Alternative: Function-based usage
def run_extraction(
    csv_path: str,
    target_category: str,
    max_sentences: int = 30000,
    pure_only: bool = True
) -> str:
    """
    Run extraction and save to hardcoded output file.
    
    Args:
        csv_path: Path to CSV file
        target_category: Category to filter by
        max_sentences: Maximum number of sentences to extract
        pure_only: If True, only sentences with ONLY this category
                   If False, sentences must have this category PLUS at least one other
    
    Returns:
        Path to output file (hardcoded)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    sentences = extract_sentences_by_category(
        csv_path=csv_path,
        target_category=target_category,
        max_sentences=max_sentences,
        pure_only=pure_only
    )
    
    save_to_txt(sentences, OUTPUT_FILE)
    return OUTPUT_FILE

# Test function to demonstrate the difference
def test_filter_difference():
    """
    Test to show the difference between PURE and MIXED modes.
    """
    print("TESTING FILTER MODES:")
    print("="*60)
    
    test_cases = [
        ("Meeting on Jan 15 at 3 PM", ["<Jan 15><DATE>", "<3 PM><TIME>"]),
        ("Price is $20 for Jan 15", ["<$20><MONEY>", "<Jan 15><DATE>"]),
        ("Only date: January 15", ["<January 15><DATE>"]),
        ("Just plain text", []),
        ("Multiple: Jan 15, 3 PM, $20", ["<Jan 15><DATE>", "<3 PM><TIME>", "<$20><MONEY>"]),
        ("Measure 5 kg at $10", ["<5 kg><MEASURE>", "<$10><MONEY>"]),
        ("Only measure: 10 meters", ["<10 meters><MEASURE>"]),
        ("Distance 5.5 km in 30 minutes", ["<5.5><DECIMAL>", "<km><MEASURE>", "<30 minutes><TIME>"]),
    ]
    
    for sentence, tags in test_cases:
        print(f"\nSentence: '{sentence}'")
        print(f"Tags: {tags}")
        
        # Extract unique categories
        categories = set()
        for tag in tags:
            if tag:
                parts = tag.split("><")
                if len(parts) == 2:
                    category = parts[1].rstrip(">")
                    categories.add(category)
        
        # Check for MEASURE (example target)
        has_measure = "MEASURE" in categories
        has_other = len(categories) > 1 and has_measure
        
        if has_measure and len(categories) == 1:
            print(f"  PURE mode: INCLUDED (only MEASURE)")
            print(f"  MIXED mode: EXCLUDED (no other tags)")
            print(f"  Output would show: {tags}")
        elif has_measure and has_other:
            print(f"  PURE mode: EXCLUDED (mixed tags)")
            print(f"  MIXED mode: INCLUDED (MEASURE + {categories - {'MEASURE'}})")
            print(f"  Output would show: {tags}")
        else:
            print(f"  PURE mode: EXCLUDED (no MEASURE)")
            print(f"  MIXED mode: EXCLUDED (no MEASURE)")

if __name__ == "__main__":
    # Run from command line
    main()
    
    # Or use programmatically:
    """
    # Example 1: Extract PURE MEASURE sentences (only MEASURE, no other tags)
    csv_path = "/raid/home/rizwank/Normalization_Data_Unprocessed/output_1.csv"
    
    # Extract PURE MEASURE sentences (default mode) - saves to hardcoded OUTPUT_FILE
    output_file1 = run_extraction(
        csv_path=csv_path,
        target_category="MEASURE",
        max_sentences=10000,
        pure_only=True  # Default
    )
    
    # Extract MIXED MEASURE sentences (MEASURE + at least one other tag)
    output_file2 = run_extraction(
        csv_path=csv_path,
        target_category="MEASURE",
        max_sentences=10000,
        pure_only=False
    )
    
    # Run test to understand the difference
    test_filter_difference()
    """
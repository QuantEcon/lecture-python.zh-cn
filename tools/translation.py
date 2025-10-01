"""
Before running the file, config the environment by adding the Anthropic API key:

echo "export ANTHROPIC_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
echo $ANTHROPIC_API_KEY # Test to see if it is added
"""

import anthropic
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import hashlib
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('translation.log')
    ]
)

def get_file_hash(filepath):
    """Calculate MD5 hash of file content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_translation_history():
    """Load translation history from JSON file."""
    history_file = 'translation_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return {}

def save_translation_history(history):
    """Save translation history to JSON file."""
    with open('translation_history.json', 'w') as f:
        json.dump(history, f, indent=2)

def needs_translation(input_file, history):
    """Check if file needs translation based on hash and modification time."""
    translated_file = input_file.replace('.md', '_cn.md')
    
    # If translated file doesn't exist, needs translation
    if not os.path.exists(translated_file):
        return True
        
    current_hash = get_file_hash(input_file)
    last_mod_time = os.path.getmtime(input_file)
    
    # Check if file has been modified since last translation
    if input_file in history:
        if (history[input_file]['hash'] != current_hash or 
            history[input_file]['last_modified'] < last_mod_time):
            return True
    else:
        return True
        
    return False

def process_file(filename):
    try:
        input_file = os.path.join(directory, filename)
        logging.info(f'Processing {input_file}')
        if os.path.isfile(input_file):
            translate_cn(input_file)
            
            # Update translation history after successful translation
            current_hash = get_file_hash(input_file)
            last_mod_time = os.path.getmtime(input_file)
            translation_history[input_file] = {
                'hash': current_hash,
                'last_modified': last_mod_time,
                'last_translated': time.time()
            }
            save_translation_history(translation_history)
            
            return f"Successfully processed {filename}"
        return f"Skipped {filename} - not a file"
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return f"Failed to process {filename}: {str(e)}"

def split_text(content, chunk_size=3000):
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size

        # If we are at the end of the content, just append the rest
        if end >= len(content):
            chunks.append(content[start:])
            break

        # Find the nearest line break before the chunk size
        next_line_break = content.rfind('\n', start, end)
        if next_line_break == -1:
            # If no line break is found within the chunk size, extend to the end
            next_line_break = end

        # Check if a code cell starts within the chunk
        code_cell_start = content.find('```{code-cell}', start, next_line_break)
        if code_cell_start != -1:
            # If a code cell starts, find its end
            code_cell_end = content.find('```', code_cell_start + 14)
            if code_cell_end != -1:
                # Move the end to the end of the code cell
                next_line_break = content.find('\n', code_cell_end) + 1

        chunks.append(content[start:next_line_break].strip())
        start = next_line_break

    return chunks

def translate_cn(input_file):
    # Initialize the Anthropic client
    client = anthropic.Anthropic()

    # Read the content of the input markdown file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into chunks
    chunks = split_text(content, chunk_size=1500)  # Using smaller chunks for better reliability
    logging.info(f"Split {input_file} into {len(chunks)} chunks")

    # Create the output file name
    output_file = input_file.replace('.md', '_cn.md')
    
    # Create or clear the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("")
    
    for i, chunk in enumerate(chunks):
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logging.info(f"\nProcessing chunk {i+1}/{len(chunks)} of {input_file} (length: {len(chunk)} chars)")
                logging.debug(f"Chunk preview: {chunk[:100]}...")
                
                # Create message with Claude
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=4000,
                    temperature=0,
                    system="You are a professional translator. Translate the given text into simplified Chinese. Maintain all markdown syntax, code blocks, and directives exactly as they are. Only output the direct translation without any explanations or system messages.",
                    messages=[
                        {
                            "role": "user",
                            "content": chunk
                        }
                    ]
                )
                
                response_text = message.content[0].text
                logging.info(f"Received response for chunk {i+1} (length: {len(response_text)} chars)")
                logging.debug(f"Response preview: {response_text[:100]}...")
                
                # Append this chunk's translation to the output file immediately
                with open(output_file, 'a', encoding='utf-8') as file:
                    file.write(response_text + "\n\n")  # Add an extra newline for spacing between chunks
                    
                logging.info(f"Wrote chunk {i+1} translation to {output_file}")
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                logging.error(f"Attempt {retry_count} failed for chunk {i+1} in {input_file}: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    logging.error(f"Failed to translate chunk after {max_retries} attempts")
                    raise

    logging.info(f"All chunks translated and saved to {output_file}")

if __name__ == "__main__":
    directory = "../lectures"
    max_workers = 2  # Adjust this based on your API rate limits and system capabilities
    
    # Load translation history
    translation_history = load_translation_history()
    
    # Get all markdown files and filter out already translated ones
    all_files = [f for f in os.listdir(directory) if f.endswith('.md') and os.path.isfile(os.path.join(directory, f))]
    files_to_translate = []
    
    for file in all_files:
        if file.endswith('_cn.md'):
            continue  # Skip already translated files
        
        input_file = os.path.join(directory, file)
        if needs_translation(input_file, translation_history):
            files_to_translate.append(file)
        else:
            logging.info(f"Skipping {file} - already translated and unchanged")
    
    if not files_to_translate:
        logging.info("No new files to translate!")
        exit(0)
        
    logging.info(f'Files to translate: {files_to_translate}')
    
    # Keep track of failed files
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files_to_translate}
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                logging.info(f"Result for {file}: {result}")
                if "Failed to process" in result:
                    failed_files.append(file)
            except Exception as e:
                logging.error(f"Exception occurred while processing {file}: {str(e)}")
                failed_files.append(file)

    if failed_files:
        logging.error(f"\nThe following files failed to translate: {failed_files}")
    else:
        logging.info("\nAll translations completed successfully!")
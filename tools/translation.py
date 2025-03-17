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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('translation.log')
    ]
)

def process_file(filename):
    try:
        input_file = os.path.join(directory, filename)
        logging.info(f'Processing {input_file}')
        if os.path.isfile(input_file):
            translate_cn(input_file)
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
    chunks = split_text(content, chunk_size=1000)  # Using smaller chunks for better reliability
    logging.info(f"Split {input_file} into {len(chunks)} chunks")

    # Create the output file name
    output_file = input_file.replace('.md', '_cn.md')
    
    # Create or clear the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("")
    
    for i, chunk in enumerate(chunks):
        max_retries = 1
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
                    file.write(response_text + "\n")
                    
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
    directory = "lectures"
    max_workers = 3  # Adjust this based on your API rate limits and system capabilities
    
    files = [f for f in os.listdir(directory) if f.endswith('.md') and os.path.isfile(os.path.join(directory, f))]
    logging.info(f'Files to translate: {files}')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                logging.info(f"Result for {file}: {result}")
            except Exception as e:
                logging.error(f"Exception occurred while processing {file}: {str(e)}")

    logging.info("\nAll translations completed!")
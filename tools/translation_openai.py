"""
Before running the file, config the environment by adding the OpenAI API key:

echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
echo $OPENAI_API_KEY # Test to see if it is added
"""

import requests
import os
import json
from concurrent.futures import ThreadPoolExecutor

def process_file(filename, function):
    input_file = os.path.join(directory, filename)
    print(f'processing {input_file}')
    if os.path.isfile(input_file):
        function(input_file)

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
    # Get the OpenAI API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # OpenAI API endpoint
    api_url = "https://api.openai.com/v1/chat/completions"
    
    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Read the content of the input markdown file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into chunks
    chunks = split_text(content, chunk_size=1000)  # Using smaller chunks for better reliability
    print(f"Split content into {len(chunks)} chunks")

    # Create the output file name
    output_file = input_file.replace('.md', '_cn.md')
    
    # Create or clear the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("")
    
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
        print(f"Chunk preview: {chunk[:100]}...")
        
        try:
            print(f"Sending request to OpenAI GPT-4o API...")
            
            # Prepare the payload for OpenAI API
            payload = {
                "model": "gpt-4o",  # Using GPT-4o model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the given text into simplified Chinese. Maintain all markdown syntax, code blocks, and directives exactly as they are. Only output the direct translation without any explanations or system messages."
                    },
                    {
                        "role": "user",
                        "content": chunk
                    }
                ],
                "temperature": 0,
                "max_tokens": 4000
            }
            
            # Make the API request
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Extract the response text
            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]
            
            print(f"Received response (length: {len(response_text)} chars)")
            print(f"Response preview: {response_text[:100]}...")
            
            # Append this chunk's translation to the output file immediately
            with open(output_file, 'a', encoding='utf-8') as file:
                file.write(response_text + "\n")
                
            print(f"Wrote chunk {i+1} translation to {output_file}")
                
        except Exception as e:
            print(f"Translation failed for chunk: {chunk[:50]}... Error: {str(e)}")
            print(f"Full error details: {e}")
            continue

    print(f"All chunks translated and saved to {output_file}")

if __name__ == "__main__":
    directory = "lectures"
    
    files = [f for f in os.listdir(directory) if f.endswith('.md') and os.path.isfile(os.path.join(directory, f))]
    print(f'files to translate: {files}')
    
    # Process all files, not just files[1:]
    for file in files:
        print(f"\nStarting translation of {file}...")
        process_file(file, translate_cn)
        print(f"Completed translation of {file}")

    print("\nAll translations completed!")
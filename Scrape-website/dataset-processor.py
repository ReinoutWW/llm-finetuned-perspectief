import os
import json
import openai
from typing import List, Dict, Any
from tqdm import tqdm
import time

class PerspectiefDatasetProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.output_file = "perspectief_dataset.jsonl"
        
    def load_files(self, directory: str) -> List[Dict[str, str]]:
        """Load all scraped files into memory"""
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract URL from first line
                        url = content.split('\n')[0].replace('URL: ', '').strip()
                        # Skip files with invalid URLs
                        if not url.startswith('http'):
                            print(f"Skipping invalid URL: {url}")
                            continue
                        
                        text = '\n'.join(content.split('\n')[1:])
                        documents.append({
                            'url': url,
                            'content': text
                        })
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    continue
        return documents

    def generate_qa_pairs(self, content: str, url: str) -> List[Dict[str, str]]:
        """Generate Q&A pairs using GPT-4"""
        print(f"Generating QA pairs for URL: {url}")
        print(f"Content length: {len(content)}")
        
        prompt = f"""You are a professional Q&A generator for workplace health and safety content.
        Given the following text, generate relevant Q&A pairs that accurately represent the content.
        Each pair should be factual and based on the provided information.
        
        URL:
        {url}

        Text:
        {content}
        
        Generate Q&A pairs in the following JSON format:
        "Q&A": [
            {{
                "Question": "Your question here",
                "Answer": "Your answer here. Source: [<page name> from Perspectief]({url})",
                "DetailedContext": "Additional context and details"
            }}
        ]
        
        Important:
        1. Only generate questions that can be answered accurately based on the text
        2. Include detailed context for each answer
        3. Generate 2-5 relevant Q&A pairs per document
        4. Make sure questions are clear and specific
        5. Answers should be concise but complete
        6. Always include the source URL in markdown format at the end of each answer: [<page name> from Perspectief]({url})
        
        Please ensure the output is valid JSON and follows the exact format specified above.
        Do not include any additional text or explanations outside of the JSON structure.
        """
        
        try:
            print("Sending request to OpenAI...")
            response = openai.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": "You are a professional Q&A generator for workplace health and safety content. Generate output in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            print("Received response from OpenAI")
            print(f"Raw response: {response}")
            
            # Get the raw JSON string from the response
            raw_json = response.choices[0].message.content
            print(f"Raw response content: {raw_json}")
            
            try:
                # Parse the JSON string
                qa_data = json.loads(raw_json)
                print(f"Parsed JSON: {qa_data}")
                
                # Handle different JSON structures
                if isinstance(qa_data, dict):
                    if "Questions" in qa_data:
                        qa_pairs = qa_data["Questions"]
                    elif "Q&A" in qa_data:
                        qa_pairs = qa_data["Q&A"]
                    else:
                        # Single QA pair
                        if all(key in qa_data for key in ['Question', 'Answer', 'DetailedContext']):
                            qa_pairs = [qa_data]
                        else:
                            print("Invalid JSON structure")
                            return []
                else:
                    print("Invalid JSON structure")
                    return []
                    
                # Validate each QA pair
                valid_pairs = []
                for qa in qa_pairs:
                    if (isinstance(qa, dict) and 
                        "Question" in qa and 
                        "Answer" in qa and 
                        "DetailedContext" in qa):
                        valid_pairs.append(qa)
                
                print(f"Extracted {len(valid_pairs)} valid QA pairs")
                return valid_pairs
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print(f"Raw content that failed to parse: {raw_json}")
                return []
            
        except Exception as e:
            print(f"Error generating Q&A pairs: {str(e)}")
            return []
            
    def process_documents(self, input_dir: str):
        """Process all documents and save to JSONL file"""
        documents = self.load_files(input_dir)
        
        try:
            # Ensure we can write to the file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                print(f"Opened file {self.output_file} for writing")
                
                for doc in tqdm(documents, desc="Processing documents"):
                    try:
                        # Skip if content is empty or invalid
                        if not doc.get('content'):
                            print(f"Skipping document with empty content: {doc.get('url', 'Unknown URL')}")
                            continue
                            
                        qa_pairs = self.generate_qa_pairs(doc['content'], doc['url'])
                        if not isinstance(qa_pairs, list):
                            print(f"Invalid QA pairs format for {doc['url']}")
                            continue
                            
                        if not qa_pairs:
                            print("No QA pairs generated")
                            continue
                
                        for qa in qa_pairs:
                            if not isinstance(qa, dict) or not all(key in qa for key in ['Question', 'Answer', 'DetailedContext']):
                                print(f"Skipping invalid QA pair: {qa}")
                                continue
                                
                            entry = {
                                "URL": doc['url'],
                                "Question": qa["Question"],
                                "Answer": qa["Answer"],
                                "DetailedContext": qa["DetailedContext"]
                            }
                            print(f"Writing entry to JSONL: {entry}")  # Debug log
                            
                            try:
                                # Convert entry to JSON string
                                entry_json = json.dumps(entry, ensure_ascii=False)
                                print(f"Entry JSON: {entry_json}")  # Debug log
                                
                                # Write to file
                                f.write(entry_json + '\n')
                                f.flush()  # Force write to disk
                                print("Successfully wrote entry to file")
                            except Exception as e:
                                print(f"Error writing to file: {str(e)}")
                        
                        # Add rate limiting to avoid hitting API limits
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error processing document {doc.get('url', 'Unknown URL')}: {str(e)}")
                        continue
                
                print(f"Finished writing to {self.output_file}")
                
        except Exception as e:
            print(f"Error opening or writing to file: {str(e)}")
                    
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = ""
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    processor = PerspectiefDatasetProcessor(api_key)
    processor.process_documents("scraped_content")

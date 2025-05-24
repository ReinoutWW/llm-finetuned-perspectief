import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin

class PerspectiefScraper:
    def __init__(self):
        self.base_url = "https://perspectief.eu"
        self.visited_urls = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_text_from_page(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text from the page
            text = soup.get_text()
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""

    def scrape_website(self, start_url):
        urls_to_visit = [start_url]
        
        while urls_to_visit:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            print(f"Scraping: {url}")
            self.visited_urls.add(url)
            
            try:
                response = self.session.get(url)
                response.raise_for_status()
                
                # Save the text content
                text = self.get_text_from_page(url)
                self.save_text(url, text)
                
                # Find all links on the page
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(self.base_url, href)
                    if full_url.startswith(self.base_url) and full_url not in self.visited_urls:
                        urls_to_visit.append(full_url)
                
                # Be polite - add delay between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")

    def save_text(self, url, text):
        # Create output directory if it doesn't exist
        output_dir = "scraped_content"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create filename from URL
        filename = url.replace("/", "_").replace(":", "_") + ".txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n\n")
            f.write(text)

if __name__ == "__main__":
    scraper = PerspectiefScraper()
    scraper.scrape_website("https://perspectief.eu")

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
import traceback

visited_urls = set()


def crawl_website(url, base_url):
    global visited_urls
    try:
        # Check if the URL has already been visited
        if url in visited_urls:
            return None

        parsed_url = urlparse(url)
        parsed_base_url = urlparse(base_url)
        if parsed_url.netloc != parsed_base_url.netloc:
           return None

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Add the URL to the set of visited URLs
            visited_urls.add(url)

            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text from the page
            extracted_text = soup.get_text()

            # Print the URL and extracted text
            print(f"URL: {url}")
            # Remove leading/trailing whitespaces
            print(extracted_text.strip())
            write_to_file(url, extracted_text.strip())

            # Find all internal links on the page and crawl them recursively
            for link in soup.find_all('a', href=True):
                if link is None:
                   continue 
                if link['href'].startswith('/'):
                    absolute_url = urljoin(url, link['href'])
                else:
                    absolute_url = link['href']
                crawl_website(absolute_url, base_url)

            return extracted_text
        else:
            print(
                f"Failed to crawl {url}. Status code: {response.status_code}")
            visited_urls.add(url)
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        visited_urls.add(url)
        return None


def url_to_unique_filename(url):
    # Parse the URL to get the filename
    if url.endswith('/'):
      url = url[:-1]
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # Generate a unique filename using a timestamp
    unique_filename = f"{parsed_url.netloc}_{filename}.txt"

    return unique_filename


def write_to_file(url, content):
    with open('./data_new/' + url_to_unique_filename(url), 'a') as out:
        out.write(content + '\n')


def main():
    print('Main thread started')

    try:
        startTime = time.time()
        print('Starting crawler')
        # crawler = crawlThreader()
        url = 'https://cloudjune.com'
        crawl_website(url, url)

        print('Run Time', time.time() - startTime)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()

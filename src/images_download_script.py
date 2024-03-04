import requests
from bs4 import BeautifulSoup
import os

def download_page(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Get the content of the page
            page_content = response.text
            
            # Save the content to a file or process it as needed
            with open('downloaded_page.html', 'w', encoding='utf-8') as file:
                file.write(page_content)
                
            print(f"Page content downloaded successfully to 'downloaded_page.html'")
            return page_content
        else:
            print(f"Failed to download page. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def find_img_with_class(html_content, class_name):
    try:
        # with open(file_path, 'r', encoding='utf-8') as file:
            # html_content = file.read()
        found_tags = []
        # Create a BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all <img> tags with the specified class recursively
        img_tags = soup.find_all('img', class_=class_name, recursive=True)
        print(len(img_tags))
        if img_tags:
            print(f"Found {len(img_tags)} <img> tag(s) with class '{class_name}':")
            for img_tag in img_tags:
                # print(img_tag)
                found_tags.append(img_tag)
        else:
            print(f"No <img> tag found with class '{class_name}'")
        return found_tags

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main_url = "https://www.indiaglitz.com/hindi-actor-varun-dhawan-photos-6918"
    actor = "VarunDhawan"
    os.makedirs(f"data/bollywood-stars/{actor}", exist_ok=True)
    
    page_content = download_page(main_url)
    tags = find_img_with_class(page_content, "galleryimage gallery-items")
    count = 0
    for tag in tags:
        file_url = tag.get("data-img")
        print(file_url)
        response = requests.get(file_url)
        format = file_url.split("/")[-1].split(".")[-1]
        with open(f"data/bollywood-stars/{actor}/image_{count}.{format}", 'wb') as file:
            file.write(response.content)
        count += 1

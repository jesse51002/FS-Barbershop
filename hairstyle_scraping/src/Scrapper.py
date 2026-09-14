import os
import shutil
from GoogleImageScraper.GoogleImageScraper import GoogleImageScraper
from GoogleImageScraper.patch import webdriver_executable
from Utils import getStop
import Constants


WEBDRIVER_PATH = os.path.normpath(os.path.join(os.getcwd(), "src", "GoogleImageScraper", 'webdriver', webdriver_executable()))

SCRAPE_COUNT = 2000
HIDE_BROWSER = False
MIN_RESOLUTION = (500,700)
MAX_RESOLUTION = (2000,2000)

HAIR_ADDS = [""] # ["", "front profile", "side profile"]

DEFAULT_QUERY = {
    "long bob hairstyle": ["short long bob hairstyle", "medium length long bob hairstyle", "long long bob hairstyle"]
}



def google_scrape(
    stop_func=None,query="long bob hairstyle", folder=Constants.RAW_IMAGES_DIR,
    img_download_callback=None, hide_browser=HIDE_BROWSER
    ):
    image_scraper = GoogleImageScraper(WEBDRIVER_PATH,"","",SCRAPE_COUNT,hide_browser,MIN_RESOLUTION,MAX_RESOLUTION)
    
    image_scraper.headless = hide_browser
    image_scraper.search_key = query
    image_scraper.image_path = folder
    image_scraper.img_download_callback = img_download_callback
    
    image_scraper.find_image_urls(stop_func)
    
def hair_scrape(style_dic, clean_queue=None, lock=None):
    # Creates file if it doesnt exist
    if not os.path.isfile(Constants.FINIHSED_RAW_TXT):
        file = open(Constants.FINIHSED_RAW_TXT, 'w')
        file.close()
        
    already_scrapped = set()
    with open(Constants.FINIHSED_RAW_TXT, 'r') as raw_finished_file:
        for x in raw_finished_file.readlines():
            line = x.strip()
            already_scrapped.add(line)
    
    def dwn_callback(img_pth):
        if clean_queue is None:
            return
        clean_queue.put(img_pth)
    
    for style in style_dic.keys():    
        for style_type in style_dic[style]:
            # Create folder if it doesnt exist
            output_folder = os.path.join(Constants.RAW_IMAGES_DIR, style, style_type)  
            
            if os.path.isdir(output_folder):
                shutil.rmtree(output_folder)
                
            os.makedirs(output_folder) 
            
            for adds in HAIR_ADDS:
                print("Scrapinng", style, style_type, adds)
                
                query = f"{style_type} {adds}".strip()
                # Scapes and downloads
                google_scrape(
                    stop_func=getStop, query=query, folder=output_folder,
                    hide_browser=HIDE_BROWSER, img_download_callback=dwn_callback
                    )

            if lock is None:
                continue 
            # Records that it finished scraping
            with lock:
                with open(Constants.FINIHSED_RAW_TXT, 'a') as raw_finished_file:
                    raw_finished_file.write(f'\n{style}/{style_type}')
                

if __name__ == "__main__":
    output = os.path.join(Constants.RAW_IMAGES_DIR, "test")
    if not os.path.isdir(output):
        os.makedirs(output)
    hair_scrape(style_dic=DEFAULT_QUERY)
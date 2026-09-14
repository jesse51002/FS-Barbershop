import sys
sys.path.insert(0,'./src')

from Utils import find_images

import Constants


if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 2:
        print("""
        Delete already uploaded folder from?
        1. Gan Dataset
        2. Hairstyle presets
        """)
        chosen = int(input())
        

        if chosen < 1 or chosen > 2:
            print(f"{chosen} is an invalid choice, pick a valid choice")
            
    if chosen == 1:
        
        accept_count = len(find_images(Constants.ACCEPTED_BODY_IMAGES_DIR))
        print("accepted body images:", accept_count)
        raw_count = len(find_images(Constants.RAW_BODY_IMAGES_DIR))
        print("raw body images:", raw_count)
        clean_count = len(find_images(Constants.CLEAN_BODY_IMAGES_DIR))
        print("clean body images:", clean_count)
    elif chosen == 2:
        accept_count = len(find_images(Constants.ACCEPTED_IMAGES_DIR))
        print("accepted body images:", accept_count)
        raw_count = len(find_images(Constants.RAW_IMAGES_DIR))
        print("raw body images:", raw_count)
        clean_count = len(find_images(Constants.CLEAN_IMAGES_DIR))
        print("clean body images:", clean_count)







import os
import time
from flickrapi import FlickrAPI
import numpy as np
import Constants
from Utils import get_flickr_creds



ATTRIBUTES = {
    "texture": ["straight", "wavy", "curly"],
    "length": ["short", "medium length", "long"],
    "fade": ["", "low fade", "mid fade", "high fade"],
    "half": ["", "half up" "half down"],
    "thick": ["thin", "regular", "thick"],
    "gender": ["men", "women"],
    "color": ["black", "brunnete", "blonde", "red", "blue", "orange", "green", "white", "purple", "yellow", "pink", "silver"],
    "color_dataset": ["afro", "curly", "wavy", "straight", "braids dreads", "men"]
}

POSTFIX_ATTR = ["gender", "half"]

EXTRA_TAGS = [
    "canon", "fujifilm", "nikon", "sony", "panasonic", "olympus", "pentax",
    "curly", "braids", "dreads", "wavy", "suit",  
    "photography", "guy", "woman", "lady", "fashion"
    ]


def get_queries():
    # Stores the results
    queries = {}
    # Gets the lines from the file
    lines = get_lines()
    
    # Counter
    combination_count = 0 
    
    for line in lines:
        line_split = line.split("|")
        
        style_name = line_split[0].strip()
        
        style_attributes = []
        if len(line_split) > 1:
            style_attributes = [s.strip() for s in line_split[1].split(",")]
        
        postfixs = []
        i = 0
        while i < len(style_attributes):
            if style_attributes[i] in POSTFIX_ATTR:
                postfixs.append(style_attributes.pop(i))
            else:
                i += 1
                
        order_arr = style_attributes + [style_name] + postfixs
        
        combinations = create_combinations("", order_arr)
        queries[style_name] = combinations
        combination_count += len(combinations)
        
        # print(style_name, combinations)
    
    print(f"Created {combination_count} different query combinations")
    
    return queries
    
# Returns all the possible combinations of the attributes for the styles
def create_combinations(cur_str, order_arr):
    if len(order_arr) == 0:
        return [cur_str.strip().lower()]
    
    cur_combs = []
    # Adds each possiblity to the tree
    if order_arr[0] in ATTRIBUTES:
        for attr in ATTRIBUTES[order_arr[0]]:
            cur_combs += create_combinations(cur_str + f"{attr} ", order_arr[1:]) 
    # Adds the raw string to the combination if its not a key in the dictionary
    else:
        cur_combs += create_combinations(cur_str + f"{order_arr[0]} ", order_arr[1:]) 
        
    return cur_combs
        

def get_lines():
    # Loads Styles lines from txt file
    # Ignoring comments
    styles = [] 
    with open(Constants.STYLES_FILES,'r') as file:
        for x in file.readlines():
            line = x.strip()
            if len(line) < 2 or line[:2] == "--" or line[0] == "#":
                continue
            styles.append(line)
    
    return styles
    
def create_body_queries():
    queries = [] 
    queries_extra = []
    with open(Constants.BODY_QUERIES_FILES,'r') as file:
        for x in file.readlines():
            line = x.strip()
            if len(line) < 2 or line[:2] == "--" or line[0] == "#":
                continue
            queries.append(line + " portrait")  
            
            query_base = queries[-1]
            for tag in EXTRA_TAGS:
                queries.append((query_base + " " + tag).strip())

    queries += queries_extra
    
    # Instanties then countries constant
    with open(Constants.COUNTRIES_FILES,'r') as file:
        for x in file.readlines():
            line = x.strip()
            queries.append(line + " portrait")  
        
    return queries      


def create_group_queries():
    group_urls = [
        ["ebony goodess", "https://www.flickr.com/groups/35144195@N00/"],
        ["black men", "https://www.flickr.com/groups/positive_black_men/"],
        ["black people", "https://www.flickr.com/groups/blackpeople/pool/with/53781885136"],
        ["black women", "https://www.flickr.com/groups/416360@N21/pool/"],
        ["brown skin beauty", "https://www.flickr.com/groups/858545@N23/"],
        ["indian girls", "https://www.flickr.com/groups/1603236@N25/pool/page3"],
        # ["Bangladesh Flickr" ,"https://www.flickr.com/groups/efty/pool/with/53464911647"]
        ["bbw", "https://www.flickr.com/groups/beautifulbigwomen/pool/"],
        ["men fashion", "https://www.flickr.com/groups/mensfashion/"],
        ["men on the street", "https://www.flickr.com/groups/menonthestreet/pool/with/53638413952"],
        ["trans_men", "https://www.flickr.com/groups/1872345@N25/pool/"],
        ["color portrait", "https://www.flickr.com/groups/colourstreetportraits/pool/"],
        ["free world", "https://www.flickr.com/groups/freeworld/pool/"],
        ["portraits", "https://www.flickr.com/groups/portrait/"],
        ["portraits unlimited", "https://www.flickr.com/groups/portraits-unlimited/pool/with/53898759349"]
    ]
    
    key, secret = get_flickr_creds()
    flickr = FlickrAPI(key, secret)
    
    for group in group_urls:
        print(group)
        flickr = FlickrAPI(key, secret, format='parsed-json')
        id = flickr.urls.lookupGroup(url=group[1])["group"]['id']
        group.append(id)
        group[0] = f"groups_{group[0]}"
        
    return group_urls

            
if __name__ == "__main__":
    get_queries()
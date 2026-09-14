import os
import json
from multiprocessing import Queue
import time
from threading import Thread

from PIL import Image
try:
    from PIL import ImageTk
    import tkinter
    from tkinter import filedialog, Tk
except ImportError:
    tkinter = None

from aws_s3_downloader import download_from_aws, get_download_folders
from Utils import get_file_path, delete_empty, find_images

os.system('xset r off')

ROOT = "./"

# Body files
CLEAN_BODY_IMAGES_DIR = os.path.join(ROOT, "clean_images")

ACCEPTED_BODY_IMAGES_DIR = os.path.join(ROOT, "accepted_images")

REJECTED_BODY_IMAGES_DIR = os.path.join(ROOT, "rejected_images")

ACCEPT_MODE = 0
EXPORT_MODE = 1
DOWNLOAD_MODE = 2


# Can either be hair or body
ACCEPT_IMAGE_SIZE = 750
CLOSED = False

root = Tk()
root.geometry(f"{ACCEPT_IMAGE_SIZE * 2}x{ACCEPT_IMAGE_SIZE + 200}") 

def close_program():
    global CLOSED
    CLOSED = True

root.protocol("WM_DELETE_WINDOW", close_program)
    
# Make the window jump above all
# root.attributes('-topmost',True)

# Style parser
# Scapes, cleans and shows cleaned image
def parse_style(accept_queue : list, root_clean_dir : str, root_accepted_dir : str, root_rejected_dir : str): 
    global CLOSED
    
    
    original_finished_count = len(find_images(root_accepted_dir)) + len(find_images(root_rejected_dir))
    absolute_total_count = original_finished_count + len(accept_queue)
    
    start_time = time.time()
    
    accept_count = 0
    reject_count = 0
    
    description_label = tkinter.Label(root, text="Starting", font=('Times 16'))
    description_label.place(x=ACCEPT_IMAGE_SIZE, y=ACCEPT_IMAGE_SIZE, anchor="n", width=ACCEPT_IMAGE_SIZE * 2, height=200)
    
    color_image_label = tkinter.Label(root)
    color_image_label.place(x=ACCEPT_IMAGE_SIZE, y=0, anchor="n", width=ACCEPT_IMAGE_SIZE, height=ACCEPT_IMAGE_SIZE)
    
    previous_stack = []
    resume_stack = []
    
    button_press_stack = []
    
    def key_press(e):
        button_press_stack.insert(0, e.char)
    
    root.bind("<KeyRelease>", key_press)
    
    while not CLOSED:
        root.update()
        
        img_pth = None
        if len(resume_stack) != 0:
            img_pth = resume_stack.pop(0)
        elif len(accept_queue) != 0:
            img_pth = accept_queue.pop(0)

        # If there is not image to accept currently
        if img_pth is None:
            break
    
        img_location = None
        if "clean_images" in img_pth:
            img_location  = root_clean_dir
        elif "accepted" in img_pth:
            img_location = root_accepted_dir
        else:
            img_location = root_rejected_dir

        
        pth_list = get_file_path(img_pth, img_location)
        
        # print("Displaying the image")
        
        # Create a photoimage object of the image in the path
        pil_image = Image.open(img_pth).resize((ACCEPT_IMAGE_SIZE, ACCEPT_IMAGE_SIZE),Image.LANCZOS)
        
        img = ImageTk.PhotoImage(pil_image)
        color_image_label.configure(image=img)
        color_image_label.image = img
         
        total_count = accept_count + reject_count
        if total_count > 0:
            accept_perc = (float(accept_count) / float(total_count)) * 100.0    
        else:
            accept_perc = 100.0
        
        time_took_min = (time.time() - start_time) / 60
            
        description_label.configure(
        text=f"""{pth_list} ||| 'A' to Accept ::: 'R' to reject ::: 'B' to back ::: 'H' to end |||  {total_count} images, {round(accept_perc, 2)}% in {round(time_took_min, 2)} minutes
        Total Progress:{original_finished_count + accept_count + reject_count} / {absolute_total_count}""")    
        
        root.update()
        
        # Gets the images save path
        accepted_dir = os.path.join(root_accepted_dir, pth_list)
        if not os.path.isdir(accepted_dir):
            os.makedirs(accepted_dir)
        accepted_pth = os.path.join(accepted_dir, os.path.basename(img_pth))
                    
        # Gets the images save path
        rejected_dir = os.path.join(root_rejected_dir, pth_list)
        if not os.path.isdir(rejected_dir):
            os.makedirs(rejected_dir)
        rejected_pth = os.path.join(rejected_dir, os.path.basename(img_pth))
        
                            
        while not CLOSED:
            key = None
            if len(button_press_stack) != 0:
                key = button_press_stack.pop(0)
            else:
                root.update()
                time.sleep(0.05)
                continue
            
            # Moves file to accepted
            if key == "A" or key == "a":
                if img_pth != accepted_pth:
                    # Write/ Overwrite accepted image
                    if os.path.isfile(accepted_pth):
                        os.remove(accepted_pth)
                    os.rename(img_pth, accepted_pth)
                    
                previous_stack.insert(0, accepted_pth)
                accept_count += 1
                print("Accepted a total of", accept_count)
                break             
            # If rejected delete the image
            elif key == "R" or key == "r":
                if img_pth != rejected_pth:
                    # Write/ Overwrite rejected image
                    if os.path.isfile(rejected_pth):
                        os.remove(rejected_pth)
                    os.rename(img_pth, rejected_pth)
                previous_stack.insert(0, rejected_pth)
                reject_count += 1
                print("Rejected a total of", reject_count)
                break
            # Go back to previous image
            elif key == "B" or key == "b":
                if len(previous_stack) == 0:
                    print("Nothing to go back to")
                    continue
                
                resume_stack.insert(0, img_pth)
                resume_stack.insert(0, previous_stack.pop(0))
                if "accept" in resume_stack[0]:
                    accept_count -= 1
                elif "reject" in resume_stack[0]:
                    accept_count -= 1
                break
            # Forcefully ends the application
            elif key == "H" or key == "h":
                CLOSED = True
                break
            else:     
                print("Invalid Key, Input valid key")
            
            # Clears key presses
            button_press_stack = []
        
        color_image_label.configure(image=None)
        color_image_label.image = None
            
        description_label.configure(text=f"Waiting...")    
        
        root.update()
    
    description_label.destroy()
    color_image_label.destroy()

    total_count = accept_count + reject_count
    if total_count > 0:
        accept_perc = (float(accept_count) / float(total_count)) * 100.0    
        time_took_min = (time.time() - start_time) / 60
        print(f"Went throught {total_count} images and accepted {round(accept_perc, 2)}% in {time_took_min} minutes")
    
    print("Ending accepting GUI")

def get_root():
    root_folder = os.path.join(os.getcwd(), "data")
    
    if not os.path.isdir(root_folder):
        os.makedirs(root_folder)
    
    return root_folder


def get_mode():
    mode_selected = -1
    
    def select_accept():
        nonlocal mode_selected
        mode_selected = ACCEPT_MODE
        
    def select_export():
        nonlocal mode_selected
        mode_selected = EXPORT_MODE
        
    label_button = tkinter.Button(root, text="Data Labeling", font=('Times 16'), command=select_accept)
    label_button.place(x=ACCEPT_IMAGE_SIZE - 450, y=int(ACCEPT_IMAGE_SIZE / 2), anchor="center", width=400, height=400)
    
    export_button = tkinter.Button(root, text="Export Data", font=('Times 16'), command=select_export)
    export_button.place(x=ACCEPT_IMAGE_SIZE + 450, y=int(ACCEPT_IMAGE_SIZE / 2), anchor="center", width=400, height=400)
    
    while mode_selected == -1:
        if CLOSED:
            break
        time.sleep(0.05)
        root.update()
        
    label_button.destroy()
    export_button.destroy()
    root.update()
    
    return mode_selected

# Starts the application
def launch():  
    global ROOT
    global CLEAN_BODY_IMAGES_DIR
    global ACCEPTED_BODY_IMAGES_DIR
    global REJECTED_BODY_IMAGES_DIR
    
    
    ROOT = get_root()

    # Body files
    CLEAN_BODY_IMAGES_DIR = os.path.join(ROOT, "clean_images")
    ACCEPTED_BODY_IMAGES_DIR = os.path.join(ROOT, "accepted_images")
    REJECTED_BODY_IMAGES_DIR = os.path.join(ROOT, "rejected_images")
        
    dirs = [
        CLEAN_BODY_IMAGES_DIR,
        ACCEPTED_BODY_IMAGES_DIR,
        REJECTED_BODY_IMAGES_DIR
        ]
    
    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    
    if CLOSED:
        return
    
    mode = get_mode()
    
    if CLOSED:
        return
    
    if mode == ACCEPT_MODE:
        description_label = tkinter.Label(root, text="Loading Images...", font=('Times 16'))
        description_label.place(x=ACCEPT_IMAGE_SIZE, y=int(ACCEPT_IMAGE_SIZE / 2), anchor="s", width=ACCEPT_IMAGE_SIZE * 2, height=200)
        root.update()
        
        accept_queue = []

        clean_images = find_images(CLEAN_BODY_IMAGES_DIR)
        
        def get_priority_numer(pth):
            for x in ["afro", "wavy"]:
                if x in pth:
                    return f"0-{pth}"
            
            return f"1-{pth}"
         
        clean_images = sorted(clean_images, key=lambda x: get_priority_numer(x))
        for img in clean_images:
            accept_queue.append(img)
        print(f"Instantiates accept queue with {len(clean_images)} clean images")
        
        description_label.destroy()
        
        parse_style(accept_queue, CLEAN_BODY_IMAGES_DIR, ACCEPTED_BODY_IMAGES_DIR, REJECTED_BODY_IMAGES_DIR)
        
        if len(accept_queue) == 0:
            description_label = tkinter.Label(root, text="FINISHED WORK ON ALL LOCAL DATA!!!!!", font=('Times 16'))
            description_label.place(x=ACCEPT_IMAGE_SIZE, y=int(ACCEPT_IMAGE_SIZE / 2), anchor="center", width=ACCEPT_IMAGE_SIZE * 2, height=200)
            
            for _ in range(50):
                time.sleep(0.1)
                root.update()
                
            description_label.destroy()
                
        # Deletes empty folders
        for dir in dirs:
            delete_empty(dir)
    elif mode == EXPORT_MODE:
        
        for dir in dirs:
            delete_empty(dir)
            
        for dir in dirs:
            if not os.path.isdir(dir):
                os.makedirs(dir)
        
        description_label = tkinter.Label(root, text="Select file save loaction", font=('Times 16'))
        description_label.place(x=ACCEPT_IMAGE_SIZE, y=int(ACCEPT_IMAGE_SIZE / 2), anchor="s", width=ACCEPT_IMAGE_SIZE * 2, height=200)
        
        confirmed = False
        
        def folder_select_pressed():
            nonlocal confirmed
            save_file_location = filedialog.asksaveasfilename(filetypes=[("json files" ,".json")])
            
            if os.path.isfile(save_file_location):
                os.remove(save_file_location)
            
            folders_started = set(os.listdir(ACCEPTED_BODY_IMAGES_DIR) + os.listdir(REJECTED_BODY_IMAGES_DIR))
            folders_uncomplete = set(os.listdir(CLEAN_BODY_IMAGES_DIR))
            folders_finished = folders_started - folders_uncomplete
            
            accepted_list = find_images(ACCEPTED_BODY_IMAGES_DIR)
            rejected_list = find_images(REJECTED_BODY_IMAGES_DIR)
            unfinished_list = find_images(CLEAN_BODY_IMAGES_DIR)
            
            accepted_percentage = 100 * len(accepted_list) / (len(accepted_list) + len(rejected_list))
            completion_percentage = 100 * (len(accepted_list) + len(rejected_list)) / (len(accepted_list) + len(rejected_list) + len(unfinished_list))
            
            
            export_dict = {
                "accepted": accepted_list,
                "rejected": rejected_list,
                "unfinished": unfinished_list,
                "finished_folders": list(folders_finished)
            }
            
            for key in export_dict.keys():
                for i in range(len(export_dict[key])):
                    img_pth = export_dict[key][i]
                    rel_pth = "/".join(img_pth.replace("\\", "/").split("/")[-2:])
                    export_dict[key][i] = rel_pth
            
            export_dict["!worker_data"] = {
                    "Total accepted": len(accepted_list),
                    "Total rejected": len(rejected_list),
                    "Images Left": len(unfinished_list),
                    "Accepting Perecentage": f"{accepted_percentage}%",
                    "Completion Percentage": f"{completion_percentage}%"
                }
            
            with open(save_file_location, "w") as f: 
                json.dump(export_dict, f, indent=2, sort_keys=True)
            
            # current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # s3resource.upload_file(save_file_location, BUCKET_NAME, "outsourced_results/" + current_datetime + ".json")
            
            confirmed = True   
               
        select_button = tkinter.Button(root, text="Select save location", font=('Times 16'), command=folder_select_pressed)
        select_button.place(x=ACCEPT_IMAGE_SIZE, y=int(ACCEPT_IMAGE_SIZE / 2), anchor="n", width=200, height=100)
        
        while not confirmed:
            if CLOSED:
                break
            time.sleep(0.05)
            root.update()
            
        description_label.destroy()
        select_button.destroy()
        root.update()
    
    root.destroy()
    
    print("ENDING...")
    
if __name__ == "__main__":
    launch()
    

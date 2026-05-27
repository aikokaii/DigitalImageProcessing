import os
import tkinter as tk
from PIL import Image, ImageTk

folder = "dataset"  

files = [f for f in os.listdir(folder) if f.endswith(".pgm")]
files.sort()

index = 0

root = tk.Tk()
root.title("BioID Image Viewer")

label = tk.Label(root)
label.pack()

def show_image():
    global index
    
    path = os.path.join(folder, files[index])
    
    img = Image.open(path)
    img = img.resize((384,286))   # optional resize
    
    tk_img = ImageTk.PhotoImage(img)
    
    label.config(image=tk_img)
    label.image = tk_img
    
    title.config(text=files[index])

def next_image():
    global index
    index = (index + 1) % len(files)
    show_image()

def prev_image():
    global index
    index = (index - 1) % len(files)
    show_image()

title = tk.Label(root,text="")
title.pack()

btn_frame = tk.Frame(root)
btn_frame.pack()

prev_btn = tk.Button(btn_frame,text="Previous",command=prev_image)
prev_btn.pack(side=tk.LEFT)

next_btn = tk.Button(btn_frame,text="Next",command=next_image)
next_btn.pack(side=tk.LEFT)

show_image()

root.mainloop()

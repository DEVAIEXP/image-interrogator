from tkinter import filedialog, Tk
import sys
import os
import easygui

ENV_EXCLUSION = ['COLAB_GPU', 'RUNPOD_POD_ID']

def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)

def get_folder_path(folder_path=''):
    if (
        not any(var in os.environ for var in ENV_EXCLUSION)
        and sys.platform != 'darwin'
    ):
        current_folder_path = folder_path

        initial_dir, initial_file = get_dir_and_file(folder_path)

        root = Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        folder_path = filedialog.askdirectory(initialdir=initial_dir)
        root.destroy()

        if folder_path == '':
            folder_path = current_folder_path

    return folder_path

def open_folder():

    choosen_path=easygui.diropenbox("Image folder:")
    return choosen_path
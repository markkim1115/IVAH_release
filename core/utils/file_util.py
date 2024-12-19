import os
import subprocess
import shutil
import yaml

def find_a_file(path:str, keyword:str):
    """
    Find a file in the given directory and return the path.
    If the file is not found, return the first file path that contains the given string.
    """

    if os.path.exists(os.path.join(path, keyword)):
        return os.path.join(path, keyword)
    else:
        file_list = sorted(os.listdir(path))

        for file in file_list:
            if keyword in file:
                return os.path.join(path, file)

def list_files(folder_path, exts=None, keyword=None):
    file_list = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, fname))
                    and (exts is None or any(fname.endswith(ext) for ext in exts))
                    and (keyword is None or (fname.find(keyword)!=-1))
        ]
    file_list = sorted(file_list)

    return file_list

import re
def custom_sorting_key(s):
    num = re.findall('\d+', s)
    if num:
        return (0, int(num[0]))  # If string contains a number, sort it before character-only strings
    else:
        return (1, s)  # If string contains only characters, sort it after strings with numbers and sort alphabetically among themselves
    
def custom_sort(l:list):
    return sorted(l, key=custom_sorting_key)

def split_path(file_path):
    file_dir, file_name = os.path.split(file_path)
    file_base_name, file_ext = os.path.splitext(file_name)
    return file_dir, file_base_name, file_ext

def create_video(img_path, out_path, fps, verbose=False):
    '''
    Creates a video from the frame format in the given directory and saves to out_path.
    '''
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    command = ['ffmpeg', '-y', '-r', str(fps), '-i', img_path, \
                    '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        command += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(command)

def create_gif(img_path, out_path, fps, verbose=False):
    '''
    Creates a gif (and video) from the frame format in the given directory and saves to out_path.
    '''
    
    vid_path = out_path[:-3] + 'mp4'
    create_video(img_path, vid_path, fps)
    command = ['ffmpeg', '-y', '-i', vid_path, \
                    '-pix_fmt', 'rgb8', out_path]
    if not verbose:
        command += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(command)


def to_video_file(img_path, gif=True, remove_frames=False):
    file_path = img_path

    file_name = '_'.join(file_path.split('/')[1:])+'.mp4'
    flist = list_files(file_path)
    fdir, filebase, ext = split_path(flist[0])
    if '_' in flist[0]:
        len_frame_num = len(filebase)
    else:
        len_frame_num = len(filebase)
    if gif:
        if not 'gif' in file_name:
            file_name = file_name[:-3] + 'gif'
        create_gif(img_path=img_path+'/'+'%0'+f'{len_frame_num}'+'d'+f'{ext}', out_path=os.path.join(os.path.dirname(file_path), 'video', file_name), fps=30) 
    else:
        if not 'mp4' in file_name:
            file_name = file_name[:-3] + 'mp4'
        create_video(img_path=img_path+'/'+'%0'+f'{len_frame_num}'+'d'+f'{ext}', out_path=os.path.join(file_path, 'video', file_name), fps=30)
    
    if remove_frames:
        shutil.rmtree(img_path)

def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)
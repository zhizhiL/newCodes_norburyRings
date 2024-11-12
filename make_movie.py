import cv2
import os
import shutil
import glob


def create_movie_from_frames(target, frame_folder, output_path, fps=3):
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.png') and f.startswith(target)]
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  

    frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs like 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        video_writer.write(frame)

    video_writer.release()

def move_to_subfoler(parent_dir, target, N_realisation):
    file_pattern = os.path.join(parent_dir, target + '_*.npy')
    file_list = glob.glob(file_pattern)

    child_dir = os.path.join(parent_dir+'/..', 'N_realisation_' + str(N_realisation) + model)
    os.makedirs(child_dir, exist_ok=True)

    for file in file_list:
        file_name = os.path.basename(file)
        shutil.move(file, os.path.join(child_dir, file_name))
    
    print('Moved files to ' + child_dir)



target = 'frame'
model = '_bounce'
N_realisation = 38

frame_folder = 'influx_random_sims_3D_duo/temp' 
output_path = frame_folder + '/output_' + target +'_realisation_' + str(N_realisation)+ model + '.mp4'
# create_movie_from_frames(target, frame_folder + '/temp', output_path)
create_movie_from_frames(target, frame_folder, output_path)
move_to_subfoler('influx_random_sims_3D_duo/temp', target, N_realisation)
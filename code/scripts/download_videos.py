import json
import os
def download(label, name, path):
    label = label.replace(" ", "_")
    
    path_data = os.path.join(path, label)
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    link_prefix = "https://www.youtube.com/watch?v="
    filename = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print("already exists, skip")
        return

    print("download the whole video for: [%s] - [%s]" % (label, name))
    command1 = 'youtube-dl '
    command1 += link + " "
    command1 += "-o " + filename + " "
    command1 += "-f best "
    command1 += '-q '  # print no log
    os.system(command1)
    print ('finish the video as: ' + filename)
    return


music_dat = 'iQuery/data/json/MUSIC_solo_videos.json'
video_pth = '../MUSIC21_dataset/videos/solo'
with open(music_dat, "r") as read_file:
    data = json.load(read_file)

for music in data['videos']:
    v = data['videos'][music]
    for vid_name in v:
        download(music, vid_name, video_pth)

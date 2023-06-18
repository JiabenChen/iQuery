import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

video_pth = '../MIT-Music/solo'

sound_list = os.listdir(video_pth)
save_pth = '../audio'


for sound in sound_list:
    audio_pth = os.path.join(video_pth, sound)
    lis = os.listdir(audio_pth)
    if not os.path.exists(os.path.join(save_pth, sound)):
        os.makedirs(os.path.join(save_pth, sound))
    exist_lis = os.listdir(os.path.join(save_pth, sound))
    for audio_id in lis:
        name = os.path.join(video_pth, sound, audio_id)
        video = VideoFileClip(name)
        audio = video.audio
        audio_name = audio_id[:-4] + '.wav'
        if audio_name in exist_lis:
            print("already exist!")
            continue
        audio.write_audiofile(os.path.join(save_pth, sound, audio_name), fps=11000)
        print("finish video id: " + audio_name)



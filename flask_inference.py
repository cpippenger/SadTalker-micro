from glob import glob
import threading
import torch
from time import  strftime
import os, sys
import tempfile
from flask import Flask, jsonify, request, render_template, send_file,redirect
from uuid import uuid4 
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path



app = Flask(__name__,static_folder="cache")

# this is the same as the arguments here 
# https://github.com/OpenTalker/SadTalker/blob/cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8/inference.py#L100
class SadTalker_Settings:
    #driven_audio='./examples/driven_audio/bus_chinese.wav'
    #source_image='./examples/source_image/full_body_1.png'
    ref_eyeblink=None
    ref_pose=None
    checkpoint_dir='./checkpoints'
    result_dir='./cache'
    pose_style=0
    batch_size=2
    size=256
    expression_scale=1.0
    input_yaw=None
    input_pitch=None
    input_roll=None
    enhancer=None
    background_enhancer=None
    cpu=False
    face3dvis=False
    still=False
    preprocess='crop'
    verbose=False
    old_version=False
    net_recon='resnet50'
    init_path=None
    use_last_fc=False
    bfm_folder='./checkpoints/BFM_Fitting/'
    bfm_model='BFM_model_front.mat'
    focal=1015.0
    center=112.0
    camera_d=10.0
    z_near=5.0
    z_far=15.0
    device='cuda'

settings=SadTalker_Settings()
current_root_path = os.path.split(sys.argv[0])[0]
sadtalker_paths = init_path(settings.checkpoint_dir, os.path.join(current_root_path, 'src/config'), settings.size, settings.old_version, settings.preprocess)
preprocess_model = CropAndExtract(sadtalker_paths, settings.device)
audio_to_coeff = Audio2Coeff(sadtalker_paths,  settings.device)    
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, settings.device)

# need to handle file objects in args
def sadtalker_main(str_wavfile,str_imgpath,settings=SadTalker_Settings()):
    runid=str(uuid4())
    #torch.backends.cudnn.enabled = False
    pic_path = str_imgpath
    audio_path = str_wavfile
    save_dir = '/tmp/sadtalker_run_' + runid
    os.makedirs(save_dir, exist_ok=True)
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, settings.preprocess,\
                                                                             source_image_flag=True, pic_size=settings.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if settings.ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, settings.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if settings.ref_pose is not None:
        if settings.ref_pose == settings.ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, settings.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, settings.device, ref_eyeblink_coeff_path, still=settings.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, settings.pose_style, ref_pose_coeff_path)

    # 3dface render
    if settings.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                settings.batch_size, settings.input_yaw, settings.input_pitch, settings.input_roll,
                                expression_scale=settings.expression_scale, still_mode=settings.still, preprocess=settings.preprocess, size=settings.size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=settings.enhancer, background_enhancer=settings.background_enhancer, preprocess=settings.preprocess, img_size=settings.size)
    outfile_name=  runid + '.mp4'
    outfile_path=settings.result_dir + '/' + outfile_name
    os.rename(result, outfile_path)
    print('The generated video is named:', outfile_path)
    
    #if not settings.verbose:
    #    shutil.rmtree(save_dir)
    return outfile_path



@app.get("/")
async def root():
    return '''<html>
<head>
    <script
      src="https://code.jquery.com/jquery-3.7.1.min.js"
      integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo="
      crossorigin="anonymous"></script> 

</head>
   <body>
        <form method=post action="/upload"  enctype="multipart/form-data">
            face image: <input name="face_file" type="file" required />   </br>
            wav file: <input name="wav_file" type="file" required />   </br>
            <input type = "submit" value="Upload">  
        </form>

    </body>
</html>'''

def temp_save(obj_file,str_filename):
    obj_file.save(str_filename)


@app.route('/upload', methods = ['POST'])
async def run_sadtalker(): 
    if request.files['face_file'].filename == '' :
        return "No Face File"
    if request.files['wav_file'].filename == '' : 
        return "No Wav File"
    # adding multi-threaded here to compensite for large file uploads
    # may remove later
    face_file=tempfile.NamedTemporaryFile().name
    wav_file=tempfile.NamedTemporaryFile().name
    threads = []
    threads.append(threading.Thread(target=temp_save, args=(request.files['wav_file'],wav_file)))
    threads.append(threading.Thread(target=temp_save, args=(request.files['face_file'],face_file)))
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    print(wav_file,face_file)
    final_file=sadtalker_main(wav_file,face_file);
    
    return send_file(final_file,os.path.basename(final_file))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7666)
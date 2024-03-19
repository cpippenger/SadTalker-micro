import numpy as np
import cv2, os, sys, torch
from tqdm import tqdm
from PIL import Image 
import magic
# 3dmm extraction
import safetensors
import safetensors.torch 
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from src.face3d.models import networks

from scipy.io import loadmat, savemat
from src.utils.croper import Preprocesser


import warnings

from src.utils.safetensor_helper import load_x_from_safetensor 
warnings.filterwarnings("ignore")

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


class CropAndExtract():

    # need this other places, broke it out to a function (untested)
    def __video_loader(input_path):    
        # loader for videos
        video_stream = cv2.VideoCapture(input_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = [] 
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break 
            full_frames.append(frame) 
            if source_image_flag:
                break
        return full_frames

    def __init__(self, sadtalker_path, device):

        self.propress = Preprocesser(device)
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        
        if sadtalker_path['use_safetensor']:
            checkpoint = safetensors.torch.load_file(sadtalker_path['checkpoint'])    
            self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        else:
            checkpoint = torch.load(sadtalker_path['path_of_net_recon_model'], map_location=torch.device(device))    
            self.net_recon.load_state_dict(checkpoint['net_recon'])

        self.net_recon.eval()
        self.lm3d_std = load_lm3d(sadtalker_path['dir_of_BFM_fitting'])
        self.device = device
    
    # return_filepaths=True will return string paths to files, otherwise return data or fileobjects
    # this will save files to disk
    def generate(self, input_path, save_dir, crop_or_resize='crop',
                source_image_flag=False, pic_size=256,return_filepaths=True):
        landmarks_path =  os.path.join(save_dir, 'landmarks.txt') 
        coeff_path =  os.path.join(save_dir, 'coeff.mat')  
        png_path =  os.path.join(save_dir, 'face.png')  
        print('!!!',input_path)
        if isinstance(input_path,str): # this is mostly as it was in the original
            if not os.path.isfile(input_path):
                raise ValueError('input_path must be a valid path to video/image file')
            elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
                # loader for first frame
                full_frames = [cv2.imread(input_path)]
                fps = 25
            else:
                print('!! processing video file...')
                full_frames = [] 
                full_frames =__video_loader(input_path)
                
            # not sure if this is needd for videos, seems to run eithe rway, i think  you
            # can pass cv2.COLOR_BGR2RGB to imread to skip this, if picture ends up blue
            # its probably something to do with this
            x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames] 

        else:
            #https://github.com/ahupp/python-magic/blob/master/magic/__init__.py#L220C5-L220C20
            print('!! file object recived')
            # figure out file type from file buffer, support above types
            magic_number=magic.from_buffer(input_path.read(16)) # 16 bytes seems to work for images
            file_type=magic_number.split(' ')[0] 
            print('!!!magic=',magic_number)
            input_path.seek(0) # reset file for next read
            if file_type in ['JPEG','PNG']:
                x_full_frames = [cv2.imdecode(np.asarray(bytearray(input_path.read()), dtype=np.uint8),cv2.IMREAD_UNCHANGED )]
            elif file_type in ['MPEG']: # this is unsupported
                raise ValueError('Loading Videos from File Object Not Supported yet')
            else:
                raise ValueError('File must be a valid PNG or JPEG')
            

        #### crop images as the 
        if 'crop' in crop_or_resize.lower(): # default crop
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else: # resize mode
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] 
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # 2. get the landmark according to the detected face. 
        if not os.path.isfile(landmarks_path): 
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch 
            video_coeffs, full_coeffs = [],  []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):
                frame = frames_pil[idx]
                W,H = frame.size
                lm1 = lm[idx].reshape([-1, 2])
            
                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2]+1)/2.
                    lm1 = np.concatenate(
                        [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
                    )
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)
 
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
                
                with torch.no_grad():
                    full_coeff = self.net_recon(im_t)
                    coeffs = split_coeff(full_coeff)

                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
 
                pred_coeff = np.concatenate([
                    pred_coeff['exp'], 
                    pred_coeff['angle'],
                    pred_coeff['trans'],
                    trans_params[2:][None],
                    ], 1)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.cpu().numpy())

            semantic_npy = np.array(video_coeffs)[:,0] 

            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})
        if return_filepaths == True:
            return coeff_path, png_path, crop_info
        else:
            return {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]}, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR), crop_info 

"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
import os
import sys
import pickle
import PIL


# project root directory
project_root = os.path.dirname(os.path.abspath(__file__)) 
# Add data directory to the Python path
data_directory = os.path.join('E:/', 'PIE_dataset')


def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength*progress))
    text = "\r[{}] {:0.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

    
######### DATA UTILITIES ##############


def img_pad(img, mode = 'warp', size = 224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size,size),PIL.Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or  \
            (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size-img_size [0])//2,
                    (size-img_size [1])//2))
        return padded_image


def squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    # width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
    bbox[0] = bbox[0] - width_change/2
    bbox[2] = bbox[2] + width_change/2
    # bbox[1] = str(float(bbox[1]) - width_change/2)
    # bbox[3] = str(float(bbox[3]) + width_change)
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0
    
    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0]-bbox[2] + img_width
        bbox[2] = img_width
    return bbox

def bbox_sanity_check(img, bbox):
    '''
    This is to confirm that the bounding boxes are within image boundaries.
    If this is not the case, modifications is applied.
    This is to deal with inconsistencies in the annotation tools
    '''
    img_width, img_heigth = img.size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox


def jitter_bbox(img_path,bbox, mode, ratio):
    '''
    This method jitters the position or dimentions of the bounding box.
    mode: 'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly sampling a value in [-ratio,ratio)
    ratio: The ratio of change relative to the size of the bounding box. For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Note: Tha ratio of change in pixels is calculated according to the smaller dimension of the bounding box.
    '''
    assert(mode in ['same','enlarge','move','random_enlarge','random_move']), \
            'mode %s is invalid.' % mode

    if mode == 'same':
        return bbox

    img = load_img(img_path)
    img_width, img_heigth = img.size

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio  = abs(ratio)
    else:
        jitter_ratio  = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample()*jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ['enlarge','random_enlarge']:
            b[0] = b[0] - width_change //2
            b[1] = b[1] - height_change //2
        else:
            b[0] = b[0] + width_change //2
            b[1] = b[1] + height_change //2

        b[2] = b[2] + width_change //2
        b[3] = b[3] + height_change //2

        # Checks to make sure the bbox is not exiting the image boundaries
        b =  bbox_sanity_check(img, b)
        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes

###############################################


class PIEIntent(object):
    """
    A convLSTM encoder decoder model for predicting pedestrian intention
    Attributes:
        _num_hidden_units: Number of LSTM hidden units
        _reg_value: the value of L2 regularizer for training
        _kernel_regularizer: Training regularizer set as L2
        _recurrent_regularizer: Training regularizer set as L2
        _activation: LSTM activations
        _lstm_dropout: input dropout
        _lstm_recurrent_dropout: recurrent dropout
        _convlstm_num_filters: number of filters in convLSTM
        _convlstm_kernel_size: kernel size in convLSTM

    Model attributes: set during training depending on the data
        _encoder_input_size: size of the encoder input
        _decoder_input_size: size of the encoder_output

    Methods:
        load_images_and_process: generates trajectories by sampling from pedestrian sequences
        get_data_slices: generate tracks for training/testing
        create_lstm_model: a helper function for creating conv LSTM unit
        pie_convlstm_encdec: generates intention prediction model
        train: trains the model
        test_chunk: tests the model (chunks the test cases for memory efficiency)
    """
    def __init__(self):

        pass

    def get_path(self,
                 type_save='models', # model or data
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder=os.path.join(project_root, 'data')):
        """
        A path generator method for saving model and config data. Creates directories
        as needed.
        :param type_save: Specifies whether data or model is saved.
        :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
        :param model_name: model name (either trained convlstm_encdec model or vgg16)
        :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
        :param data_subset: train, test or val
        :param data_type: type of the data (e.g. features_context_pad_resize)
        :param save_root_folder: The root folder for saved data.
        :return: The full path for the save folder
        """
        assert(type_save in ['models', 'data'])
        if data_type != '':
            assert(any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def load_images_and_process(self,
                                img_sequences,
                                bbox_sequences,
                                ped_ids,
                                num_samples,
                                save_path,
                                # data_type='train',
                                regen_pkl=False):
        """
        Generates image features for convLSTM input. The images are first
        cropped to 1.5x the size of the bounding box, padded and resized to
        (224, 224) and fed into pretrained VGG16.
        :param img_sequences: a list of frame names
        :param bbox_sequences: a list of corresponding bounding boxes
        :ped_ids: a list of pedestrian ids associated with the sequences
        :save_path: path to save the precomputed features
        :data_type: train/val/test data set
        :regen_pkl: if set to True overwrites previously saved features
        :return: a list of image features
        """
        
        if num_samples>0:
            img_sequences = img_sequences[:num_samples]
            ped_ids = ped_ids[:num_samples]

        sequences = []
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_sequences[i], pid):
                set_id = imp.split('\\')[-3]
                vid_id = imp.split('\\')[-2]
                img_name = imp.split('\\')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
                
                if os.path.exists(img_save_path) and not regen_pkl:
                    with open(img_save_path, 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                else:
                    img_data = load_img(imp)
                    bbox = jitter_bbox(imp, [b],'enlarge', 2)[0]
                    bbox = squarify(bbox, 1, img_data.size[0])
                    bbox = list(map(int,bbox[0:4]))
                    cropped_image = img_data.crop(bbox)
                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)                        
                    image_array = img_to_array(img_data)
                    img_features = np.expand_dims(image_array, axis=0)
                    img_features = np.permute_dims(img_features, (0, 3, 1, 2))
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)
                img_features = np.squeeze(img_features)
                img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        return sequences
    

    def get_tracks(self, dataset, data_type, seq_length, overlap):
        """
        Generate tracks by sampling from pedestrian sequences
        :param dataset: raw data from the dataset
        :param data_type: types of data for encoder/decoder input
        :param seq_length: the length of the sequence
        :param overlap: defines the overlap between consecutive sequences (between 0 and 1)
        :return: a dictionary containing sampled tracks for each data modality
        """
        overlap_stride = seq_length if overlap == 0 else \
        int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        d_types = []
        for k in data_type.keys():
            d_types.extend(data_type[k])
        d = {}

        if 'bbox' in d_types:
            d['bbox'] = dataset['bbox']
        if 'intention_binary' in d_types:
            d['intention_binary'] = dataset['intention_binary']
        if 'intention_prob' in d_types:
            d['intention_prob'] = dataset['intention_prob']

        bboxes = dataset['bbox'].copy()
        images = dataset['image'].copy()
        ped_ids = dataset['ped_id'].copy()

        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i+seq_length] for i in\
                             range(0,len(track)\
                            - seq_length + 1, overlap_stride)])
            d[k] = tracks

        pid = []
        for p in ped_ids:
            pid.extend([p[i:i+seq_length] for i in\
                         range(0,len(p)\
                        - seq_length + 1, overlap_stride)])
        ped_ids = pid

        im = []
        for img in images:
            im.extend([img[i:i+seq_length] for i in\
                         range(0,len(img)\
                        - seq_length + 1, overlap_stride)])
        images = im

        bb = []
        for bbox in bboxes:
            bb.extend([bbox[i:i+seq_length] for i in\
                         range(0,len(bbox)\
                        - seq_length + 1, overlap_stride)])

        bboxes = bb
        return d, images, bboxes, ped_ids

    def concat_data(self, data, data_type):
        """
        Concatenates different types of data specified by data_type.
        Creats dummy data if no data type is specified
        :param data_type: type of data (e.g. bbox)
        """
        if not data_type:
            return []
        # if more than one data type is specified, they are concatenated
        d = []
        for dt in data_type:
            d.append(np.array(data[dt]))
        if len(d) > 1:
            d = np.concatenate(d, axis=2)
        else:
            d = d[0]
        return d

    def get_train_val_data(self, data, data_type, seq_length, overlap):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param data_type: The data types defined for encoder and decoder
        :return: A unified data representation as a list.
        """
        tracks, images, bboxes, ped_ids = self.get_tracks(data, data_type, seq_length, overlap)

        # Generate observation data input to encoder
        encoder_input = self.concat_data(tracks, data_type['encoder_input_type'])
        decoder_input = self.concat_data(tracks, data_type['decoder_input_type'])
        output = self.concat_data(tracks, data_type['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)

        return {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'output': output}

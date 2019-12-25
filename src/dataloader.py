import torch.utils.data as data
from PIL import Image
import os
import os.path
import random


def _make_dataset(dir):
    """
    Creates a 2D list of all the frames in N clips containing
    M frames each.

    2D List Structure:
    [[frame00, frame01,...frameM]  <-- clip0
     [frame00, frame01,...frameM]  <-- clip1
     :
     [frame00, frame01,...frameM]] <-- clipN

    Parameters
    ----------
        dir : string
            root directory containing clips.

    Returns
    -------
        list
            2D list described above.
    """


    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))
    return framesPath

def _make_video_dataset(dir):
    """
    Creates a 1D list of all the frames.

    1D List Structure:
    [frame0, frame1,...frameN]

    Parameters
    ----------
        dir : string
            root directory containing frames.

    Returns
    -------
        list
            1D list described above.
    """


    framesPath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
        # Add path to list.
        framesPath.append(os.path.join(dir, image))
    return framesPath

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens image at `path` using pil and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, optional
            coordinates for cropping image. Default: None
        resizeDim : tuple, optional
            dimensions for resizing image. Default: None
        frameFlip : int, optional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            2D list described above.
    """


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')


class SeDraw_vimeo90k(data.Dataset):

    def __init__(self, root, transform=None, dim=(448, 256), randomCropSize=(208, 208), train=True, test=False):


        # Populate the list with image paths for all the
        # frame in `root`.
        train_f = open(root + '/tri_trainlist.txt', 'r')
        test_f = open(root + '/tri_testlist.txt', 'r')
        self.train_path = train_f.read().split('\n')
        self.train_path = [item for item in self.train_path if item != '']
        test_val_path = test_f.read().split('\n')
        self.test_path = [item for index, item in enumerate(test_val_path) if item != '']
        self.validation_path = [item for index, item in enumerate(test_val_path) if index % 100 == 0 and item != '']

        self.randomCropSize = randomCropSize
        self.cropX0 = dim[0] - randomCropSize[0]
        self.cropY0 = dim[1] - randomCropSize[1]
        self.root = root
        self.transform = transform
        self.train = train
        self.test = test

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - I0 and I1 -
        and a random frame chosen from the 7 intermediate frames
        available between I0 and I1 along with it's relative index.

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            tuple
                (sample, returnIndex) where sample is
                [I0, intermediate_frame, I1] and returnIndex is
                the position of `random_intermediate_frame`.
                e.g.- `returnIndex` of frame next to I0 would be 0 and
                frame before I1 would be 6.
        """

        sample = []

        if (self.train):
            ### Data Augmentation ###

            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            if (random.randint(0, 1)):
                frameRange = [0, 1, 2]
            else:
                frameRange = [2, 1, 0]
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            frameRange = [0, 1, 2]
            randomFrameFlip = 0

        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            if self.train:
                image = _pil_loader(
                    self.root + '/sequences/' + self.train_path[index] + '/im{}.png'.format(frameIndex + 1),
                    cropArea=cropArea, frameFlip=randomFrameFlip)
            elif self.test:
                image = _pil_loader(
                    self.root + '/sequences/' + self.test_path[index] + '/im{}.png'.format(frameIndex + 1),
                    cropArea=cropArea, frameFlip=randomFrameFlip)
            else:
                image = _pil_loader(
                    self.root + '/sequences/' + self.validation_path[index] + '/im{}.png'.format(frameIndex + 1),
                    cropArea=cropArea, frameFlip=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)

        return sample

    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """
        if self.train:
            return len(self.train_path)
        elif self.test:
            return len(self.test_path)
        else:
            return len(self.validation_path)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class UCI101Test(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:

        |-- clip0
            |-- frame00
            |-- frame01
            |-- frame02
        |-- clip1
            |-- frame00
            |-- frame01
            |-- frame02
        :
        :
        |-- clipN
            |-- frame00
            |-- frame01
            |-- frame02

    ...

    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.

    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """

    def __init__(self, root, transform=None):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            transform : callable, optional
                A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
        """

        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.framesPath = framesPath
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - I0 and I1 -
        and a intermediate frame between I0 and I1.

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            tuple
                (sample, returnIndex) where sample is
                [I0, intermediate_frame, I1] and returnIndex is
                the position of `intermediate_frame`.
                The returnIndex is always 3 and is being returned
                to maintain compatibility with the `SeDraw`
                dataloader where 3 corresponds to the middle frame.
        """

        sample = []
        # Loop over for all frames corresponding to the `index`.
        for framePath in self.framesPath[index]:
            # Open image using pil.
            image = _pil_loader(framePath)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample, 3

    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """

        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SeDraw(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:

        |-- clip0
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
        |-- clip1
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
        :
        :
        |-- clipN
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12

    ...

    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.

    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """


    def __init__(self, root, transform=None, dim=(448, 256), randomCropSize=(208, 208), train=True):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            transform : callable, optional
                A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            dim : tuple, optional
                Dimensions of images in dataset. Default: (640, 360)
            randomCropSize : tuple, optional
                Dimensions of random crop to be applied. Default: (352, 352)
            train : boolean, optional
                Specifies if the dataset is for training or testing/validation.
                `True` returns samples with data augmentation like random 
                flipping, random cropping, etc. while `False` returns the
                samples without randomization. Default: True
        """


        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train

        self.framesPath     = framesPath

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - I0 and I1 -
        and a random frame chosen from the 7 intermediate frames
        available between I0 and I1 along with it's relative index.

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            tuple
                (sample, returnIndex) where sample is 
                [I0, intermediate_frame, I1] and returnIndex is 
                the position of `random_intermediate_frame`. 
                e.g.- `returnIndex` of frame next to I0 would be 0 and
                frame before I1 would be 6.
        """


        sample = []
        
        if (self.train):
            ### Data Augmentation ###
            # To select random 9 frames from 12 frames in a clip
            firstFrame = 0
            # Apply random crop on the 9 input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            #frameRange = range(firstFrame, firstFrame + 9) if (random.randint(0, 1)) else range(firstFrame + 8, firstFrame - 1, -1)
            # IFrameIndex = random.randint(firstFrame + 1, firstFrame + 7)
            IFrameIndex = 1
            if (random.randint(0, 1)):
                frameRange = [firstFrame, IFrameIndex, firstFrame + 2]
                returnIndex = 1
            else:
                frameRange = [firstFrame + 2, IFrameIndex, firstFrame]
                returnIndex = 1
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = 1
            returnIndex = 1
            frameRange = [0, IFrameIndex, 2]
            randomFrameFlip = 0
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            
        return sample, returnIndex


    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """


        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

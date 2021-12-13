from torch.utils import data
from torch import from_numpy
import torchvision.transforms as transforms
import os, platform
from pathlib import Path
from PIL import Image
import numpy as np
import cv2 as cv
import json


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        raise ValueError('No images in the data folder')
        return None
    else:
        return allfiles


class BSDS_Loader(data.Dataset):
    """
    for BRIND
    """

    def __init__(self, root='data/HED-BSDS', split='train', transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class BSDS_VOCLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train',
                 transform=False, threshold=0.3, ablation=False, arg=None,
                 mean_bgr=[103.939, 116.779, 123.68]):
        self.root = root
        self.split = split
        self.mean_bgr = mean_bgr
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        # normalize = transforms.Normalize(mean=[103.939, 116.779, 123.68])
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        if self.split == 'train':
            print('error here train')
            if ablation:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            self.root = os.path.join(self.root,arg.test_data[0])
            if ablation:
                print('error here train')
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                filelist = os.path.join(self.root, arg.test_list)
        else:
            raise ValueError("Invalid split type!")

        self.filelist =[]
        with open(filelist, 'r') as f:
            files = f.readlines()
        files = [line.strip() for line in files]
        pairs = [line.split() for line in files]
        for pair in pairs:
            tmp_img = pair[0]
            self.filelist.append(
                (os.path.join(self.root, tmp_img)))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1
        else:
            img_file = self.filelist[index].rstrip()

        # with open(os.path.join(self.root, img_file), 'rb') as f:
        #     img = Image.open(f)
        #     # img = img.convert('RGB') # ORI
        #     img = img.convert('RGB')
        img =cv.imread(os.path.join(self.root, img_file))
        img = np.array(img, dtype=np.float32)
        img -=self.mean_bgr
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class MDBD_Loader(data.Dataset):
    """
    Dataloader for Multicue
    """

    def __init__(self, root='data/', split='train', transform=False, threshold=0.3, setting=['boundary', '1']):
        """
        setting[0] should be 'boundary' or 'edge'
        setting[1] should be '1' or '2' or '3'
        """
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                self.root, 'train_pair_%s_set_%s.lst' % (setting[0], setting[1]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                self.root, 'test_%s_set_%s.lst' % (setting[0], setting[1]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class NYUD_Loader(data.Dataset):
    """
    Dataloader for NYUDv2
    """

    def __init__(self, root='data/', split='train', transform=False, threshold=0.4, setting=['image']):
        """
        There is no threshold for NYUDv2 since it is singlely annotated
        setting should be 'image' or 'hha'
        """
        self.root = root
        self.split = split
        self.threshold = 128
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                self.root, '%s-train_da.lst' % (setting[0]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                self.root, '%s-test.lst' % (setting[0]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        scale = 1.0
        if self.split == "train":
            img_file, lb_file, scale = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            scale = float(scale.strip())
            pil_image = Image.open(os.path.join(self.root, lb_file))
            if scale < 0.99:  # which means it < 1.0
                W = int(scale * pil_image.width)
                H = int(scale * pil_image.height)
                pil_image = pil_image.resize((W, H))
            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            if scale < 0.9:
                img = img.resize((W, H))
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class Custom_Loader(data.Dataset):
    """
    Custom Dataloader
    """

    def __init__(self, root='data/'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.imgList = fold_files(os.path.join(root))

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.imgList[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        filename = Path(self.imgList[index]).stem

        return img, filename


class BipedDataset(data.Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 train_data='BIPED', arg=None
                 ):
        self.data_root = os.path.join(data_root,arg.train_data[0])
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg
        self.train_data = train_data
        self.train_list = arg.train_list

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        data_root = os.path.join(data_root, 'edges')
        if self.train_data.lower() == 'biped':

            images_path = os.path.join(data_root,
                                       'imgs',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)
            labels_path = os.path.join(data_root,
                                       'edge_maps',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)

            for directory_name in os.listdir(images_path):
                image_directories = os.path.join(images_path, directory_name)
                for file_name_ext in os.listdir(image_directories):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (os.path.join(images_path, directory_name, file_name + '.jpg'),
                         os.path.join(labels_path, directory_name, file_name + '.png'),)
                    )
        else:
            file_path = os.path.join(data_root, self.train_list)
            if self.train_data.lower() == 'brind':

                with open(file_path, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return image, label

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.  # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        # data = []
        # if self.scale is not None:
        #     for scl in self.scale:
        #         img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
        #         data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
        #     return data, gt
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else 352  # MDBD=480 BPED=352

        # for BSDS
        if i_w> crop_size and i_h>crop_size:
            i = np.random.randint(0, i_h - crop_size)
            j = np.random.randint(0, i_w - crop_size)
            img = img[i:i + crop_size , j:j + crop_size ]
            gt = gt[i:i + crop_size , j:j + crop_size ]

        # # for BIPED
        # if np.random.random() >= 0.5:  # l
        #     h, w = gt.shape
        #     LR_img_size = 256  # l BIPED=256, 240 200 # MDBD= 352
        #     i = np.random.randint(0, h - LR_img_size)
        #     j = np.random.randint(0, w - LR_img_size)
        #     # if img.
        #     img = img[i:i + LR_img_size, j:j + LR_img_size]
        #     gt = gt[i:i + LR_img_size, j:j + LR_img_size]
        #     img = cv.resize(img, dsize=(crop_size, crop_size), )
        #     gt = cv.resize(gt, dsize=(crop_size, crop_size))
        else:
            # New addidings
            img = cv.resize(img, dsize=(crop_size, crop_size))
            gt = cv.resize(gt, dsize=(crop_size, crop_size))
        # for  BIPED and BRIND
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        # for MDBD
        # gt[gt > 0.1] +=0.3
        ## gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = from_numpy(img.copy()).float()
        gt = from_numpy(np.array([gt])).float()
        return img, gt


class TestDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):

        self.data_root = os.path.join(data_root,test_data)
        self.test_data = test_data
        self.test_list = arg.test_list
        self.args = arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # image and label paths are located in a list file

            if not self.test_list:
                raise ValueError(
                    f"Test list not provided for dataset: {self.test_data}")
            # just for biped test dataset
            # if self.test_data=='BIPED':
            #     list_name = os.path.join(self.data_root, 'edges', self.test_list)
            # else:
            list_name = os.path.join(self.data_root, self.test_list)
            if self.test_data=='BIPED':

                with open(list_name) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
            else:
                with open(list_name, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        image_path = self.data_index[idx][0]
        label_path = None if self.test_data[0] == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # # base dir
        # if self.test_data[0] == 'CLASSIC':
        #     img_dir = self.data_root
        #     gt_dir = None
        # else:
        #     img_dir = self.data_root
        #     gt_dir = self.data_root

        # load data
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if not self.test_data[0] == "CLASSIC":
            label = cv.imread(label_path, cv.IMREAD_COLOR)
        else:
            label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        # return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)
        return image, file_name

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data[0] == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv.resize(img, (img_width, img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv.resize(img, (self.img_width, self.img_height))  # 512
            gt = cv.resize(gt, (self.img_width, self.img_height))  # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv.resize(img, (img_width, img_height))
            gt = cv.resize(gt, (img_width, img_height))
        else:
            img_width = self.img_width
            img_height = self.img_height
            img = cv.resize(img, (img_width, img_height))
            gt = cv.resize(gt, (img_width, img_height))

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = from_numpy(np.array([gt])).float()

        return img, gt
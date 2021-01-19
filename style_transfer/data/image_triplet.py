import cv2
import numpy as np
import torch
import os
from typing import Optional, Tuple, Union


class ImageTriplet(torch.utils.data.Dataset):
    '''
    Loads, holds and saves images for Neural Style Transfer

    Args:
        content_image_path: path to the file containing content image.
        style_image_path: path to the file containing style image.
        resize_size (optional): the size to which the images will be resized before performing neural style transfer 
            or string "content" to use content image's size.
        save_path (optional): the path to where save the resulting image, if not provided saves in result folder.
        save_resized (optional): if true will save the resized image in addition to original size.
    '''

    content_image_path: str
    style_image_path: str
    save_path: str
    save_path_original_size: str
    save_resized: bool
    content_image: Optional[torch.Tensor]
    style_image: Optional[torch.Tensor]
    result_image: Optional[torch.Tensor]
    original_size: Tuple[int, int]
    resize_size: Union[Tuple[int, int], str]


    def __init__(
        self,
        content_image_path: str, 
        style_image_path: str,
        resize_size: Union[Tuple[int, int], str] = (512, 512),
        save_path: Optional[str] = None,
        save_resized: bool = False,
        ) -> None:

        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.resize_size = resize_size
        self.save_resized = save_resized
        if save_path is not None:
            self.save_path = save_path
        else:
            result_name = \
                os.path.basename(self.content_image_path).split('.')[0] + \
                "_as_" + \
                os.path.basename(self.style_image_path).split('.')[0]
            self._save_path_without_extension = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', result_name)
            self.save_path = self._save_path_without_extension + '_resized.jpg'
            self.save_path_original_size = self._save_path_without_extension + '.jpg'

        self.content_image = None
        self.style_image = None
        self.result_image = None
        self._load_images()

    def _load_image(self, image_path: str) -> np.ndarray:
        return cv2.imread(image_path)

    def _save_image(self, save_path: str, image: np.ndarray) -> bool:
        return cv2.imwrite(save_path, image)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        '''
        Preprocesses the image from write-ready to model-ready.

        Args:
            image: a numpy array containing the image to preprocess.
        '''

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resize_size)
        image = image.astype(dtype=np.float32)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image /= 255.0
        
        return image.unsqueeze(0)
    
    def _deprocess_image(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Deprocesses the image from model-ready to write-ready.

        Args:
            image: a tensor containing the image to deprocess.
        '''

        image = image.squeeze(0)
        image *= 255.0
        image = image.permute(1, 2, 0)
        image = image.detach().numpy()
        image = image.astype(dtype=np.uint8)
        image = np.clip(image, 0, 255)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, cv2.resize(image, self.original_size)

    def save_result(self, is_intermediate=False, epoch_n=None, verbose=True) -> None:
        '''
        Deprocesses the result_image and saves it at save_path (+save_path_original_size).

        Args:
            is_intermediate (optional): save the result_image as intermediate given current epoch_n.
            epoch_n (optional): current epoch, must be provided if is_intermediate is True.
            verbose (optional): if True, will print logs
        '''

        if is_intermediate:
            if epoch_n is None:
                return
            result, _ = self._deprocess_image(self.result_image.detach().clone().to('cpu'))
            path = self._save_path_without_extension + '{}.jpg'.format(epoch_n + 1)
            res = self._save_image(path, result)
            if verbose:
                print("Saved intermediate result at \n\t{}, success={}".format(path, res))
        elif self.result_image is not None:
            result, result_original_size = self._deprocess_image(self.result_image.detach().clone())
            res1 = self._save_image(self.save_path_original_size, result_original_size)
            if self.save_resized:
                res2 = self._save_image(self.save_path, result)
                if verbose:
                    print("Saved results at \n\t{}, success={}\n\t{}, success={}".format(self.save_path, res1, self.save_path_original_size, res2))
            else:
                print("Saved result at \n\t{}, success={}".format(self.save_path_original_size, res1))

    def _load_images(self) -> None:
        '''
        Loads content and style images and creates a base for the result image
        '''

        if self.content_image is None:
            content_image = self._load_image(self.content_image_path)
            self.original_size = content_image.shape[:2][::-1]
            if self.resize_size == "content":
                self.resize_size = self.original_size
            self.content_image = self._preprocess_image(content_image)

            self.result_image = self.content_image.clone()

        if self.style_image is None:
            self.style_image = self._preprocess_image(self._load_image(self.style_image_path))


    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, idx) -> torch.Tensor:
        return self.result_image.squeeze(0)
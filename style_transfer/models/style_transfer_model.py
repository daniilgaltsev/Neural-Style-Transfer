import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from ..data.image_triplet import ImageTriplet
from .neural_style_losses import ContentLoss, StyleLoss, TotalVariationLoss
from collections import defaultdict
from typing import List


class NeuralStyleTransferModel(pl.LightningModule):
    """
    A module to perform a neural style transfer between images.

    Args:
        image_triplet: a data structure containing images for transfer as well as a place to store the result.
        content_weight (optional): a coefficient for the content loss.
        style_weight (optional): a coefficient for the style loss.
        total_variation_weight (optional): a coefficient for total variation loss
        verbose (optional): a boolean determining if the object should print anything during training
        save_intermediate (optional): if true the model will save intermediate results
    """

    image_triplet: ImageTriplet
    content_weight: float
    style_weight: float
    total_variation_weight: float
    verbose: bool
    save_intermediate: bool
    model: nn.Module
    content_losses: List[ContentLoss]
    style_losses: List[StyleLoss]
    total_variation_loss: TotalVariationLoss

    DEFAULT_STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    DEFAULT_CONTENT_LAYERS = ["conv4_2"]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self, 
        image_triplet: ImageTriplet,
        content_weight: float = 0.1,
        style_weight: float = 100000.0,
        total_variation_weight: float = 1.0,
        verbose: bool = False,
        save_intermediate: bool = False
    ) -> None:        
        super().__init__()

        self.image_triplet = image_triplet
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        

        self.content_losses = []
        self.style_losses = []
        self.total_variation_loss = TotalVariationLoss()
        self.model = nn.Sequential(
            torchvision.transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
            self.total_variation_loss
        )

    def _prepare_model(self) -> None:
        '''
        Constructs the model by inserting the loss layers and precomputing features from style and content images
        '''

        backbone = torchvision.models.vgg19(pretrained=True).to(self.device)

        layer_count = defaultdict(int)
        layer_count["block"] = 1
        name_type = ""
        last_loss_layer_index = 0
        idx = len(self.model)
        content_image = self.image_triplet.content_image.to(self.device)
        style_image = self.image_triplet.style_image.to(self.device)

        for layer in backbone.features.children():
            if isinstance(layer, nn.Conv2d):
                name_type = "conv"
            elif isinstance(layer, nn.ReLU):
                name_type = "relu"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name_type = "pool"
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                raise RuntimeError('Not supported layer: {}'.format(layer.__class__.__name__))

            layer_count[name_type] += 1
            name = name_type + "{}_{}".format(layer_count["block"], layer_count[name_type])
            if name_type == "pool":
                layer_count["block"] += 1
                layer_count["conv"] = 0
                layer_count["relu"] = 0

            self.model.add_module(name, layer)

            if name in self.DEFAULT_CONTENT_LAYERS:
                content = self.model(content_image)
                content_loss = ContentLoss(content)
                layer_count["content_loss"] += 1
                self.model.add_module("content_loss_{}".format(layer_count["content_loss"]), content_loss)
                self.content_losses.append(content_loss)
                idx += 1
                last_loss_layer_index = idx

            if name in self.DEFAULT_STYLE_LAYERS:
                style = self.model(style_image)
                style_loss = StyleLoss(style)
                layer_count["style_loss"] += 1
                self.model.add_module("style_loss_{}".format(layer_count["style_loss"]), style_loss)
                self.style_losses.append(style_loss)
                idx += 1
                last_loss_layer_index = idx
            
            idx += 1

        self.model = self.model[:last_loss_layer_index + 1].eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False


    def forward(self, x):
        return self.model(x)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.image_triplet, batch_size=1, pin_memory=True)  

    def training_step(self, batch, batch_idx):
        self.model(batch)

        style_loss = 0.0
        for sl in self.style_losses:
            style_loss += sl.loss
        style_loss *= self.style_weight

        content_loss = 0.0
        for cl in self.content_losses:
            content_loss += cl.loss
        content_loss *= self.content_weight

        total_variation_loss = self.total_variation_loss.loss * self.total_variation_weight
        
        loss = content_loss + style_loss + total_variation_loss

        self.log('content_loss', content_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('style_loss', style_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('tv_loss', total_variation_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        self.image_triplet.result_image.requires_grad = True
        return torch.optim.Adam(
            [
                {'params': [self.image_triplet.result_image]}
            ], 
            lr=0.05
        )

    def on_fit_start(self) -> None:
        self._prepare_model()

    def on_train_epoch_end(self, outputs) -> None:
        self.image_triplet.result_image.data.clamp_(0, 1)
        if self.save_intermediate and ((self.current_epoch + 1) % 20 == 0):
            self.image_triplet.save_result(True, self.current_epoch, self.verbose)




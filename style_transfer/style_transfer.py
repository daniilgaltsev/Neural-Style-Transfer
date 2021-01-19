import argparse
import pytorch_lightning as pl
import torch
import style_transfer.data.image_triplet
import style_transfer.models.style_transfer_model
from typing import Any


def _parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        'content_image',
        type=str,
        help="A path to the content image",
    )
    parser.add_argument(
        'style_image',
        type=str,
        help="A path to the style image",
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=False,
        help="A path where to save the combined image"
    )

    parser.add_argument(
        '--gpu', 
        default=False, 
        dest='use_gpu', 
        action='store_true', 
        help='If true, then will use a gpu for style transfer if available'
    )

    parser.add_argument(
        '--number_of_steps',
        default=40,
        type=int,
        required=False,
        help="Number of steps to perform (recommened 10/20/40/100 depending on the desired influence of style and running time)"
    )

    parser.add_argument(
        '--resize_dim',
        default=512,
        type=int,
        required=False,
        help="The size at which the transfer will happen (use 0 for content image's original size)."
    )

    parser.add_argument(
        '--content_weight',
        default=0.1,
        type=float,
        required=False,
        help="The weight of content loss in the result"
    )
    parser.add_argument(
        '--style_weight',
        default=100000.0,
        type=float,
        required=False,
        help="The weight of style loss in the result"
    )
    parser.add_argument(
        '--tv_weight',
        default=1.0,
        type=float,
        required=False,
        help="The weight of total variation loss in the result"
    )


    args = parser.parse_args()
    if args.resize_dim == 0:
        args.resize_dim = "content"
    else:
        args.resize_dim = (args.resize_dim, args.resize_dim)

    return args


def perform_style_transfer(args: Any) -> None:
    
    content_path = args.content_image
    style_path = args.style_image

    print("Content image path: {}".format(content_path))
    print("Style image path: {}".format(style_path))
    print("Using resize dimension of {}".format(args.resize_dim))

    image_triplet = style_transfer.data.image_triplet.ImageTriplet(content_path, style_path, resize_size=args.resize_dim)

    gpus = 0
    if args.use_gpu:
        if torch.cuda.is_available():
            gpus = 1
            print("Using gpu")
        else:
            print("Flag --gpu is set, but gpu is not available (no gpu or no torch_cuda")

    print("Will use weights: style={:.2f}, content={:.4f}, tv={:.2f}".format(
        args.style_weight, args.content_weight, args.tv_weight
    ))

    model = style_transfer.models.style_transfer_model.NeuralStyleTransferModel(
        image_triplet, 
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        total_variation_weight=args.tv_weight,
        save_intermediate=False, 
        verbose=False
    )

    print("Will perform {} steps".format(args.number_of_steps))

    trainer = pl.Trainer(
        max_steps=args.number_of_steps, 
        gpus=gpus, log_every_n_steps=1, 
        default_root_dir=None, 
        checkpoint_callback=False, 
        logger=False
    )
    trainer.fit(model)

    image_triplet.save_result()
    

if __name__ == "__main__":
    args = _parse_args()
    perform_style_transfer(args)

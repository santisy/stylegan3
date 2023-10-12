import argparse
import inspect
import logging
import os
from datetime import timedelta
from pathlib import Path
import numpy as np
from typing import Optional
import dnnlib
import legacy

import accelerate
import cv2
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from diffusions.decode import decode_nc

import diffusers
from diffusers import UNet2DModel
from diffusions.ddpm_scheduler_custom import DDPMSchedulerCustom as DDPMScheduler
from diffusions.ddpm_pipeline import DDPMPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--encoder_decoder_network", type=str,
        help="The encoder decoder network pkl path string.", required=True
    )
    parser.add_argument(
        "--feat_spatial_size",
        type=int,
        default=64,
        help="Feature spatial size"
    ) 
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--exported_root",
        type=str,
        default="exported",
        help="The exported image root dir."
        )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument(
        "--total_gen_nk",
        type=int,
        default=50,
        help="Total generate image number."
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # My settings
    parser.add_argument(
        "--init_dim", type=int, default=256, help="Initial dimension for the diffusion UNet."
    )
    parser.add_argument(
        "--dim_mults", type=lambda x: [int(y) for y in x.split(',')],
        help='The channel multiplication of the network.',
        default='1,2,3,4,4'
    )
    parser.add_argument(
        "--seed_start", type=int, default=0,
        help="Start seed value")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    seed_start = args.seed_start
    exported_dir = os.path.join(args.exported_root,
                                (os.path.basename(args.output_dir) + "_" + 
                                 str(args.total_gen_nk) + 'k' +
                                 f"_seed{seed_start}"))
    os.makedirs(exported_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # The encoder decoder one
    with dnnlib.util.open_url(args.encoder_decoder_network) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        G = G.eval()
        G = G.to(accelerator.device)

    # Initialize the model
    block_out_channels = [args.init_dim * mult_ for mult_ in args.dim_mults]
    down_block_types = ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"]
    up_block_types = ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"]
    # Try to use the default one
    model = UNet2DModel(
        sample_size=args.feat_spatial_size,
        in_channels=G.feat_coord_dim,
        out_channels=G.feat_coord_dim,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=2,
    )

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)


    # Prepare everything with our `accelerator`.
    model= accelerator.prepare(model)

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_step = args.total_gen_nk * 1000 // args.batch_size
    logger.info("***** Running Inference *****")
    logger.info(f"Total generate image number: {args.total_gen_nk}k")
    logger.info(f"Inference batch size: {args.batch_size}")
    logger.info(f"Save image at folder (exported_dir): {exported_dir}")


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

    # Infer !
    progress_bar = tqdm(total=total_step, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Inference step {total_step}")
    model.eval()
    unet = accelerator.unwrap_model(model)
    if args.use_ema:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

    for i in range(total_step):

        # Save images
        if accelerator.is_main_process:

            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=noise_scheduler,
            )

            generator = torch.Generator(device=pipeline.device).manual_seed(i + seed_start)
            # run pipeline in inference (sample random noise and denoise)
            sample_ni = pipeline(
                generator=generator,
                batch_size=args.batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
            )
            sample_ni = torch.clip((sample_ni + 1.0) / 2.0, 0, 1)

            with torch.no_grad():
                sample_imgs: np.ndarray = decode_nc(G, sample_ni).cpu().numpy()
            
            sample_imgs = (sample_imgs + 1) / 2.0
            sample_imgs = np.clip(sample_imgs, 0, 1) * 255.0
            sample_imgs = sample_imgs.astype(np.uint8)
            sample_imgs = sample_imgs.transpose(0, 2, 3, 1)

            for j, img_ in enumerate(sample_imgs):
                img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                idx = j + i * args.batch_size
                img_path = os.path.join(exported_dir, f"{idx:06d}.png")
                cv2.imwrite(img_path, img_)

        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
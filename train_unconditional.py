import argparse
import inspect
import logging
import math
import os
import shutil
from pathlib import Path
import numpy as np
from typing import Optional
import dnnlib
import legacy

import accelerate
import torch
import torch.nn.functional as F
import torch.distributed as tdist
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from training.dataset import ImageFolderDataset as Dataset
from diffusions.decode import decode_nc
from training.training_loop import save_image_grid

import diffusers
from diffusions.ddpm_scheduler_custom import DDPMSchedulerCustom as DDPMScheduler
from diffusers import UNet2DModel
from diffusions.ddpm_pipeline import DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from utils.utils import copy_back_fn


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
        "--train_data",
        type=str,
        required=True,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
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
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_steps_k", type=int, default=4, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
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
        "--class_condition", action="store_true"
    )
    parser.add_argument(
        "--condition_scale", type=float, default=0.3,
        help="Classifier free guidance scale."
    )
    parser.add_argument(
        "--p_uncond", type=float, default=0.2,
        help="Probability of unconditional training."
    )
    parser.add_argument(
        "--work_on_tmp_dir", action="store_true"
    )
    parser.add_argument(
        "--min_snr_gamma", type=float, default=-1,
    )
    parser.add_argument(
        "--no_noise_perturb", type=bool, default=True
    )
    parser.add_argument(
        "--copy_back", type=bool, default=False
    )

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
    copy_back_flag = args.work_on_tmp_dir and args.copy_back

    if args.work_on_tmp_dir:
        tmp_dir = os.getenv("SLURM_TMPDIR")
        output_dir = os.path.join(tmp_dir, args.output_dir)
        new_data_root = os.path.join(tmp_dir, "datasets")
        os.makedirs(new_data_root, exist_ok=True)
        dataset_path = os.path.join(new_data_root, os.path.basename(args.train_data))
        local_dir = args.output_dir
        os.makedirs(local_dir, exist_ok=True)
        local_eval_sample_dir = os.path.join(local_dir, 'eval_sample')
        os.makedirs(local_eval_sample_dir, exist_ok=True)
    else:
        output_dir = args.output_dir
        dataset_path = args.train_data

    logging_dir = os.path.join(output_dir, args.logging_dir)
    eval_sample_dir = os.path.join(output_dir, 'eval_sample')
    os.makedirs(eval_sample_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Different seed for different rank (local rank)
    rank_now = tdist.get_rank() if tdist.is_initialized() else 0
    np.random.seed(rank_now)
    torch.manual_seed(rank_now)

    if accelerator.is_local_main_process and args.work_on_tmp_dir and not os.path.exists(dataset_path):
        print(f"\033[92mCopying dataset {args.train_data} to {tmp_dir} ...\033[00m")
        os.system(f"cp {args.train_data} {new_data_root}") 
        print("\033[92mFinished copying.\033[00m")
    if tdist.is_initialized():
        tdist.barrier()

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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
                repo_name = get_full_repo_name(Path(output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # The encoder decoder one
    with dnnlib.util.open_url(args.encoder_decoder_network) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
        G = G.eval()
        G = G.to(accelerator.device)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = Dataset(dataset_path,
                      imagenet_flag=args.class_condition)

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
        num_class_embeds=dataset.class_dim + 1 if args.class_condition else None,
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

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).


    logger.info(f"Dataset size: {len(dataset)}")

    def collate_fn(batch_tuple):
        img = torch.tensor(torch.stack([torch.tensor(x[0]) for x in batch_tuple], dim=0))
        img = img.float() / 127.5 - 1.0
        if not args.class_condition:
            label = None
        else:
            # Shift the label to +1, Zero is the unconditional label
            label = torch.cat([torch.tensor(x[1]).reshape(1) for x in batch_tuple]).long() + 1
            # Randomly drop label to zero (null)
            # Cond drop ratio is 0.1
            drop_idx = torch.where(torch.rand(size=label.shape) < args.p_uncond)[0]
            label[drop_idx] = 0
        return img, label

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # NOTE: Try to use the local file to resume
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

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            image_input = batch[0].to(accelerator.device)
            if args.class_condition:
                labels = batch[1].to(accelerator.device)
            else:
                labels = None
            with torch.no_grad():
                # `Clean feat coords`
                encode_fn = getattr(G, "encode",
                                    lambda x, **kwargs: (G.img_encoder(x),))
                try:
                    clean_images = encode_fn(image_input,
                                              no_noise_perturb=args.no_noise_perturb
                                              )[0].detach()
                except:
                    clean_images = encode_fn(image_input)[0].detach()
                clean_images = clean_images * 2.0 - 1.0

            # Sample noise that we'll add to the images
            noise = torch.randn(
                clean_images.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
            ).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps, class_labels=labels).sample

                # Get snr
                alpha_t = _extract_into_tensor(
                    noise_scheduler.alphas_cumprod.to(timesteps.device),
                    timesteps,
                    (clean_images.shape[0], 1, 1, 1)
                )
                snr = alpha_t / (1 - alpha_t)

                if args.prediction_type == "epsilon":
                    if args.min_snr_gamma > 0:
                        snr_weights = snr.clamp_max(args.min_snr_gamma) / snr
                        snr_weights = snr_weights.detach()
                    else:
                        snr_weights = 1.0
                    loss = F.mse_loss(model_output, noise,
                                      reduction="none") * snr_weights  # this could have different weights!
                    loss = loss.mean()
                elif args.prediction_type == "sample":
                    snr_weights = snr
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if copy_back_flag: 
                        copy_back_fn(save_path, local_dir, isdir=True)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Save images
            if accelerator.is_main_process:
                if (global_step + 1) % (args.save_images_steps_k * 1000) == 0:
                    unet = accelerator.unwrap_model(model)

                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())

                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler
                    )

                    generator = torch.Generator(device=pipeline.device).manual_seed(global_step//1000)
                    # run pipeline in inference (sample random noise and denoise)
                    sample_ni = pipeline(
                        generator=generator,
                        batch_size=args.eval_batch_size,
                        num_inference_steps=args.ddpm_num_inference_steps,
                        class_condition=args.class_condition,
                        condition_scale=args.condition_scale,
                    )
                    sample_ni = torch.clip((sample_ni + 1.0) / 2.0, 0, 1)

                    with torch.no_grad():
                        sample_imgs = decode_nc(G, sample_ni).cpu().numpy()
                    # Save image to local target folder
                    save_image_path = os.path.join(eval_sample_dir,
                                                   f'fakes{(global_step + 1) // 1000:06d}.png')
                    save_image_grid(sample_imgs,
                                    save_image_path,
                                    drange=[-1,1],
                                    grid_size=(int(np.sqrt(args.eval_batch_size)),
                                            int(np.sqrt(args.eval_batch_size))))
                    if copy_back_flag:
                        copy_back_fn(save_image_path, local_eval_sample_dir)
                    if args.use_ema:
                        ema_model.restore(unet.parameters())

        progress_bar.close()

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
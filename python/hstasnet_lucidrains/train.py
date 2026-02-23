import datetime
import time

from shutil import rmtree
import fire

import musdb
from hs_tasnet import HSTasNet, Trainer, MusDB18HQ

import torchaudio
from torch.utils.data import Dataset, ConcatDataset


def train(
    continue_from_checkpoint = False,
    small = False,
    stereo = False,
    batch_size = 16,
    max_steps = 50_000,
    max_epochs = 20,
    use_wandb = False,
    wandb_project = 'HS-TasNet',
    wandb_run_name = None,
    split_dataset_for_eval = True,
    split_dataset_eval_frac = 0.05,
    clear_folders = False,
    use_full_musdb_dataset = False,
    full_musdb_dataset_root = "../training_data",
    model_name= "lr_params"
):

    model = HSTasNet(
        small = small,
        stereo = stereo
    )

    # trainer

    from hs_tasnet import Trainer

    mydataset = MusDB18HQ(dataset_path=full_musdb_dataset_root + "/musdb18hq_augmented")

    trainer = Trainer(
        model,
        continue_from_checkpoint = continue_from_checkpoint,
        dataset = mydataset,                                # add your own
        concat_musdb_dataset = False,                       # whether to concat the musdb dataset
        use_full_musdb_dataset = use_full_musdb_dataset,    # whether to use sample musdb or full
        full_musdb_dataset_root = full_musdb_dataset_root,
        batch_size = batch_size,
        max_steps = max_steps,
        max_epochs = max_epochs,
        use_wandb = use_wandb,
        experiment_project = wandb_project,
        experiment_run_name = wandb_run_name,
        random_split_dataset_for_eval_frac = 0. if not split_dataset_for_eval else split_dataset_eval_frac
    )

    if clear_folders:
        trainer.clear_folders()

    print("Starting training.")
    start_time = time.time()
    trainer()
    end_time = time.time()

    seconds_total = int( end_time - start_time)
    minutes, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes, 60)
    formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"Training done. Took {formatted_duration}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out_path = "./models/" + F"{model_name}_{timestamp}.pt"
    model.save(model_out_path)
    print("Saved model to " + model_out_path)

# fire cli
# --small for small model

if __name__ == '__main__':
    fire.Fire(train)

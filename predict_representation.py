import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='splits/transporters_only.csv')
parser.add_argument('--templates_dir', type=str, default=None)
parser.add_argument('--msa_dir', type=str, default='./alignment_dir')
parser.add_argument('--mode', choices=['alphafold'], default='alphafold')
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--outpdb', type=str, default='./outpdb/default')
parser.add_argument('--pdb_id', nargs='*', default=[])
args = parser.parse_args()
import torch, os
from tqdm import tqdm
import numpy as np
from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import AlphaFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
from alphaflow.data.inference import AlphaFoldCSVDataset, CSVDataset
from openfold.utils.import_weights import import_jax_weights_
from alphaflow.config import model_config
import pickle

torch.cuda.set_device(0)

from alphaflow.utils.logging import get_logger
logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

@torch.no_grad()
def main():

    valset = {
        'alphafold': AlphaFoldCSVDataset,
    }[args.mode](
        data_cfg,
        args.input_csv,
        msa_dir=args.msa_dir,
        templates_dir=args.templates_dir,
    )
    # valset[0]
    logger.info("Loading the model")
    model_class = {'alphafold': AlphaFoldWrapper}[args.mode]

    model = model_class(config, None, training=False)
    if args.mode == 'alphafold':
        import_jax_weights_(model.model, 'params_model_1.npz', version='model_3')
        model = model.cuda()
    else:
        NotImplementedError()
    model.eval()
    logger.info("Model has been loaded")
    os.makedirs(args.outpdb, exist_ok=True)

    # Generate representations
    pbar = tqdm(valset, desc="Getting representations")
    for i, item in enumerate(pbar):
        pbar.set_postfix({"Protein": item['name']})
        if args.pdb_id and item['name'] not in args.pdb_id:
            continue
        batch = collate_fn([item])
        batch = tensor_tree_map(lambda x: x.cuda(), batch)  
        single_representation, pair_representation = model.get_evoformer_representation(batch)

        # Save representation
        evoformer_representation_i = {"representation": {}}
        evoformer_representation_i["representation"]["single"] = single_representation.detach().cpu().numpy()
        evoformer_representation_i["representation"]["pair"] = pair_representation.detach().cpu().numpy()

        # Filename
        filename = f"{args.outpdb}/{item['name']}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(evoformer_representation_i, f)

    logger.info("Representations saved")
if __name__ == "__main__":
    main()

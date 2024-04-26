# Evoformer Representations

Get representations from AlphaFold's Evoformer. Setup the environment from [AlphaFlow](https://github.com/bjing2016/alphaflow).

## Installation
In an environment with Python 3.9 (for example, `mamba create -n [NAME] python=3.9`), run:
```
pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.1 absl-py einops
pip install pytorch_lightning==2.0.4 fair-esm mdtraj wandb
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037'
```

### Preparing input files

1. Prepare a input CSV with an `name` and `seqres` entry for each row. See `splits/demo.csv` for examples.
2. If running an **AlphaFlow** model, prepare an **MSA directory** and place the alignments in `.a3m` format at the following paths: `{alignment_dir}/{name}/a3m/{name}.a3m`. If you don't have the MSAs, there are two ways to generate them:
    1. Query the ColabFold server with `python -m scripts.mmseqs_query --split [PATH] --outdir [DIR]`.
    2. Download UniRef30 and ColabDB according to https://github.com/sokrypton/ColabFold/blob/main/setup_databases.sh and run `python -m scripts.mmseqs_search_helper --split [PATH] --db_dir [DIR] --outdir [DIR]`.
3. If running an **MD+Templates** model, place the template PDB files into a templates directory with filenames matching the names in the input CSV. The PDB files should include only a single chain with no residue gaps.

### Running the model
The basic command for getting Evoformer representation from is:
```
python3 predict_representation.py --mode alphafold --input_csv [PATH] --msa_dir [DIR] --outpdb [DIR]
```

## Model Weights
Download the pretrained AlphaFold weights into the repository root via
```
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xvf alphafold_params_2022-12-06.tar params_model_1.npz
```

## References
This code is based on [AlphaFlow](https://github.com/bjing2016/alphaflow)[1]
1. Bowen Jing, Bonnie Berger, & Tommi Jaakkola. (2024). AlphaFold Meets Flow Matching for Generating Protein Ensembles.

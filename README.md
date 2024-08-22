# PreMut2
PreMut2: deep learning prediction of structures of protein mutants

Accurate prediction of the structure of any protein mutant with a single-site mutation with equivariant graph neural networks. PreMut takes as input a wild-type protein structure and a single-site mutation to predict the structure of the mutated protein with the mutation.

## Installation
* Setup environment by running the following command
```
conda env create -f environment.yml
```
* Activate the environment by running the following command
```
conda activate PreMut2
```

* Install Scwrl4 from [SCWRL4](http://dunbrack.fccc.edu/lab/SCWRLdownload)
* Install TM-align from [TM-align](https://zhanggroup.org/TM-align/)
* Install TM-score from [TM-score](https://zhanggroup.org/TM-score/)

* Run the following command from the terminal to add Scwrl4 to the PATH variable
```
export PATH=/path/to/scwrl4:$PATH
```
## Prediction
* Run the following command
```
python src/Predict.py wild_pdb_path mutation_info chain_id output_dir name
```

* Here wild_pdb_path is the location of the wild pdb file.
* mutation_info is the info regarding the wild residue, position of the wild residue and the mutation residue which replaces it. e.g C_145_A.
* chain_id is the id of the chain in the wild pdb.
* output_dir is the directory where the prediction will be saved.
* name is the desired name you want to give for the predicted pdb file.

## Training
* Run the following command
```
python src/PreMut_train.py
```
## Acknowledgments
The EGNN model code is partially adapted and built upon the source code from the following project [egnn](https://github.com/vgsatorras/egnn). We thank all the contributors and maintainers.



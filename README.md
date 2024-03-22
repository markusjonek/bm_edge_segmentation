# Basement Membrane segmentation GNN

## Usage
Set the config params in `config.py`. Then train and evaluate on the test set by runnnig:

````bash
python train.py --gnn <gnn_type> --graphlet <graphlet_type> --classifier <classifier_type>
````

- `--gnn`: Specifies the type of GNN model to use. Options are:
  - `graphsage`: Uses GraphSAGE for message passing.
  - `eagnn`: Uses EAGNN for message passing.

- `--graphlet`: Specifies the type of graphlet to use. Options are:
  - `edge`: Uses the edge graphlet.
  - `kite`: Uses the kite graphlet.

- `--classifier`: Specifies the type of classifier to use. Options are:
  - `mlp`: Uses a MLP as the classifier.
  - `gcnn`: Uses the GCNN as the classifier.


### Example:
````bash
python train.py --gnn eagnn --graphlet edge --classifier mlp
````
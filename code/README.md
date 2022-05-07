#### Training environment: python3.7, tensorflow2.2, numpy1.18

#### Last_Model_With_Pretrain.py
* Parameters:
Save_dir: The location of the prepared data set.
seq_list: Input data for model training.
label_atrix: Label data for model training.
batch_size: the amount of data in a batch while training.
epochs: Maximum iteration algebra for training

* Function:
pretrain: The network will remove the top constraint layer, and carries out training to obtain the pre-training model.
continue_pretrain: The selected pre-training model will be loaded and the algebras are trained by data iteration.
start_train: Load the selected pre-training model, add a top-level constraint layer on the pre-training model, train the complete model with data, which can be used for transfer learning, and modify the code to add various fine_tune policies.
nopre_train: Create a new complete network with constraint layer, use data to train the whole network from scratch.
start_predict: Predict the data using the trained complete model.

#### data_generator
Raw data will be processed into generated data that can be trained

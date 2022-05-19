# LTPConstraint
LTPConstraint: a transfer learning based end-to-end method for RNA secondary structure prediction
## About the Availability of Data and Materials
Since the amount of data is too large and it is inconvenient to upload, the data acquisition and processing methods are provided here.
### Availability of Data
The source data can be obtained according to the url in the row_data folder.
### How to process data
First, the cd-hit tool needs to be installed. The tool is then used to de-redundant the source data according to the method described in the paper.
Then, store the processed data in the same path, and then modify /code/data_generator/data_construct.py, where File_dir is the path just mentioned, and Save_dir is the storage location of the generated dataset. Run the script after modification.
An example of a generated dataset is /data/512/data_telomerase.npz.
## code
code stored here.
## data
The processed training data is stored in a .npz file as a numpy array.
## logs
Tensorboard log storage location.
## model
The location where the model obtained by training is stored.

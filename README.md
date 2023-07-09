# Trainer

Train and Test a TOD model on the SGD dataset.

Run the trainer using the following command

```bash
sh trainer.sh
```

Different training run configurations are present in the `config/trainer` folder. 
Config files have to be specified in the trainer.sh file, `zs_tod_trainer.yaml` is the default config file.

## Steps

The raw sgd data is converted to an intermediate format using the DataPrep module and csv files are kept in the `processed_data` folder of the project root. The csv file names contain the configuration.

The datamodules load the processed csv files and feeds it into the GPT2(or any other) model.
Early stopping using eval loss is used to prevent overfitting.
After training, the inference module is called, which generates the target given the context as the prompt. Different metrics related to TOD systems are calculated and printed in the log files and also in system output.

## Training Configuration


`trainer_base.yaml` is the baseline method that is similar to SimpleTOD.
`zs_tod_trainer.yaml` is the configuration for the Zs-ToD model in this paper.

The project root needs to be specified, as a lot of paths are relative to it.
If GPU out of memory occurs, please adjust the batch size of training/pretraining, eval and test.
Domains for each step can be provided in the domain settings. Example is in the commented out section in `trainer_base.yaml` file.
Data split percent is an array which denotes how much of the data should be used for each step. The value ranges from `0-1`. A value of `0.1,1,1` would mean train on `10%` of the data, dev on `100%` and test on `100%`.
Num_dialogs specifies the number of dialog files to use. Its an array of length 3, which represents train, dev and test.
Overwrite parameter, when true will not perform data prep if the processed data with the current configuration already exists.

# Data Prep

Reads raw sgd data and converts it into zs-tod input format so that it can be fed into the model.

Configurations can be changed by editing the following file
```
config/data_prep/zs_tod.yaml
```

Run this procedure using the following command from the root directory of the project

```bash
sh data_prep.sh
```


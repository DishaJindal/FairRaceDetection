# Run

python main_script.py

**Paths**

Dataset (main_script.py): Edit "train_dataset" and "test_dataset" paths in the main_script

Logger (helper_functions.py): Edit "logs_base_dir" path with the logger directory

## Scipts

**preprocessing_script.py**: We preprocess the data and save it as numpy arrays so avoid file IO at each epoch

**dataset.py**: Dataset class to load the data

**attention.py**: Contains different attention modules

**backbone.py**: Contains all custom resnet related utilities

**helper_functions.py**: Functions for the logger, deleting logs and nvidia stats

**metrics.py**: Bias Metrics

Please refer report for more information!

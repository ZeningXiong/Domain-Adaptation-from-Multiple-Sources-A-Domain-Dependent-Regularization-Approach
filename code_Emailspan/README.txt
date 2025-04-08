FastDAM：
First run the run-svm-fr.py to get the basic classifier
According to N= 20 as described in paper.

Then run the save-mmd-fr.py to get mmd-values.

Finally run the run-fast-dam.py to get the results of FastDAM.

All the results are saved to result-target.txt files.




UniverDAM:
run_univer_dam.py
This file is the core of the whole process, responsible for loading data, setting training parameters, calling the main training function to train the model, and saving the results. Finally, the training results will be output to a .txt file, and the visualization of the results will be displayed.

load_data.py
Contains the relevant functions for data loading, loading the default data set for training.

main_univerdam_m.py
This file implements the core training process of the model and returns the training results.

show_result_all_univer_dam.py
Responsible for displaying the visualization content of the training results.
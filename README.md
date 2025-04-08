For 20Newsgroups task

FastDAM：
First run the run-svm-s.py to get the basic classifier, where s is the setting, you can change for different dataset like sci vs comp
According to N=0 to 20 as described in paper.

Then run the save-mmd-at.py to get mmd-values.

Finally run the run-fast-dam.py to get the results of FastDAM.

All the results are saved to result-target.txt files.


UniverDAM: 
By running the run_univer_dam file, different data sets are automatically selected and processed, relevant model training is performed, and the results are saved in the specified folder. You can select different data sets for experiments by modifying the setting parameters.

run_univer_dam.py:
The core function of this file is to load the corresponding data set according to the given setting and perform model training. The training results will be saved in the specified folder, and the visualization results will also be displayed.

load_data.py:
This file contains related functions for data loading, and different data sets are loaded according to different settings.

show_result_all_univer_dam.py: 
This file is responsible for displaying the visualization of all experimental results.

main_univerdam_m.py:
This file implements the main training process of the model and returns the training results.



\\
\\
\\



For Emailspan task

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

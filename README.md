# NLP-Project
Assessing the Funniness of Edited News Headlines (SemEval-2020 Task 7)

## To Run Task 1 and 2 using XGB:

### Task 1
  >python xgb_task1.py

###Task 2
  >python xgb_task2.py

## To Run Task 1 using Linear Regression:

### Train a new Model:
 >python task1.py -tr yes -ds path/of/csv/file -m /model/directory/to/save -tts optional

### To Test a Model:
  >python task1.py -te yes -ds path/of/csv/file -m /model/directory
  *run this command to test the existing pre-trained model*
  >python3 task1.py -te yes -ds ./training-data/task-1/train_funlines.csv -m ./training-data/task-1/

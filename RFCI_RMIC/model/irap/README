# IRAP

An intelligent protein analysis package based on the Reduced Amino Acids Codes and Position Specific Scoring Matrix.

## Introduction

IRAP package is a dedicated python package for protein analysis. It can analyze the sequence and site characteristics of the protein through the Reduced Amino Acids Codes and Position Specific Scoring Matrix (RAAC-PSSM) method, and use Support Vector Machines (SVM) for protein function and site prediction. IRAP has built-in complete machine learning functions and personalized analysis modules, which can meet the general needs for sequence analysis. In addition, we provide users with convenient operation instructions on Windows system, Linux system and PythonIDE environment, respectively. In the following we will focus on the installation and application of IRAP under Linux system.
## Installation
The IRAP package needs to be run in a python environment. We recommend users to install via conda or pip. IRAP needs the support of pyecharts, scikit-learn, ray, joblib, and they will be installed automatically when you install IRAP. We published IRAP in the GitHub community, you need to install git first and then install IRAP through the following link.

```bash
# install git
$conda install git

# install IRAP by conda
$conda install git+https://github.com/KingoftheNight/IRAP.git

# install IRAP by pip

$pip install git+https://github.com/KingoftheNight/IRAP.git
```
## Run IRAP by windows
```bash
$irap windows
```
## Run IRAP by linux
```bash
$irap [Fuctions] <parameters>
```
## Load IRAP in PythonIDE

```python
from irap import SVM as isvm
from irap import Load as iload
from irap import Read as iread
from irap import Extract as iextra
from irap import Evaluate as ieval
from irap import Select as iselect
```

## Usage For Linux

You can use the easy function to complete all the following processes in one step, and IRAP will automatically help you optimize all parameters.

```bash
$irap easy -tp file_name -tn file_name -pp file_name -pn file_name -db blast_database_name -raa RAACBook -s ifs_method

# optional arguments:
#   -tp        input your train positive datasets name.
#   -tn        input your train negative datasets name.
#   -pp        input your predict positive datasets name.
#   -pn        input your predict negative datasets name.
#   -db        choose the blast database, or you can make your database through Makedb function (17).
#   -raa       raac book saved in raacDB folder in IRAP, and we provide two RAACBook files with size 8 (minCODE) and size 671 (raaCODE).
#   -s         choose the feature selection method, and we provide two method, rf (relief-fscore) and pca (pcasvd).
```

### Example

```bash
$irap easy -tp train_positive.fasta -tn train_negative.fasta -pp predict_positive.fasta -pn predict_negative.fasta -db pdbaa -raa minCODE -s rf
```

You can also run the following independent functions according to your needs.

### 1. Read
Load Fasta datasets and split them into separate fasta files.
##### Command line
```bash
$irap read file_name -o out_folder

# optional arguments:
#   file_name  input your Fasta datasets name.
#   -o         input the out folder name, and it will be saved by default in Reads folder.
```
##### Example
```bash
$irap read train_positive.fasta -o train_p
$irap read train_negative.fasta -o train_n
$irap read predict_positive.fasta -o predict_p
$irap read predict_negative.fasta -o predict_n
```


### 2. Blast

Get PSSM profiles through psiblast function provided by _BLAST+_ (https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).
##### Command line
```bash
$irap blast folder_name -db blast_database_name -n num_iterations -ev expected_value -o out_folder_name

# optional arguments:
#   folder_name  input the Fasta folder name which has been created in Read function (1).
#   -db          choose the blast database, or you can make your database through Makedb function (17).
#   -n           input the iteration number.
#   -ev          input the expected value.
#   -o           input the out folder name, and it will be saved by default in PSSMs folder.
```
##### Example
```bash
$irap blast train_p -db pdbaa -n 3 -ev 0.001 -o pssm-tp
$irap blast train_n -db pdbaa -n 3 -ev 0.001 -o pssm-tn
$irap blast predict_p -db pdbaa -n 3 -ev 0.001 -o pssm-pp
$irap blast predict_n -db pdbaa -n 3 -ev 0.001 -o pssm-pn
```
### 3. Extract
Extract PSSM files through 7 extraction methods, and output them in a new folder.
#### Command line
```bash
$irap extract folder_name_1 folder_name_2 -raa RAACBook -o out_folder -l windows_size -r self_raa_code

# optional arguments:
#   folder_name  input the two PSSM folders name which has been created in Blast function (2).
#   -raa         raac book saved in raacDB folder in rpct, and you can not use this parameter and -r together.
#   -o           input the out folder name
#   -l           input the window size for the RAACSW extraction method.
#   -r           self_raa_code format should contain all amino acid types, and be separated by '-', for example: LVIMC-AGST-PHC-FYW-EDNQ-KR.
```
##### Example
```bash
$irap extract pssm-tp pssm-tn -raa minCODE -o Train_fs -l 5
$irap extract pssm-tp pssm-tn  -o Train_fs_sr -l 5 -r LVIMC-AGST-PHC-FYW-EDNQ-KR
$irap extract pssm-pp pssm-pn -raa minCODE -o Predict_fs -l 5
```
### 4. Search
Search hyperparameters of the target feature file or folder through the grid function.
##### Command line
```bash
$irap search -d document_name -f folder_name

# optional arguments:
#   -d    input the target feature file and output a single result of it, and you can not use this parameter and -f together.
#   -f    input the target feature folder and output the Hyperparameters file which contains all results of the feature folder, and you can not use this parameter and -d together.
```
##### Example
```bash
$irap search -d .\Train_fs\0001-t1s2-0123456.rap
$irap search -f Train_fs
```
### 5. Filter
Filter the features of the target feature file through the IFS-RF method (Incremental Feature Selection based on the Relief-Fscore method). And output an Accuracy line chart and a feature sort file for the target feature file.
##### Command line
```bash
$irap filter document_name -c c_number -g gamma -cv cross_validation_fold -o out_file_name -r random_number

# optional arguments:
#   document_name   input the target feature file name.
#   -c              the penalty coefficient of SVM, you can get it through Search function (4) or define it by your experience.
#   -g              the gamma of RBF-SVM, you can get it through Search function (4) or define it by your experience.
#   -cv             the cross validation fold of SVM, you can choose 5, 10 or -1 or define it by your experience.
#   -o              input the out file name, and the ACC Chart and Feature sort file will be saved by defualt in current folder.
#   -r              the random sampling number of Relief method.
```
##### Example
```bash
$irap filter .\Train_fs\0001-t1s2-0123456.rap -c 128 -g 0.03125 -cv 5 -o t1s2 -r 30
```
### 6. Filter Features File Setting
Create a filtered feature file of the target feature file through the feature sort file which has been output in Filter function.
##### Command line
```bash
$irap fffs document_name -f feature_sort_file -n stop_feature_number -o out_file_name

# optional arguments:
#   document_name    input the target feature file which has been chosen in Filter function.
#   -f               input the Feature_Sort_File which has been created in Filter function.
#   -n               the stop feature number of target feature file, and you can find it in the ACC_Chart which has been created in Filter function (5).
#   -o               input the out file name.
```
##### Example
```bash
$irap fffs .\Train_fs\0001-t1s2-0123456.rap -f t1s2\Fsort-rf.txt -n 6 -o t1s2-6.rap
$irap fffs .\Predict_fs\0001-t1s2-0123456.rap -f t1s2\Fsort-rf.txt -n 6 -o t1s2-6_pred.rap
```
### 7. Train
Train feature files through the SVM.
##### Command line
```bash
$irap train -d document_name -f input_folder_name -c c_number -g gamma -o out_folder -cg Hyperparameters_name

# optional arguments:
#   -d    input the target feature file, and you can not use this parameter with -f and -cg together.
#   -f    input the feature folder, and you can not use this parameter with -d, -c and -g together.
#   -c    the penalty coefficient of SVM, and you can not use this parameter with -f and -cg together.
#   -g    the gamma of RBF-SVM, and you can not use this parameter with -f and -cg together.
#   -o    if you choose the parameter -f, you should input a folder name, and if you choose the parameter -d, you should input a file name.
#   -cg   the Hyperparameters file which has been created in Search function (4) or defined in Mhys function (14), and you can not use this parameter with -d, -c and -g together.
```
##### Example
```bash
$irap train -d .\Train_fs\0001-t1s2-0123456.rap -c 128 -g 0.03125 -o t1s2-6.model
$irap train -f Train_fs -o Models -cg Hys_Train_fs.txt
```
### 8. Eval
Evaluate feature files through the Cross-validation function.
##### Command line
```bash
$irap eval -d document_name -f folder_name -c c_number -g gamma -o out_folder -cg Hyperparameters_name -cv cross_validation_fold

# optional arguments:
#   -d    input the target feature file, and you can not use this parameter with -f and -cg together.
#   -f    input the feature folder, and you can not use this parameter with -d, -c and -g together.
#   -c    the penalty coefficient of SVM, and you can not use this parameter with -f and -cg together.
#   -g    the gamma of RBF-SVM, and you can not use this parameter with -f and -cg together.
#   -o    if you choose the parameter -f, you should input a folder name, and if you choose the parameter -d, you should input a file name.
#   -cg   the Hyperparameters file which has been created in Search function (4) or defined in Mhys function (14), and you can not use this parameter with -d, -c and -g together.
#   -cv   the cross validation fold of SVM, you can choose 5, 10 or -1 or define it by your experience.
```
##### Example
```bash
$irap eval -d .\t1s2-2.rap -c 128 -g 0.03125 -cv 5 -o t1s2-2.txt
$irap eval -f Train_fs -cv 5 -o Eval_fs -cg Hys_Train_fs.txt
```
### 9. ROC
Draw the ROC-Cruve by scikit-learn.
##### Command line
```bash
$irap roc document_name -c c_number -g gamma -o out_file_name

# optional arguments:
#   document_name    input the target feature file.
#   -c               the penalty coefficient of SVM, and you can not use this parameter with -f and -cg together.
#   -g               the gamma of RBF-SVM, and you can not use this parameter with -f and -cg together.
#   -o               input the out file name, and the ROC-Cruve will be saved by defualt in current folder.
```
##### Example
```bash
$irap roc .\t1s2-2.rap -c 128 -g 0.03125 -o t1s2-roc
```
### 10. Predict
Evaluate the target model with a feature files which from an independent datasets. And output a Evaluation_file and a Prediction_result for the target model.
##### Command line
```bash
$irap predict document_name -m model_name -o out_file_name

# optional arguments:
#   document_name    input the target feature file, and make sure it has the same reduce type with the target model.
#   -m               input the target model file, and make sure it has the same reduce type with the target feature file.
#   -o               input the out file name, and the predict result will be saved by defualt in current folder.
```
##### Example
```bash
$irap predict .\t1s2-2_pred.rap -m .\t1s2-2.model -o t1s2.csv
```
### 11. Res
Reduce amino acids by personal rules. And output a personal RAAC list from size_2 to size_19.
##### Command line
```bash
$irap res aaindex_id

# optional arguments:
#   aaindex_id    the ID of physical and chemical characteristics in AAindex Database, and you can check it in aaindexDB folder in rpct folder or view it online.
```
##### Example
```bash
$irap res CHAM830102
```
### 12. Lblast
Download the full-featured blast program at the fastest speed so that you can accidentally delete or damage blast-related functions. The download includes the sequence alignment program necessary for IRAP and the pdbaa alignment database.

##### Command line

```bash
$irap lblast
```
### 13. PCA
Filter the features of the target feature file through the IFS-PCASVD method (Incremental Feature Selection based on the Singular Value Decomposition Principal Component Analysis method). And output an Accuracy line chart and a feature sort file for the target feature file.
##### Command line
```bash
$irap pca document_name -c c_number -g gamma -o out_file_name -cv cross_validation_fold

# optional arguments:
#   document_name    input the feature file name.
#   -c               the penalty coefficient of SVM.
#   -g               the gamma of RBF-SVM.
#   -o               input the out file name, and the ACC Chart and Feature sort file will be saved by defualt in current folder.
#   -cv              the cross validation fold of SVM, you can choose 5, 10 or -1 or define it by your experience.
```
##### Example
```bash
$irap pca .\Train_fs\0001-t1s2-0123456.rap -c 128 -g 0.03125 -o t1s2-pca -cv 5
```
### 14. Mhys
Define hyperparameters file for a target feature folder by your experience.
##### Command line
```bash
$irap mhys folder_name -c c_number -g gamma -o out_file_name

# optional arguments:
#   folder_name    input the train feature folder name.
#   -c             the penalty coefficient of SVM.
#   -g             the gamma of RBF-SVM.
#   -o             input the out file name, and the Hyperparameters file will be saved by defualt in current folder.
```
##### Example
```bash
$irap mhys Train_fs -c 2 -g 0.125 -o Hys_2.txt
```
### 15. Rblast
Use the ray package for multi-threaded psiblast comparison.
##### Command line
```bash
$irap rblast folder_name -db blast_database_name -n num_iterations -ev expected_value -o out_folder_name

# optional arguments:
#   folder_name  input the Fasta folder name which has been created in Read function (1).
#   -db          choose the blast database, or you can make your database through Makedb function (17).
#   -n           input the iteration number.
#   -ev          input the expected value.
#   -o           input the out folder name, and it will be saved by default in PSSMs folder.
```
##### Example
```bash
$irap rblast train_p -db pdbaa -n 3 -ev 0.001 -o pssm-tp
$irap rblast train_n -db pdbaa -n 3 -ev 0.001 -o pssm-tn
$irap rblast predict_p -db pdbaa -n 3 -ev 0.001 -o pssm-pp
$irap rblast predict_n -db pdbaa -n 3 -ev 0.001 -o pssm-pn
```
### 16. Reduce
Draw a reduced sequence diagram of the specified FASTA sequence.
##### Command line
```bash
$irap reduce file_name -raa raacode -o out_file_name

# optional arguments:
#   file_name    input the target fasta file which has been created in Read function (1).
#   -raa         reduce amino acid code like 'LVIMC-AGST-PHC-FYW-EDNQ-KR'.
#   -o           input the out folder name, and the folder saved by default in PSSMs folder.
```
##### Example
```bash
$irap reduce .\Reads\train_p\1.fasta -raa LVIMC-AGST-PHC-FYW-EDNQ-KR -o selfraa.html
```
### 17. Makedb
Make blast database by makeblastdb function provided by BLAST+. You can make a personal datasets or choose a public database from Blast (https://ftp.ncbi.nlm.nih.gov/blast/db/).
##### Command line
```bash
$irap makedb datasets_name -o out_database_name

# optional arguments:
#   database_name    input the target fasta database, and make sure it located in the current folder.
#   -o               input the out database name, and it will be saved by default in blastDB folder in rpct folder.
```
##### Example
```bash
$irap makedb pdbaa -o pdbaa
```
### 18. View
View the RAAC Map of different types in target RAAC Book. And the result will be saved in a html file by pyecharts packages.
##### Command line
```bash
$irap view raac_book_name -t type_raac

# optional arguments:
#   raac_book_name    input the target RAAC book name.
#   -t                input the target type which you want to view.
```
##### Example
```bash
$irap view raaCODE -t 5
```
### 19. Pmlogo
View the reduce pssm logo of target pssm profile.
##### Command line
```bash
$irap pmlogo pssm_file_name -raa raac_book_name -r reduce_type -o out_file_name

# optional arguments:
#   pssm_file_name    input the target pssm file path.
#   -raa              input the target RAAC book name.
#   -r                input the target reduce type
#   -o                input the out file name, and the file saved by default in current folder.
```
##### Example
```bash
$irap pmlogo .\PSSMs\pssm-tp\1.pssm -raa minCODE -r 0001-t1s2 -o t1s2.html
```
### 20. Check Blast Database
Check and remove the repetitive sequences in blast database.
##### Command line
```bash
$irap checkdb database_name

# optional arguments:
#   database_name    input the target blast database of FASTA format.
```
##### Example
```bash
$irap checkdb pdbaa
```

## Usage For Windows

You can input command in CMD to run IRAP GUI on windows platform.

```bash
$irap windows
```

## Usage For Python

You can import IRAP in PythonIDE.

```python
from irap import SVM as isvm
from irap import Load as iload
from irap import Read as iread
from irap import Extract as iextra
from irap import Evaluate as ieval
from irap import Select as iselect
```


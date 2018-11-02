@echo off
echo Started: %date% %time%
REM make mktree
REM OC1_v3\mktree -tdata\iris\OC1_iris_train.data -V150
REM OC1_v3\mktree -tdata\multiple-features\OC1_multi_feat_train.data -V5

REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
javac -Xlint:unchecked -d project/target project/src/*.java
set test=data/arcene/arcene_valid.data
set train=data/arcene/arcene_train.data
set labels=data/arcene/arcene_train.labels
set final_result=data/arcene/results/FODT_results.data
set sparse=dense
java -cp project/target ClassificationDriver %sparse% %test% %train% %labels% %final_result%
python diff.py %final_result% data/arcene/arcene_valid.labels
echo Completed: %date% %time%

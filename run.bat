@echo off
echo Started: %date% %time%
REM make mktree
REM OC1_v3\mktree -tdata\iris\OC1_iris_train.data -V150
REM OC1_v3\mktree -tdata\multiple-features\OC1_multi_feat_train.data -V5

REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
javac -Xlint:unchecked -d project/target project/src/*.java
set dataset=dorothea
set train=data/%dataset%/%dataset%.data
set labels=data/%dataset%/%dataset%.labels
set sparse=sparse
java -cp project/target CVDriver %sparse% %train% %labels%
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
echo Completed: %date% %time%

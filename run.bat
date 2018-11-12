@echo off
echo Started: %date% %time%
set dataset=iris
set n=1
set train=data/%dataset%/folds/5-folds/%dataset%%n%-train.data
set test=data/%dataset%/folds/5-folds/%dataset%%n%-test.data
REM OC1_v3\mktree -t%train% -T%test%
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
javac -Xlint:unchecked -d project/target project/src/*.java
set dataset=arcene
set train=data/%dataset%/%dataset%.data
set labels=data/%dataset%/%dataset%.labels
set sparse=dense
java -cp project/target CVDriver %sparse% %train% %labels% 5 1005 -f
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
echo Completed: %date% %time%

@echo off
echo Started: %date% %time%
REM set dataset=dermatology
REM FOR /l %%n IN (1,1,5) DO (
REM   OC1_v3\mktree -tdata/%dataset%/folds/5-folds/%dataset%%%n-train.data -Tdata/%dataset%/folds/5-folds/%dataset%%%n-test.data
REM )

REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
javac -Xlint:unchecked -d project/target project/src/*.java
set dataset=multiple-features
set train=data/%dataset%/%dataset%.data
set labels=data/%dataset%/%dataset%.labels
set sparse=dense
java -cp project/target CVDriver %sparse% %train% %labels% 5 1005
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
echo Completed: %date% %time%

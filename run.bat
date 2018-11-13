@echo off
echo Started: %date% %time%
set dataset=arcene
FOR /l %%n IN (1,1,5) DO (
  OC1_v3\mktree -tdata/%dataset%/folds/5-folds/%dataset%%%n-train.data -Tdata/%dataset%/folds/5-folds/%dataset%%%n-test.data
)
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
REM javac -Xlint:unchecked -d project/target project/src/*.java
REM set dataset=iris
REM set train=data/%dataset%/%dataset%.data
REM set labels=data/%dataset%/%dataset%.labels
REM set sparse=dense
REM java -cp project/target CVDriver %sparse% %train% %labels% 5 1005
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
echo Completed: %date% %time%

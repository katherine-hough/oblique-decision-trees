@echo off
set dataset=iris
set sparse=dense
REM ---------------------------------------------------------------------------------
FOR /l %%n IN (1,1,5) DO (
  OC1\mktree -tdata/%dataset%/folds/5-folds/%dataset%%%n-train.data -Tdata/%dataset%/folds/5-folds/%dataset%%%n-test.data -s1005 -z
)
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
REM javac -Xlint:unchecked -d project/target project/src/*.java
REM java -cp project/target CVDriver %sparse% data/%dataset%/%dataset%.data data/%dataset%/%dataset%.labels 5 1005
REM ---------------------------------CART---------------------------------------------
REM echo Started: %date% %time%
REM python CART/main.py data/%dataset%/folds/5-folds/%dataset% 5 %sparse%
REM echo Completed: %date% %time%

@echo off
set dataset=wine
set sparse=dense
REM ---------------------------------------------------------------------------------
REM echo Started: %date% %time%
REM FOR /l %%n IN (1,1,5) DO (
REM   OC1\mktree -tdata/%dataset%/folds/5-folds/%dataset%%%n-train.data -Tdata/%dataset%/folds/5-folds/%dataset%%%n-test.data
REM )
REM echo Completed: %date% %time%
REM ----------------------------FAST OBLIQUE DECISION TREE----------------------------------------
javac -Xlint:unchecked -d project/target project/src/*.java
java -cp project/target CVDriver %sparse% data/%dataset%/%dataset%.data data/%dataset%/%dataset%.labels 5 1005
REM ---------------------------------CART---------------------------------------------
REM echo Started: %date% %time%
REM python CART/main.py data/%dataset%/folds/5-folds/%dataset% 5 %sparse%
REM echo Completed: %date% %time%

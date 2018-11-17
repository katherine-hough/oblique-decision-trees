@echo off
set dataset=arcene
set sparse=dense

FOR /l %%n IN (1,1,1) DO (
  OC1\mktree -tdata/%dataset%/folds/5-folds/%dataset%%%n-train.data -Tdata/%dataset%/folds/5-folds/%dataset%%%n-test.data -s1005 -vv
)

REM javac -Xlint:unchecked -d project/target project/src/*.java
REM java -cp project/target CVDriver %sparse% data/%dataset%/%dataset%.data data/%dataset%/%dataset%.labels 5 484 GA-ODT

REM python CART/main.py data/%dataset%/folds/5-folds/%dataset% 5 %sparse%

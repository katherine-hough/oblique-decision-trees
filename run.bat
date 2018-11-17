@echo off
set dataset=multiple-features
set sparse=dense

REM FOR /l %%n IN (1,1,5) DO (
REM   OC1\mktree -tdata/%dataset%/folds/5-folds/%dataset%%%n-train.data -Tdata/%dataset%/folds/5-folds/%dataset%%%n-test.data -s1005 -z
REM )

javac -Xlint:unchecked -d project/target project/src/*.java
java -cp project/target CVDriver %sparse% data/%dataset%/%dataset%.data data/%dataset%/%dataset%.labels 5 484 GA-ODT

REM python CART/main.py data/%dataset%/folds/5-folds/%dataset% 5 %sparse%

@echo off
set dataset=iris
set sparse=dense

javac -Xlint:unchecked -d project/target project/src/*.java
java -cp project/target CVDriver %sparse% data/%dataset%/%dataset%.data data/%dataset%/%dataset%.labels 5 484 DT

REM java -cp project/target ClassificationDriver %sparse% data/%dataset%/%dataset%.data data/%dataset%/%dataset%.data data/%dataset%/%dataset%.labels predictions.labels GA-ODT

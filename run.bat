@echo off
REM ============================================================
REM Run the application - SVM + HOG + SMO
REM If an OpenCV jar exists in lib\, it is added to the classpath.
REM The webcam works ONLY with OpenCV in lib\ and the native
REM library (opencv_javaXXX.dll) in lib\ or on the PATH.
REM ============================================================

set CP=bin

REM Add every jar in lib\ to the classpath
if exist lib (
    for %%f in (lib\*.jar) do call :append %%f
)

echo Classpath: %CP%
REM --enable-native-access=ALL-UNNAMED silences JDK 21+ warnings
REM -Xmx4g lets Java use up to 4 GB of RAM (needed for large trainings
REM with many images - the kernel matrix grows as n^2).
java -Xmx4g --enable-native-access=ALL-UNNAMED -cp "%CP%" -Djava.library.path=lib svm.Main
goto :eof

:append
set CP=%CP%;%1
goto :eof
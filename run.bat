@echo off
REM ============================================================
REM Rulare aplicatie - SVM + HOG + SMO
REM Daca exista un jar OpenCV in lib\, e inclus in classpath.
REM Webcam-ul functioneaza NUMAI cu OpenCV in lib\ si biblioteca
REM nativa (opencv_javaXXX.dll) in lib\ sau pe PATH.
REM ============================================================

set CP=bin

REM Adauga toate jar-urile din lib\ in classpath
if exist lib (
    for %%f in (lib\*.jar) do call :append %%f
)

echo Classpath: %CP%
REM --enable-native-access=ALL-UNNAMED elimina warning-urile JDK 21+
REM -Xmx4g aloca pana la 4 GB RAM pentru Java (necesar pentru antrenari
REM mari cu multe poze - matricea kernel creste rapid cu n^2).
java -Xmx4g --enable-native-access=ALL-UNNAMED -cp "%CP%" -Djava.library.path=lib svm.Main
goto :eof

:append
set CP=%CP%;%1
goto :eof

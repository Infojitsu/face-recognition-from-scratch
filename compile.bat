@echo off
REM ============================================================
REM Compilare proiect Java - SVM + HOG + SMO
REM Cerinta: sa se foloseasca javac (JDK) fara niciun IDE.
REM ============================================================

if not exist bin mkdir bin

REM Enumeram direct toate pachetele (mai sigur decat dir /s /b, care
REM pe Windows poate produce fisiere cu codare sau BOM gresit).
javac -d bin ^
    src\svm\*.java ^
    src\svm\core\*.java ^
    src\svm\hog\*.java ^
    src\svm\svm\*.java ^
    src\svm\detector\*.java ^
    src\svm\io\*.java ^
    src\svm\util\*.java ^
    src\svm\ui\*.java

if errorlevel 1 (
    echo.
    echo [EROARE] Compilare esuata.
    pause
    exit /b 1
)

echo.
echo [OK] Compilare reusita. Fisierele .class sunt in bin\
pause

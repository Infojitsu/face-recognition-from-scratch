@echo off
REM ============================================================
REM Java project compilation - SVM + HOG + SMO
REM Requirement: use javac (JDK) without any IDE.
REM ============================================================

if not exist bin mkdir bin

REM Enumerate all packages directly (safer than dir /s /b, which
REM on Windows can produce files with wrong encoding or BOM).
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
    echo [ERROR] Compilation failed.
    pause
    exit /b 1
)

echo.
echo [OK] Compilation successful. The .class files are in bin\
pause

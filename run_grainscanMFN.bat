@echo off
REM Активуємо віртуальне середовище
call venv\Scripts\activate

REM Запускаємо Streamlit-додаток
streamlit run GrainScanAppMFN.py

REM Залишаємо консоль відкритою після завершення
pause

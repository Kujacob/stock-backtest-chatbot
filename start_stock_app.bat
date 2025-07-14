@echo off
REM ================================================================
REM ==         自然語言股票策略回測系統 - 一鍵啟動腳本         ==
REM ================================================================

ECHO.
ECHO  正在啟動後端伺服器...
ECHO.
REM 在一個新的命令提示字元視窗中，切換到 backend 資料夾並啟動 uvicorn 伺服器
start "Backend Server" cmd /k "cd backend && uvicorn main:app --reload"

ECHO.
ECHO  正在啟動前端應用程式... (請稍候)
ECHO.
REM 等待 5 秒，讓後端有足夠的時間初始化
timeout /t 5 > nul

REM 在另一個新的命令提示字元視窗中，切換到 frontend 資料夾並啟動 React 開發伺服器
REM (這個指令假設您的前端是使用 React)
start "Frontend Server" cmd /k "cd frontend && npm start"

ECHO.
ECHO  準備在瀏覽器中開啟應用程式...
ECHO.
REM 再等待 10 秒，讓前端開發伺服器完成編譯
timeout /t 10 > nul

REM 在您的預設瀏覽器中打開應用程式
start http://localhost:3000

ECHO.
ECHO  所有服務皆已啟動！
ECHO  您可以分別在 "Backend Server" 和 "Frontend Server" 視窗中查看日誌。
ECHO.
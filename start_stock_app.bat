@echo off
REM ================================================================
REM ==         �۵M�y���Ѳ������^���t�� - �@��Ұʸ}��         ==
REM ================================================================

ECHO.
ECHO  ���b�Ұʫ�ݦ��A��...
ECHO.
REM �b�@�ӷs���R�O���ܦr���������A������ backend ��Ƨ��ñҰ� uvicorn ���A��
start "Backend Server" cmd /k "cd backend && uvicorn main:app --reload"

ECHO.
ECHO  ���b�Ұʫe�����ε{��... (�еy��)
ECHO.
REM ���� 5 ��A����ݦ��������ɶ���l��
timeout /t 5 > nul

REM �b�t�@�ӷs���R�O���ܦr���������A������ frontend ��Ƨ��ñҰ� React �}�o���A��
REM (�o�ӫ��O���]�z���e�ݬO�ϥ� React)
start "Frontend Server" cmd /k "cd frontend && npm start"

ECHO.
ECHO  �ǳƦb�s�������}�����ε{��...
ECHO.
REM �A���� 10 ��A���e�ݶ}�o���A�������sĶ
timeout /t 10 > nul

REM �b�z���w�]�s���������}���ε{��
start http://localhost:3000

ECHO.
ECHO  �Ҧ��A�ȬҤw�ҰʡI
ECHO  �z�i�H���O�b "Backend Server" �M "Frontend Server" �������d�ݤ�x�C
ECHO.
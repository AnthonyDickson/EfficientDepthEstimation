
@ECHO OFF
SETLOCAL

IF "%selfWrapped%"=="" (
  REM this is necessary so that we can use "exit" to terminate the batch file,
  REM and all subroutines, but not the original cmd.exe
  SET selfWrapped=true
  %ComSpec% /s /c ""%~0" %*"
  GOTO :EOF
)

setlocal enabledelayedexpansion

FOR /L %%i IN (1,1,5) DO (
    FOR %%d IN (hu2018, lasinger2019) DO (
        FOR %%e IN (efficientnet-b0, efficientnet-b4, resnet50) DO (
            ECHO python -m ReSIDE.train --encoder %%e --decoder %%d
            python -m ReSIDE.train --encoder %%e --decoder %%d
            IF !errorlevel! NEQ 0 EXIT !errorlevel!
        )
    )
)
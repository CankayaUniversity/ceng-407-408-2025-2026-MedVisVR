Dim scriptDir, batPath
scriptDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
batPath    = scriptDir & "Start_Desktop_App_OneClick.bat"
Dim ws
Set ws = CreateObject("WScript.Shell")
ws.Run Chr(34) & batPath & Chr(34), 0, False

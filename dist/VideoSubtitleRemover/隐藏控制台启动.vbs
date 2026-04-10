Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & WScript.ScriptFullName & "\..\VideoSubtitleRemover.exe" & chr(34), 0, False

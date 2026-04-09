Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class KeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
}
"@

$endTime = (Get-Date).AddHours(12)
Write-Host "=========================================="
Write-Host " SCREEN KEEP-ALIVE ACTIVE"
Write-Host " Until: $($endTime.ToString('HH:mm:ss on yyyy-MM-dd'))"
Write-Host " Press Ctrl+C to stop early"
Write-Host "=========================================="

while ((Get-Date) -lt $endTime) {
    [KeepAwake]::SetThreadExecutionState(
        [KeepAwake]::ES_CONTINUOUS -bor
        [KeepAwake]::ES_DISPLAY_REQUIRED -bor
        [KeepAwake]::ES_SYSTEM_REQUIRED
    ) | Out-Null
    $remaining = $endTime - (Get-Date)
    Write-Host "`r[$(Get-Date -Format 'HH:mm:ss')] Alive - $([math]::Floor($remaining.TotalHours))h $($remaining.Minutes)m remaining" -NoNewline
    Start-Sleep -Seconds 60
}

# Reset to default when done
[KeepAwake]::SetThreadExecutionState([KeepAwake]::ES_CONTINUOUS) | Out-Null
Write-Host ""
Write-Host "Keep-alive ended. Normal power settings restored."

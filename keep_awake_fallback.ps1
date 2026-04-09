Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class KeepAwakeFallback {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, int dx, int dy, uint dwData, int dwExtraInfo);
    public const uint MOUSEEVENTF_MOVE = 0x0001;

    public static void Jiggle() {
        mouse_event(MOUSEEVENTF_MOVE, 1, 0, 0, 0);
        System.Threading.Thread.Sleep(50);
        mouse_event(MOUSEEVENTF_MOVE, -1, 0, 0, 0);
    }
}
"@

# Also force power settings via powercfg
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-dc 0
powercfg /change standby-timeout-dc 0

$endTime = (Get-Date).AddHours(12)
Write-Host "=========================================="
Write-Host " FALLBACK KEEP-ALIVE (mouse jiggle + API)"
Write-Host " Until: $($endTime.ToString('HH:mm:ss on yyyy-MM-dd'))"
Write-Host " powercfg timeouts set to NEVER"
Write-Host "=========================================="

while ((Get-Date) -lt $endTime) {
    # Method 1: SetThreadExecutionState
    [KeepAwakeFallback]::SetThreadExecutionState(
        [KeepAwakeFallback]::ES_CONTINUOUS -bor
        [KeepAwakeFallback]::ES_DISPLAY_REQUIRED -bor
        [KeepAwakeFallback]::ES_SYSTEM_REQUIRED
    ) | Out-Null

    # Method 2: Mouse jiggle (moves 1px right then back)
    [KeepAwakeFallback]::Jiggle()

    $remaining = $endTime - (Get-Date)
    Write-Host "`r[$(Get-Date -Format 'HH:mm:ss')] Alive - $([math]::Floor($remaining.TotalHours))h $($remaining.Minutes)m remaining" -NoNewline
    Start-Sleep -Seconds 45
}

# Restore defaults (user can change back later)
[KeepAwakeFallback]::SetThreadExecutionState([KeepAwakeFallback]::ES_CONTINUOUS) | Out-Null
Write-Host ""
Write-Host "Fallback keep-alive ended."

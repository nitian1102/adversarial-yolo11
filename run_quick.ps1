param(
  [string[]]$Deblur = @("wiener"),
  [string[]]$Images = @(),
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ArgsList = @(".\run_experiment.py", "--preset", "quick", "--deblur") + $Deblur + @("--device", $Device)
if ($Images.Count -gt 0) {
  $ArgsList += @("--images") + $Images
}
python @ArgsList

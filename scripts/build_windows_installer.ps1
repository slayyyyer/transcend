[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [Parameter(Mandatory = $true)]
    [string]$Version,

    [ValidateSet("msi", "app-image")]
    [string]$Type = "msi",

    [string]$OutputDir = "",

    [switch]$SkipMaven
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$JavaDir = Join-Path $Root "java"
$UpgradeUuid = "6a3fbc4c-6f8a-4a0e-8f2b-2a9f6f0b3d51"

function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

function Get-RequiredModelFiles {
    return @(
        "model_config.json",
        "generation.json",
        "manifest.json",
        "onnx/encoder.onnx",
        "onnx/decoder_init.onnx",
        "onnx/decoder_step.onnx",
        "tokenizer/tokenizer.json",
        "tokenizer/merges.txt"
    )
}

function Assert-ModelPackage([string]$Directory) {
    foreach ($relative in (Get-RequiredModelFiles)) {
        $path = Join-Path $Directory $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Model package is missing required file: $path"
        }
    }
}

function Get-Executable([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    if ($env:JAVA_HOME) {
        $candidate = Join-Path $env:JAVA_HOME "bin\$Name"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "$Name was not found. Install JDK 21 and put its bin directory on PATH."
}

if ($Version -notmatch "^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$") {
    throw "Version must be a numeric jpackage version such as 0.1.0 (received '$Version')."
}

$modelRoot = (Resolve-ProjectPath $ModelDir).TrimEnd([char[]]@("\", "/"))
Assert-ModelPackage $modelRoot

if (-not $OutputDir) {
    $OutputDir = Join-Path $JavaDir "target\release"
}
$OutputDir = Resolve-ProjectPath $OutputDir

$jpackage = Get-Executable "jpackage.exe"

if ($Type -ne "app-image" -and $env:WIX) {
    foreach ($wixBin in @($env:WIX, (Join-Path $env:WIX "bin"))) {
        if (Test-Path -LiteralPath (Join-Path $wixBin "candle.exe") -PathType Leaf) {
            $env:Path = "$wixBin;$env:Path"
            break
        }
    }
}

if ($Type -ne "app-image") {
    foreach ($tool in @("candle.exe", "light.exe")) {
        $command = Get-Command $tool -ErrorAction SilentlyContinue
        if (-not $command) {
            throw "$tool was not found. Install WiX Toolset 3.x and put its bin directory on PATH."
        }
    }
}

if (-not $SkipMaven) {
    $mvn = Get-Command mvn -ErrorAction SilentlyContinue
    if (-not $mvn) {
        throw "Maven was not found on PATH. Install Maven or pass -SkipMaven after running mvn package."
    }
    & $mvn.Source -B -ntp -f (Join-Path $JavaDir "pom.xml") package "-Dtranscend.model.dir=$modelRoot"
    if ($LASTEXITCODE -ne 0) {
        throw "Maven package failed with exit code $LASTEXITCODE."
    }
}

$stage = Join-Path $JavaDir "target\windows-installer\stage"
$appInput = Join-Path $stage "app"
$appContentModels = Join-Path $stage "app-content\models"
$modelDestination = Join-Path $appContentModels "scaled_export"
$dist = Join-Path $stage "dist"

Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $appInput -Force | Out-Null
New-Item -ItemType Directory -Path $modelDestination -Force | Out-Null
New-Item -ItemType Directory -Path $dist -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$jars = @(
    Get-ChildItem -LiteralPath (Join-Path $JavaDir "target") -Filter "transcend-nmt-*.jar" -File |
        Where-Object { $_.Name -notmatch "-sources|-javadoc" } |
        Sort-Object LastWriteTime -Descending
)
if ($jars.Count -eq 0) {
    throw "No transcend-nmt jar found. Run the Maven package step first."
}
$applicationJar = $jars[0]
Copy-Item -LiteralPath $applicationJar.FullName -Destination (Join-Path $appInput $applicationJar.Name)

$dependencies = @(Get-ChildItem -LiteralPath (Join-Path $JavaDir "target\lib") -Filter "*.jar" -File)
if ($dependencies.Count -eq 0) {
    throw "No runtime dependencies found in java/target/lib."
}
Copy-Item -LiteralPath $dependencies.FullName -Destination $appInput

$bundledBytes = 0L
Get-ChildItem -LiteralPath $modelRoot -Recurse -File |
    Where-Object { $_.Extension -ne ".pt" } |
    ForEach-Object {
        $relative = $_.FullName.Substring($modelRoot.Length).TrimStart([char[]]@("\", "/"))
        $destination = Join-Path $modelDestination $relative
        $parent = Split-Path -Parent $destination
        if ($parent) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $destination
        $bundledBytes += $_.Length
    }
Assert-ModelPackage $modelDestination

$jpackageArgs = @(
    "--type", $Type,
    "--name", "Transcend Translator",
    "--app-version", $Version,
    "--vendor", "Transcend",
    "--description", "Offline English-French translation pilot",
    "--copyright", "Copyright 2026 Transcend",
    "--input", $appInput,
    "--main-jar", $applicationJar.Name,
    "--main-class", "org.transcend.nmt.ui.DesktopLauncher",
    "--dest", $dist,
    "--app-content", $appContentModels
)
if ($Type -ne "app-image") {
    $jpackageArgs += @(
        "--win-menu",
        "--win-menu-group", "Transcend",
        "--win-shortcut",
        "--win-dir-chooser",
        "--win-per-user-install",
        "--win-upgrade-uuid", $UpgradeUuid
    )
}

Write-Output ("PACKAGED_MODEL_BYTES=" + $bundledBytes)
& $jpackage @jpackageArgs
if ($LASTEXITCODE -ne 0) {
    throw "jpackage failed with exit code $LASTEXITCODE."
}

if ($Type -eq "app-image") {
    $appImage = Get-ChildItem -LiteralPath $dist -Directory | Select-Object -First 1
    if (-not $appImage) {
        throw "jpackage did not produce an app image."
    }
    $appImageOutput = Join-Path $OutputDir $appImage.Name
    Remove-Item -LiteralPath $appImageOutput -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $appImage.FullName -Destination $appImageOutput
    Write-Output ("APP_IMAGE=" + $appImageOutput)
    exit 0
}

$builtMsi = Get-ChildItem -LiteralPath $dist -Filter "*.msi" -File | Select-Object -First 1
if (-not $builtMsi) {
    throw "jpackage did not produce an MSI."
}

$stableName = "Transcend-Translator-Setup.msi"
$stablePath = Join-Path $OutputDir $stableName
$checksumPath = "$stablePath.sha256"
Copy-Item -LiteralPath $builtMsi.FullName -Destination $stablePath -Force
$hash = (Get-FileHash -LiteralPath $stablePath -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText($checksumPath, "$hash  $stableName`n")

Write-Output ("INSTALLER=" + $stablePath)
Write-Output ("CHECKSUM=" + $checksumPath)
Write-Output ("SHA256=" + $hash)

[CmdletBinding()]
param(
    [string]$ModelDir = "models/scaled_export",
    [string]$OutputPath = "java/target/Transcend-Model.zip"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

function Assert-ModelPackage([string]$Directory) {
    foreach ($relative in @(
            "model_config.json",
            "generation.json",
            "manifest.json",
            "onnx/encoder.onnx",
            "onnx/decoder_init.onnx",
            "onnx/decoder_step.onnx",
            "tokenizer/tokenizer.json",
            "tokenizer/merges.txt")) {
        $path = Join-Path $Directory $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Model package is missing required file: $path"
        }
    }
}

$modelRoot = (Resolve-ProjectPath $ModelDir).TrimEnd([char[]]@("\", "/"))
$archivePath = Resolve-ProjectPath $OutputPath
Assert-ModelPackage $modelRoot

$stage = Join-Path $Root "java/target/model-archive"
$packageDir = Join-Path $stage "scaled_export"
Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $packageDir -Force | Out-Null

Get-ChildItem -LiteralPath $modelRoot -Recurse -File |
    Where-Object { $_.Extension -ne ".pt" } |
    ForEach-Object {
        $relative = $_.FullName.Substring($modelRoot.Length).TrimStart([char[]]@("\", "/"))
        $destination = Join-Path $packageDir $relative
        $parent = Split-Path -Parent $destination
        if ($parent) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $destination
    }
Assert-ModelPackage $packageDir

$archiveParent = Split-Path -Parent $archivePath
if ($archiveParent) {
    New-Item -ItemType Directory -Path $archiveParent -Force | Out-Null
}
Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
Compress-Archive -LiteralPath $packageDir -DestinationPath $archivePath -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Model archive was not created: $archivePath"
}

$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Output ("MODEL_ARCHIVE=" + $archivePath)
Write-Output ("MODEL_BYTES=" + (Get-Item -LiteralPath $archivePath).Length)
Write-Output ("SHA256=" + $hash)
Write-Output "Attach this archive to the release as Transcend-Model.zip before publishing."

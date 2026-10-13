$ErrorActionPreference = 'Stop'
$datasetRepo = 'nvidia/PhysicalAI-Autonomous-Vehicles'
$datasetRevision = '33f9bf447ed3bcb7d545ce13f4226f824214fafb'
$stagingDirectory = 'E:\Lighten_VLA_data_staging\physicalai_train_20261002'
$archivePath = 'E:\Lighten_VLA_data_staging\physicalai_train_chunk0000_20261002.tar'
$datasetFiles = @(
  'clip_index.parquet',
  'features.csv',
  'metadata/data_collection.parquet',
  'metadata/feature_presence.parquet',
  'calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet',
  'calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet',
  'labels/egomotion/egomotion.chunk_0000.zip',
  'camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_0000.zip',
  'camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_0000.zip',
  'camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_0000.zip',
  'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip'
)
New-Item -ItemType Directory -Force -Path $stagingDirectory | Out-Null
& hf download $datasetRepo @datasetFiles --repo-type dataset --revision $datasetRevision --local-dir $stagingDirectory --max-workers 2
if ($LASTEXITCODE -ne 0) { throw 'Hugging Face download failed. Rerun this script to resume the same revision.' }
$manifestFiles = foreach ($relative in $datasetFiles) {
  $path = Join-Path $stagingDirectory $relative
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $relative" }
  $fileInfo = Get-Item -LiteralPath $path
  [ordered]@{ path=$relative; bytes=$fileInfo.Length; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$manifest = [ordered]@{
  repository=$datasetRepo; revision=$datasetRevision; chunk_id=0
  scope='New data staging only. Agent must verify clip identities, completeness and disjoint splits before training.'
  files=@($manifestFiles)
}
$manifestPath = Join-Path $stagingDirectory 'download_manifest.json'
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
if (Test-Path -LiteralPath $archivePath) { throw "Archive already exists: $archivePath. Data downloaded and verified; do not overwrite it automatically." }
& tar -cf $archivePath -C $stagingDirectory @datasetFiles 'download_manifest.json'
if ($LASTEXITCODE -ne 0) { throw 'Archive creation failed.' }
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(($archivePath + '.sha256'), ($archiveHash + '  ' + [IO.Path]::GetFileName($archivePath) + "`n"), [Text.UTF8Encoding]::new($false))
Write-Host "READY: $archivePath"
Write-Host "SHA256: $archiveHash"
Write-Host 'No authentication token or Hugging Face cache is included in this archive.'

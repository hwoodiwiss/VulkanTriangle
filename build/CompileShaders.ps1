param(
	[string]$ProjectDir = { throw "ProjectDir is required" },
	[string]$OutDir = { throw "OutDir is required" }
)

$Types = ('.frag', '.vert')

$Files = Get-ChildItem "$ProjectDir\shaders\*" | Where-Object { $_.extension -in $Types }
New-Item "$OutDir\shaders" -ItemType Directory

foreach ($file in $Files) {
	$fileName = [System.IO.Path]::GetFilenameWithoutExtension($file)
	Write-Host "Compiling $($file.name)..."
	glslc $file -o "$($OutDir)shaders\$($fileName).spv"
	Write-Host "Compilation complete"
}
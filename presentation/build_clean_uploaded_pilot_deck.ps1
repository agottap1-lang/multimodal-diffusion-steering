$ErrorActionPreference = "Stop"

$sourcePptx = "C:\Users\anude\Downloads\anudeep vlm pilot study for legible robot motion (1).pptx"
$outputPptx = Join-Path $PSScriptRoot "anudeep_vlm_pilot_study_cleaned.pptx"
$tempRoot = Join-Path $env:TEMP ("pptx_edit_" + [guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $tempRoot "deck.zip"
$unpacked = Join-Path $tempRoot "unpacked"

if (-not (Test-Path $sourcePptx)) {
    throw "Source PPTX not found: $sourcePptx"
}

New-Item -ItemType Directory -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Path $unpacked | Out-Null
Copy-Item -LiteralPath $sourcePptx -Destination $zipPath
Expand-Archive -LiteralPath $zipPath -DestinationPath $unpacked -Force

function Set-SlideText {
    param(
        [string]$SlideXmlPath,
        [string[]]$Texts
    )

    [xml]$xml = Get-Content -LiteralPath $SlideXmlPath -Raw
    $ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
    $ns.AddNamespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
    $nodes = $xml.SelectNodes("//a:t", $ns)

    if ($Texts.Count -gt $nodes.Count) {
        throw "Too many replacement strings for $SlideXmlPath. Have $($nodes.Count) nodes, got $($Texts.Count)."
    }

    for ($i = 0; $i -lt $nodes.Count; $i++) {
        if ($i -lt $Texts.Count) {
            $nodes[$i].InnerText = $Texts[$i]
        } else {
            $nodes[$i].InnerText = ""
        }
    }

    $xml.Save($SlideXmlPath)
}

$slidesDir = Join-Path $unpacked "ppt/slides"

$slideText = @{
    "slide1.xml" = @(
        "VLM as a Proxy for Human Judgment of Robot Motion Legibility",
        "Pilot study for legible robot motion",
        "April 2026",
        "Anudeep",
        "Gottapu"
    )
    "slide2.xml" = @(
        "Overview",
        "What this pilot study asks",
        "Study design",
        "Video set",
        "Human study",
        "VLM evaluation",
        "Main result 1",
        "Accuracy comparison",
        "Main result 2",
        "Human-VLM agreement",
        "Main result 3",
        "Time to legibility"
    )
    "slide3.xml" = @(
        "Study design overview",
        "8 videos: 4 legible and 4 ambiguous",
        "6 human participants",
        "Binary goal inference task: Goal A or Goal B",
        "Fair comparison: the VLM was evaluated at the same selected timepoints as the human study"
    )
    "slide4.xml" = @(
        "Video categories",
        "",
        "",
        "Legible",
        "",
        "trajectories",
        ": clear early visual commitment",
        "visual",
        "",
        "cues",
        "le_d_drawer_close",
        ",",
        "le_l_block",
        ",",
        "le_r_block",
        ",",
        "le_t_drawer_close",
        "Ambiguous trajectories",
        ": competing goals until later in the motion",
        "amb_d_drawer_close",
        ",",
        "amb_l_block",
        ",",
        "amb_r_block",
        ",",
        "amb_to_drawer_close"
    )
    "slide5.xml" = @("Drawer-closing examples")
    "slide6.xml" = @("Block-picking examples")
    "slide7.xml" = @("Example: ambiguous versus legible motion")
    "slide8.xml" = @("Example: ambiguous versus legible motion")
    "slide9.xml" = @(
        "Methodology",
        "Human study: participants saw cumulative frame prefixes at selected timepoints",
        "The 4 timepoints were chosen to capture: 1) start of motion, 2) first confident correct inference, 3) a possible flip point, and 4) the end of the clip",
        "VLM study: Gemini 2.5 Flash saw the same cumulative frame prefixes and predicted the goal"
    )
    "slide10.xml" = @("VLM evaluation pipeline")
    "slide11.xml" = @("VLM evaluation pipeline")
    "slide12.xml" = @(
        "Data collection summary",
        "234 human observations (6 participants x 8 videos)",
        "31 matched VLM predictions",
        "Study interface used for the pilot"
    )
    "slide13.xml" = @("Human user study")
    "slide14.xml" = @("Human study examples")
    "slide15.xml" = @()
    "slide16.xml" = @("Human study observations")
    "slide17.xml" = @(
        "Result 1: Accuracy comparison",
        "Human: 69.7% versus VLM: 64.5%",
        "Gap: 5.1 percentage points",
        "Takeaway: the VLM is reasonably close to human accuracy in this pilot"
    )
    "slide18.xml" = @(
        "Accuracy by video",
        "VLM outperforms humans on 3 videos",
        "Largest failure case",
        "amb_r_block",
        "(25%)"
    )
    "slide19.xml" = @(
        "Result 2: Human-VLM agreement",
        "73.1% agreement on goal selections",
        "Interpretation: VLM and humans chose the same option about 73% of the time",
        "Best:",
        "amb_l_block",
        "(95.8%)",
        "Worst:",
        "amb_r_block",
        "(45.8%)"
    )
    "slide20.xml" = @(
        "Summary table",
        "Metric",
        "Human",
        "VLM",
        "Reading",
        "Accuracy",
        "69.7%",
        "64.5%",
        "Small gap",
        "Agreement",
        "-",
        "73.1%",
        "Strong alignment",
        "Timing advantage",
        "-",
        "6/8 videos",
        "VLM earlier on most videos",
        "Legible accuracy",
        "71.9%",
        "66.7%",
        "Close",
        "Ambiguous accuracy",
        "67.5%",
        "62.5%",
        "Close"
    )
    "slide21.xml" = @(
        "Conclusion",
        "Strengths of the VLM approach",
        "Consistent performance with no fatigue",
        "Scalable to many more videos",
        "Only about a 5% accuracy gap in this pilot",
        "About 73% agreement with humans",
        "Current limits",
        "Still below human accuracy",
        "Should not fully replace human judgment yet"
    )
    "slide22.xml" = @(
        "How this pilot study supports the thesis",
        "Component",
        "Why it matters",
        "Human study",
        "Provides the reference point for legibility judgment",
        "Matched VLM evaluation",
        "Tests whether the proxy tracks people on the same evidence",
        "Agreement analysis",
        "Shows where the proxy is reliable and where it fails",
        "Timing analysis",
        "Connects the proxy to early intent inference, which is the core legibility idea",
        "Thesis role",
        "Supports using a VLM as a scalable evaluator for legible robot motion"
    )
}

foreach ($name in $slideText.Keys) {
    $path = Join-Path $slidesDir $name
    Set-SlideText -SlideXmlPath $path -Texts $slideText[$name]
}

if (Test-Path $outputPptx) {
    Remove-Item -LiteralPath $outputPptx -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipOut = Join-Path $tempRoot "rewritten.zip"
$archive = [System.IO.Compression.ZipFile]::Open($zipOut, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $files = Get-ChildItem -LiteralPath $unpacked -Recurse -File
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($unpacked.Length + 1).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $relative) | Out-Null
    }
}
finally {
    $archive.Dispose()
}

Move-Item -LiteralPath $zipOut -Destination $outputPptx
Remove-Item -LiteralPath $tempRoot -Recurse -Force
Write-Output "Created: $outputPptx"

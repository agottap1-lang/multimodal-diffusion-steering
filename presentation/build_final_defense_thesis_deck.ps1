$ErrorActionPreference = "Stop"

$sourcePptx = "C:\Users\anude\Downloads\anudeep vlm pilot study for legible robot motion (1).pptx"
$outputPptx = Join-Path $PSScriptRoot "anudeep_final_defense_thesis_strict.pptx"
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

    $limit = [Math]::Min($Texts.Count, $nodes.Count)
    for ($i = 0; $i -lt $limit; $i++) {
        $nodes[$i].InnerText = $Texts[$i]
    }

    for ($i = $limit; $i -lt $nodes.Count; $i++) {
        $nodes[$i].InnerText = ""
    }

    $xml.Save($SlideXmlPath)
}

function Replace-MediaImage {
    param(
        [string]$MediaRelativePath,
        [string]$SourceImagePath
    )

    if (-not (Test-Path $SourceImagePath)) {
        throw "Replacement image not found: $SourceImagePath"
    }

    $dest = Join-Path $unpacked $MediaRelativePath.Replace("/", "\")
    Copy-Item -LiteralPath $SourceImagePath -Destination $dest -Force
}

$slidesDir = Join-Path $unpacked "ppt/slides"

$slideText = @{
    "slide1.xml" = @(
        "VLM-Guided Diffusion Policies for Legible Robot Motion",
        "Final defense focus: VLM as a proxy for human judgment of legibility",
        "Arizona State University | Approved April 2026 | Nakul Gopalan (Chair), Hani Ben Amor, Kunal Garg",
        "Anudeep Sai",
        "Gottapu"
    )
    "slide2.xml" = @(
        "Thesis roadmap",
        "Abstract",
        "Research questions",
        "Task setting",
        "Thesis pipeline",
        "VLM method",
        "Pilot study",
        "Accuracy",
        "Timing",
        "Agreement",
        "Diffusion placeholders",
        "Conclusion"
    )
    "slide3.xml" = @(
        "Abstract and research questions",
        "Can robot motion be evaluated for legibility without running a full human study every time?",
        "Can a VLM approximate human judgments of robot motion legibility?",
        "Does legibility appear mainly in how early the goal becomes readable?",
        "How does this pilot study support the larger thesis on legible robot motion?"
    )
    "slide4.xml" = @(
        "Stimulus Set",
        "",
        "",
        "Legible",
        "",
        "trajectories",
        ": clearer early goal commitment",
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
        ": intent stays unclear until later",
        "amb_d_drawer_close",
        ",",
        "amb_l_block",
        ",",
        "amb_r_block",
        ",",
        "amb_to_drawer_close"
    )
    "slide5.xml" = @("Task family 1: drawer-closing trajectories")
    "slide6.xml" = @("Task family 2: block-picking trajectories")
    "slide7.xml" = @("Example: legible versus ambiguous motion in the study")
    "slide8.xml" = @("Full thesis pipeline and where this talk focuses")
    "slide9.xml" = @(
        "Diffusion policy methodology placeholder",
        "Environment: TwoBlockPick, 400 demonstrations, and multi-goal robot motion.",
        "Model: DDPM and DDIM action-chunk generation with training-free steering options.",
        "This deck keeps the policy section brief and puts the main emphasis on the VLM and pilot-study evaluation chapter."
    )
    "slide10.xml" = @("VLM evaluation pipeline")
    "slide11.xml" = @("Prompt design, annotated frames, and VLO")
    "slide12.xml" = @(
        "Pilot-study design summary",
        "8 participants, 8 videos, and 312 human observations",
        "31 matched VLM evaluations at the same selected timepoints",
        "Fair comparison: humans and the VLM saw the same evidence at the same moments"
    )
    "slide13.xml" = @("Human study phase A: cumulative frame probes")
    "slide14.xml" = @("Human study phase B: video stop-time judgment")
    "slide15.xml" = @()
    "slide16.xml" = @("Result 1: accuracy by trajectory type")
    "slide17.xml" = @(
        "Result 2: temporal dynamics of goal inference",
        "Human and VLM judgments both improve as more motion is revealed.",
        "The clearest legibility effect appears early, not only at the final frame.",
        "That matches the thesis definition of legibility: earlier correct inference."
    )
    "slide18.xml" = @(
        "Result 3: time to legibility",
        "VLM: 3.25 s on legible videos vs 6.75 s on ambiguous videos",
        "Strongest delay",
        "amb_r_block",
        "(VLM first correct at 14 s)"
    )
    "slide19.xml" = @(
        "Result 4: agreement and outliers",
        "Overall agreement: 74.2%",
        "Best case:",
        "amb_l_block",
        "(96.9%)",
        "Worst case:",
        "amb_r_block",
        "(46.9%)"
    )
    "slide20.xml" = @(
        "Key numbers to remember",
        "Metric",
        "Human",
        "VLM",
        "Read",
        "Overall accuracy",
        "66.7%",
        "64.5%",
        "Comparable overall",
        "Above chance",
        "p < 1e-33",
        "p = 3.75e-4",
        "Both significant",
        "Time to legibility",
        "5.60 / 6.80 s",
        "3.25 / 6.75 s",
        "VLM earlier on legible motion",
        "Best-aligned case",
        "-",
        "amb_l_block",
        "96.9% IoU",
        "Failure case",
        "-",
        "amb_r_block",
        "46.9% IoU"
    )
    "slide21.xml" = @(
        "Diffusion policy results placeholder",
        "Why this slide stays brief",
        "The full thesis includes base-policy performance and guidance comparisons.",
        "What belongs here",
        "Success rate, VLO on generated trajectories, and method comparisons.",
        "Why it is condensed in this deck",
        "This version is centered on the pilot study that validates the VLM evaluator.",
        "If asked",
        "I can walk through the policy repo and those results orally."
    )
    "slide22.xml" = @(
        "Conclusion and questions",
        "Main message",
        "This pilot study supports using a VLM as a practical proxy for human judgment of legible robot motion.",
        "Evidence 1",
        "Human and VLM accuracy are comparable overall.",
        "Evidence 2",
        "Agreement is high, but important failure cases remain.",
        "Evidence 3",
        "Timing gives the clearest signal of legibility.",
        "Thesis implication",
        "The VLM can scale evaluation before larger human studies.",
        "Questions",
        ""
    )
}

foreach ($name in $slideText.Keys) {
    $path = Join-Path $slidesDir $name
    Set-SlideText -SlideXmlPath $path -Texts $slideText[$name]
}

Replace-MediaImage -MediaRelativePath "ppt/media/image5.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick\figures\fig7_system_overview.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image6.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\vlm_flow_simplified.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image7.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\pipeline_diagram.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image8.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\vlm_pipeline_with_frames.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image11.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig8_summary_dashboard.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image12.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig1_accuracy_by_type.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image13.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig2_temporal_dynamics.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image14.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig4_time_to_legibility.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image15.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\figure3_video_iou.png"

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

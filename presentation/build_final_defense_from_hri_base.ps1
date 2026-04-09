$ErrorActionPreference = "Stop"

$sourcePptx = "C:\Users\anude\Downloads\HRI_Legibility_Thesis_Analysis.pptx"
$outputPptx = Join-Path $PSScriptRoot "Anudeep_Thesis_Final_Defense_Rebuilt.pptx"
$tempRoot = Join-Path $env:TEMP ("pptx_hri_rebuild_" + [guid]::NewGuid().ToString("N"))
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

function New-PlaceholderImage {
    param(
        [string]$Path,
        [string]$Title,
        [string]$Subtitle
    )

    Add-Type -AssemblyName System.Drawing
    $bmp = New-Object System.Drawing.Bitmap 1600, 900
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $bg = [System.Drawing.Color]::FromArgb(250,250,250)
        $maroon = [System.Drawing.Color]::FromArgb(140,29,64)
        $gold = [System.Drawing.Color]::FromArgb(255,198,39)
        $gray = [System.Drawing.Color]::FromArgb(90,90,90)
        $light = [System.Drawing.Color]::FromArgb(237,237,237)
        $g.Clear($bg)
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $pen = New-Object System.Drawing.Pen($maroon, 6)
        $g.DrawRectangle($pen, 90, 110, 1420, 680)
        $brushMaroon = New-Object System.Drawing.SolidBrush($maroon)
        $brushGold = New-Object System.Drawing.SolidBrush($gold)
        $brushGray = New-Object System.Drawing.SolidBrush($gray)
        $brushLight = New-Object System.Drawing.SolidBrush($light)
        $fontTitle = New-Object System.Drawing.Font("Aptos", 38, [System.Drawing.FontStyle]::Bold)
        $fontSub = New-Object System.Drawing.Font("Aptos", 22, [System.Drawing.FontStyle]::Regular)
        $fontBody = New-Object System.Drawing.Font("Aptos", 24, [System.Drawing.FontStyle]::Regular)
        $g.FillRectangle($brushLight, 120, 220, 1360, 460)
        $g.FillRectangle($brushGold, 120, 690, 1360, 10)
        $sfCenter = New-Object System.Drawing.StringFormat
        $sfCenter.Alignment = [System.Drawing.StringAlignment]::Center
        $sfCenter.LineAlignment = [System.Drawing.StringAlignment]::Center
        $g.DrawString($Title, $fontTitle, $brushMaroon, (New-Object System.Drawing.RectangleF(140, 145, 1320, 90)), $sfCenter)
        $g.DrawString($Subtitle, $fontSub, $brushGray, (New-Object System.Drawing.RectangleF(170, 310, 1260, 220)), $sfCenter)
        $g.DrawString("Reserved in the defense deck to match the thesis structure", $fontBody, $brushGray, (New-Object System.Drawing.RectangleF(170, 560, 1260, 60)), $sfCenter)
        $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $g.Dispose()
        $bmp.Dispose()
    }
}

$slidesDir = Join-Path $unpacked "ppt/slides"

$slideText = @{
    "slide1.xml" = @(
        "Vision-Language Models as Proxies for Human Judgment of Robot Motion Legibility",
        "MS Thesis Final Defense",
        "Anudeep Sai Gottapu",
        "Arizona State University | April 2026",
        "Committee: Nakul Gopalan (Chair) | Hani Ben Amor | Kunal Garg"
    )
    "slide2.xml" = @(
        "Abstract",
        "Thesis abstract in presentation form",
        "This thesis asks whether a vision-language model can serve as a practical proxy for human judgment of robot motion legibility.",
        "I compare a VLM and human observers on robot-motion videos using matched evidence, matched timepoints, and timing-based legibility measures.",
        "Overall performance is comparable, while the clearest signal of legibility appears in how early the observer can infer the goal.",
        "The study supports VLMs as scalable evaluators of legibility, while still requiring human validation for difficult cases."
    )
    "slide3.xml" = @(
        "Thesis Structure",
        "Following the thesis document",
        "1. Introduction",
        "2. Related Work",
        "3. Technical Background",
        "4. System Design",
        "5. Experiments and Results",
        "6. Discussion and Honest Assessment",
        "7. Conclusion and Future Work",
        "Defense emphasis today: system design, pilot study, VLM-as-proxy results, and thesis implications."
    )
    "slide4.xml" = @(
        "Research Questions",
        "Three questions drive this defense",
        "RQ1",
        "Can a VLM approximate human judgments of robot motion legibility?",
        "RQ2",
        "Does legibility show up more clearly in timing than in final accuracy?",
        "RQ3",
        "Where do human and VLM judgments align, and where do they diverge?",
        "Slides 9–12 answer these directly with results."
    )
    "slide5.xml" = @(
        "System Design",
        "How the evaluation chapter fits the full thesis",
        "Robot motion videos are processed into matched observations for both humans and the VLM.",
        "Evaluation focuses on accuracy, time-to-legibility, and agreement.",
        "This is the chapter that connects the pilot study to the larger thesis."
    )
    "slide6.xml" = @(
        "Pilot Study Design",
        "Human study and VLM protocol",
        "8 videos: 4 legible and 4 ambiguous",
        "8 participants",
        "Phase A: cumulative frame probes",
        "Phase B: full-video stop-time judgment",
        "31 matched VLM evaluations at the same study timepoints"
    )
    "slide7.xml" = @(
        "Diffusion Policy Methodology",
        "Reserved chapter in the defense",
        "This slide intentionally leaves space for the policy-generation methodology.",
        "It corresponds to the system-design material in the thesis."
    )
    "slide8.xml" = @(
        "Diffusion Policy Results",
        "Reserved chapter in the defense",
        "This slide intentionally leaves space for policy results.",
        "The current defense draft keeps the main emphasis on the pilot study and VLM-as-proxy analysis."
    )
    "slide9.xml" = @(
        "RQ1: Can a VLM approximate human judgment?",
        "Overall accuracy comparison",
        "Human accuracy: 66.7%",
        "VLM accuracy: 64.5%",
        "Overall difference is small and not statistically significant.",
        "Both are significantly above chance."
    )
    "slide10.xml" = @(
        "RQ2: Does legibility appear mainly in timing?",
        "Temporal dynamics of inference",
        "The main separation between legible and ambiguous motion happens early.",
        "Final accuracy converges, but early inference differs.",
        "That is exactly the thesis definition of legibility."
    )
    "slide11.xml" = @(
        "RQ2: Time to Legibility",
        "Timing is the clearest result",
        "Human TTL: 6.45 s on legible vs 7.90 s on ambiguous",
        "VLM TTL: 3.25 s on legible vs 6.75 s on ambiguous",
        "The VLM is earlier on 6 of 8 videos."
    )
    "slide12.xml" = @(
        "RQ3: Agreement and Divergence",
        "Where human and VLM judgments match, and where they fail",
        "Overall agreement: 73.8%",
        "Best agreement: amb_l_block",
        "Strong failure case: amb_r_block",
        "This is why the VLM is useful, but not a replacement for human judgment."
    )
    "slide13.xml" = @(
        "Answers to the Research Questions",
        "RQ1",
        "Yes. In this pilot, VLM and human accuracy are comparable overall.",
        "RQ2",
        "Yes. Timing is a stronger legibility signal than final accuracy.",
        "RQ3",
        "Agreement is high overall, but there are meaningful failure cases and participant differences."
    )
    "slide14.xml" = @(
        "Discussion and Honest Assessment",
        "What this study supports",
        "The VLM is a practical legibility evaluator for rapid analysis and screening.",
        "What this study does not support",
        "The VLM does not fully replace human studies, especially on harder trajectories.",
        "Most important lesson",
        "The variation across videos is part of the scientific contribution, not a weakness."
    )
    "slide15.xml" = @(
        "Conclusion and Future Work",
        "Main conclusion",
        "Vision-language models can act as useful proxies for human judgment of robot motion legibility.",
        "Next steps",
        "Larger human studies, more trajectories, stronger temporal VLM evaluation, and integration with the full policy pipeline."
    )
    "slide16.xml" = @(
        "Thank You",
        "Questions?",
        "Anudeep Sai Gottapu",
        "Arizona State University"
    )
}

foreach ($name in $slideText.Keys) {
    $path = Join-Path $slidesDir $name
    Set-SlideText -SlideXmlPath $path -Texts $slideText[$name]
}

$phMethod = Join-Path $tempRoot "diffusion_method_placeholder.png"
$phResults = Join-Path $tempRoot "diffusion_results_placeholder.png"
New-PlaceholderImage -Path $phMethod -Title "Diffusion Policy Methodology" -Subtitle "Reserved placeholder slide in the final defense deck"
New-PlaceholderImage -Path $phResults -Title "Diffusion Policy Results" -Subtitle "Reserved placeholder slide in the final defense deck"

Replace-MediaImage -MediaRelativePath "ppt/media/image1.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\pipeline_diagram.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image2.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\vlm_pipeline_with_frames.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image3.png" -SourceImagePath $phMethod
Replace-MediaImage -MediaRelativePath "ppt/media/image4.png" -SourceImagePath $phResults
Replace-MediaImage -MediaRelativePath "ppt/media/image5.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\figure1_main_comparison.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image6.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig2_temporal_dynamics.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image7.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig4_time_to_legibility.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image8.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\figure3_video_iou.png"

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

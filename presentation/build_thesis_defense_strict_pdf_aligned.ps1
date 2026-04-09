$ErrorActionPreference = "Stop"

$sourcePptx = "C:\Users\anude\Downloads\HRI_Legibility_Thesis_Analysis.pptx"
$outputPptx = Join-Path $PSScriptRoot "Anudeep_Thesis_Defense_PDF_Aligned.pptx"
$tempRoot = Join-Path $env:TEMP ("pptx_pdf_aligned_" + [guid]::NewGuid().ToString("N"))
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
        $bg = [System.Drawing.Color]::FromArgb(249, 248, 245)
        $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
        $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
        $gray = [System.Drawing.Color]::FromArgb(88, 88, 88)
        $light = [System.Drawing.Color]::FromArgb(236, 233, 229)
        $g.Clear($bg)
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($light)), 110, 120, 1380, 640)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 6)), 110, 120, 1380, 640)
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 110, 730, 1380, 18)

        $titleFont = New-Object System.Drawing.Font("Arial", 34, [System.Drawing.FontStyle]::Bold)
        $subFont = New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Regular)
        $bodyFont = New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Regular)
        $sf = New-Object System.Drawing.StringFormat
        $sf.Alignment = [System.Drawing.StringAlignment]::Center
        $sf.LineAlignment = [System.Drawing.StringAlignment]::Center

        $g.DrawString($Title, $titleFont, (New-Object System.Drawing.SolidBrush($maroon)), ([System.Drawing.RectangleF]::new(160, 175, 1280, 70)), $sf)
        $g.DrawString($Subtitle, $subFont, (New-Object System.Drawing.SolidBrush($gray)), ([System.Drawing.RectangleF]::new(180, 325, 1240, 120)), $sf)
        $g.DrawString("Reserved to preserve the thesis structure while keeping the defense emphasis on the pilot study and VLM-as-proxy analysis.", $bodyFont, (New-Object System.Drawing.SolidBrush($gray)), ([System.Drawing.RectangleF]::new(190, 525, 1220, 90)), $sf)

        $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $g.Dispose()
        $bmp.Dispose()
    }
}

function New-TaskOverviewMontage {
    param(
        [string]$Path
    )

    Add-Type -AssemblyName System.Drawing

    $items = @(
        @{ Label = "Ambiguous | Bottom drawer"; Type = "Ambiguous"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\amb_d_drawer_cclose\t_005.png" },
        @{ Label = "Ambiguous | Left block"; Type = "Ambiguous"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\amb_l_block\t_006.png" },
        @{ Label = "Ambiguous | Right block"; Type = "Ambiguous"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\amb_r_block\t_007.png" },
        @{ Label = "Ambiguous | Top drawer"; Type = "Ambiguous"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\amb_to_drawer_close\t_010.png" },
        @{ Label = "Legible | Bottom drawer"; Type = "Legible"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\le_d_drawer_close\t_003.png" },
        @{ Label = "Legible | Left block"; Type = "Legible"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\le_l_block\t_004.png" },
        @{ Label = "Legible | Right block"; Type = "Legible"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\le_r_block\t_004.png" },
        @{ Label = "Legible | Top drawer"; Type = "Legible"; Image = "C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\stimuli\frames\le_t_drawer_close\t_005.png" }
    )

    $bmp = New-Object System.Drawing.Bitmap 1800, 1100
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $bg = [System.Drawing.Color]::FromArgb(250, 248, 245)
        $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
        $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
        $blue = [System.Drawing.Color]::FromArgb(72, 116, 161)
        $terracotta = [System.Drawing.Color]::FromArgb(171, 94, 74)
        $gray = [System.Drawing.Color]::FromArgb(70, 70, 70)
        $white = [System.Drawing.Color]::White
        $g.Clear($bg)
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 95)
        $g.DrawString("Task Setup and Video Scenarios", (New-Object System.Drawing.Font("Arial", 28, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 50, 24)
        $g.DrawString("Eight robot-manipulation videos used across the pilot study and matched Gemini evaluation", (New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($white)), 52, 60)

        $tileW = 390
        $tileH = 215
        $left = 50
        $top = 150
        $gapX = 35
        $gapY = 65

        for ($i = 0; $i -lt $items.Count; $i++) {
            $col = $i % 4
            $row = [Math]::Floor($i / 4)
            $x = $left + $col * ($tileW + $gapX)
            $y = $top + $row * ($tileH + $gapY)

            $accent = if ($items[$i].Type -eq "Legible") { $blue } else { $terracotta }
            $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), $x, $y, $tileW, $tileH)
            $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $x, $y, $tileW, $tileH)
            $g.FillRectangle((New-Object System.Drawing.SolidBrush($accent)), $x, $y, $tileW, 24)

            if (Test-Path $items[$i].Image) {
                $img = [System.Drawing.Image]::FromFile($items[$i].Image)
                try {
                    $destRect = [System.Drawing.Rectangle]::new(($x + 8), ($y + 32), ($tileW - 16), ($tileH - 68))
                    $g.DrawImage($img, $destRect)
                }
                finally {
                    $img.Dispose()
                }
            }

            $g.DrawString($items[$i].Label, (New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($gray)), ([System.Drawing.RectangleF]::new(($x + 8), ($y + $tileH - 34), ($tileW - 16), 26)))
        }

        $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 50, 1015, 1700, 18)
        $g.DrawString("Ambiguous examples keep multiple goals plausible for longer; legible examples communicate intent earlier.", (New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 52, 1040)

        $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $g.Dispose()
        $bmp.Dispose()
    }
}

function New-PilotFlowImage {
    param(
        [string]$Path
    )

    Add-Type -AssemblyName System.Drawing
    $bmp = New-Object System.Drawing.Bitmap 1800, 1000
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $bg = [System.Drawing.Color]::FromArgb(250, 248, 245)
        $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
        $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
        $gray = [System.Drawing.Color]::FromArgb(72, 72, 72)
        $blue = [System.Drawing.Color]::FromArgb(72, 116, 161)
        $terracotta = [System.Drawing.Color]::FromArgb(171, 94, 74)
        $white = [System.Drawing.Color]::White

        $g.Clear($bg)
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality

        $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
        $g.DrawString("Matched Human-VLM Pilot Study Design", (New-Object System.Drawing.Font("Arial", 28, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 55, 24)

        $boxes = @(
            @{ X = 80;  Y = 200; W = 260; H = 150; Fill = $blue;       Title = "8 videos"; Body = "4 legible\n4 ambiguous" },
            @{ X = 400; Y = 200; W = 260; H = 150; Fill = $terracotta; Title = "8 participants"; Body = "Within-study\npilot sample" },
            @{ X = 720; Y = 200; W = 300; H = 150; Fill = $blue;       Title = "Phase A"; Body = "Cumulative frame probes\nmatched timepoints" },
            @{ X = 1080;Y = 200; W = 300; H = 150; Fill = $terracotta; Title = "Phase B"; Body = "Full-video stop-time\njudgment" },
            @{ X = 1440;Y = 200; W = 260; H = 150; Fill = $blue;       Title = "Gemini 2.5 Flash"; Body = "31 matched\nVLM predictions" }
        )

        foreach ($box in $boxes) {
            $g.FillRectangle((New-Object System.Drawing.SolidBrush($box.Fill)), $box.X, $box.Y, $box.W, $box.H)
            $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $box.X, $box.Y, $box.W, $box.H)
            $g.DrawString($box.Title, (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($box.X + 18), ($box.Y + 18), ($box.W - 36), 32)))
            $g.DrawString($box.Body, (New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($box.X + 18), ($box.Y + 62), ($box.W - 36), 68)))
        }

        for ($i = 0; $i -lt ($boxes.Count - 1); $i++) {
            $x1 = $boxes[$i].X + $boxes[$i].W
            $x2 = $boxes[$i + 1].X
            $midY = $boxes[$i].Y + 75
            $pen = New-Object System.Drawing.Pen($maroon, 5)
            $g.DrawLine($pen, $x1 + 10, $midY, $x2 - 18, $midY)
            $tipX = [int]($x2 - 18)
            $tailX = [int]($x2 - 42)
            $midYInt = [int]$midY
            $points = [System.Drawing.Point[]]@(
                [System.Drawing.Point]::new($tipX, $midYInt),
                [System.Drawing.Point]::new($tailX, ($midYInt - 12)),
                [System.Drawing.Point]::new($tailX, ($midYInt + 12))
            )
            $g.FillPolygon((New-Object System.Drawing.SolidBrush($maroon)), $points)
        }

        $panelX = 180
        $panelY = 500
        $panelW = 1440
        $panelH = 250
        $g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)), $panelX, $panelY, $panelW, $panelH)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $panelX, $panelY, $panelW, $panelH)
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), $panelX, $panelY, $panelW, 18)

        $g.DrawString("Outputs aligned across human and Gemini evaluation", (New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($maroon)), 220, 545)
        $g.DrawString("312 human observations", (New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($gray)), 240, 620)
        $g.DrawString("31 matched VLM predictions", (New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($gray)), 720, 620)
        $g.DrawString("Metrics: accuracy, IoU, time-to-legibility", (New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($gray)), 1140, 620)
        $g.DrawString("Fairness rule: the VLM is queried only at the same selected timepoints shown to human participants.", (New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 240, 685)

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
        "Thesis Defense | Arizona State University | April 2026",
        "Anudeep Sai Gottapu | Master of Science in Technology",
        "Chair: Nakul Gopalan | Hani Ben Amor | Kunal Garg"
    )
    "slide2.xml" = @(
        "Abstract",
        "From the thesis document",
        "Legibility in robot motion refers to how easily an observer infers a robot's intent from its movement.",
        "This thesis examines whether a vision-language model can serve as a proxy for human judgment in evaluating robot motion legibility.",
        "Gemini 2.5 Flash analyzes cumulative video frames of robot manipulation tasks and infers the intended goal.",
        "Human pilot data and Gemini predictions are compared at matched timepoints using accuracy, IoU, and time-to-legibility.",
        "Results suggest VLMs capture meaningful aspects of human goal inference from partial robot motion.",
        "Humans remain more reliable overall, so the VLM is best used as a scalable evaluation tool rather than a full replacement."
    )
    "slide3.xml" = @(
        "Thesis Structure",
        "Directly aligned to the PDF table of contents",
        "Chapter 1",
        "Introduction",
        "Chapter 2",
        "Background and Related Work",
        "Chapter 3",
        "Methodology",
        "Overall Thesis Pipeline",
        "Task Setup and Video Scenarios",
        "Pilot Study Design",
        "VLM Goal-Inference Evaluation Pipeline",
        "Prefix-Frame Prompting Strategy",
        "Metrics: Accuracy, IoU, Time-to-Legibility",
        "Diffusion-Policy Steering Framework",
        "Chapter 4 | Results and Discussion",
        "Chapter 5 | Conclusion and Future Work"
    )
    "slide4.xml" = @(
        "Research Questions",
        "Exact questions from Chapter 1.3",
        "RQ1",
        "Can a multimodal vision-language model predict similarly to human judgment from partial robot motion?",
        "RQ2",
        "To what extent do Gemini model predictions align with human judgments in evaluating legibility of robot motion?",
        "RQ3",
        "Can multimodal vision-language models provide a practical signal for scalable legibility evaluation in robot manipulation tasks?",
        "How this defense answers them",
        "RQ1 through overall human-versus-Gemini comparison",
        "RQ2 through agreement, failure cases, and time-to-legibility",
        "RQ3 through the matched evaluation pipeline and the scalability argument"
    )
    "slide5.xml" = @(
        "Introduction: Motivation and Problem Statement",
        "Why this thesis is needed",
        "Humans infer robot intent from partial motion, so legibility matters in shared workspaces.",
        "Current legibility evaluation still depends heavily on human-subject studies.",
        "That makes evaluation costly, time-consuming, and difficult to scale across many trajectories.",
        "This thesis studies whether Gemini 2.5 Flash can serve as a practical proxy for human judgment of legibility."
    )
    "slide6.xml" = @(
        "Background and Related Work",
        "Thesis gap and overall pipeline",
        "Legibility is an observer-centered property of motion, not only a property of task success.",
        "Human goal inference in HRI motivates matched comparisons between people and models.",
        "Vision-language models offer a scalable way to analyze cumulative visual evidence from robot motion.",
        "This thesis brings those ideas into one evaluation pipeline and keeps diffusion-policy steering as an exploratory extension."
    )
    "slide7.xml" = @(
        "Methodology: Pilot Study Design",
        "Human study setup",
        "8 participants",
        "8 videos",
        "Two evaluation modes",
        "Phase A: cumulative frame probes",
        "Phase B: full-video stop-time judgment",
        "Recorded outputs",
        "Goal choice, confidence, cue text, and stop-time judgment",
        "Fair comparison",
        "Gemini was evaluated only at the same selected timepoints shown to human participants"
    )
    "slide8.xml" = @(
        "Methodology: VLM Goal-Inference Evaluation Pipeline",
        "Matched model evaluation",
        "Gemini 2.5 Flash receives the same cumulative prefix frames used in the human study.",
        "At each selected timepoint, the model predicts the intended goal from matched visual evidence.",
        "Prompting uses cumulative frames, explicit candidate goals, and structured JSON output.",
        "Primary metrics",
        "Accuracy",
        "Intersection-over-Union",
        "Time-to-Legibility",
        "These metrics let us compare human and Gemini judgments at matched stages of the same motion."
    )
    "slide9.xml" = @(
        "Methodology: Diffusion-Policy Steering Framework",
        "Reserved in the thesis structure",
        "Section 3.7 is intentionally left open in this defense deck.",
        "The core emphasis today is the pilot study and the Gemini-as-proxy evaluation.",
        "This preserves space for diffusion-policy methodology without centering the talk on it."
    )
    "slide10.xml" = @(
        "Results: Pilot Study Results",
        "Chapter 4.1",
        "312 human observations | 8 participants | 8 videos",
        "Human accuracy: 66.7% | above 33.3% chance (p < 0.001)",
        "This pilot establishes the human baseline for the Gemini comparison."
    )
    "slide11.xml" = @(
        "Results: Gemini vs Human Comparison",
        "RQ1",
        "Human accuracy: 66.7%",
        "Gemini accuracy: 64.5%",
        "Difference is not statistically significant: p = 0.966",
        "Both human and Gemini performance are significantly above chance.",
        "Time-to-legibility also separates legible from ambiguous motion.",
        "Gemini TTL: 3.25 s legible vs 6.75 s ambiguous",
        "Human TTL: 6.45 s legible vs 7.90 s ambiguous"
    )
    "slide12.xml" = @(
        "Results: Agreement and Failure Cases",
        "RQ2",
        "Overall human-Gemini agreement is 74.2% IoU. Alignment is strongest in several left-block examples, while amb_r_block is the main failure case. The pattern supports Gemini as a useful proxy with known limits."
    )
    "slide13.xml" = @(
        "Answers to the Research Questions",
        "Directly aligned to Chapter 1.3",
        "RQ",
        "Question",
        "Main evidence",
        "Answer",
        "RQ1",
        "Can a multimodal VLM predict similarly to human judgment from partial robot motion?",
        "Human 66.7%, Gemini 64.5%, p = 0.966",
        "Yes, overall performance is comparable in this pilot.",
        "RQ2",
        "To what extent do Gemini predictions align with human judgments in evaluating legibility?",
        "74.2% IoU overall; amb_r_block is the key outlier",
        "Alignment is strong overall, but not uniform across videos.",
        "RQ3",
        "Can VLMs provide a practical signal for scalable legibility evaluation?",
        "Matched pipeline, early TTL signal, no participant fatigue",
        "Yes, as a scalable proxy, but not a full replacement for human studies."
    )
    "slide14.xml" = @(
        "Exploratory Diffusion Results and Limitations",
        "What is intentionally left open and what this chapter still shows",
        "Section 4.4",
        "Exploratory VLM-Based Steering Results",
        "Reserved placeholder for diffusion-policy results in the thesis structure",
        "Section 4.5",
        "Limitations and Caveats",
        "Pilot study sample size is small.",
        "Per-video tests have limited power.",
        "Gemini captures meaningful patterns but still misses hard cases.",
        "The VLM should support human evaluation, not replace it."
    )
    "slide15.xml" = @(
        "Discussion",
        "What the thesis supports",
        "Legibility is most visible in when the goal becomes inferable, not only in final accuracy.",
        "Gemini tracks human judgments closely enough to be useful as a scalable evaluator.",
        "Agreement and failure-case analysis reveal where proxy evaluation is trustworthy and where human oversight still matters.",
        "Contribution revisited",
        "The thesis connects a human pilot study, a matched Gemini pipeline, and an exploratory path toward VLM-guided trajectory selection."
    )
    "slide16.xml" = @(
        "Conclusion and Future Work",
        "Summary of findings",
        "Vision-language models can act as useful proxies for human judgment of robot motion legibility.",
        "The strongest legibility signal in this thesis is time-to-legibility under matched observation.",
        "Future work",
        "Larger human studies, more trajectories, stronger temporal prompting, and fuller integration with diffusion-policy steering.",
        "Thank you"
    )
}

foreach ($name in $slideText.Keys) {
    $path = Join-Path $slidesDir $name
    Set-SlideText -SlideXmlPath $path -Texts $slideText[$name]
}

$taskMontage = Join-Path $tempRoot "task_overview_montage.png"
$pilotFlow = Join-Path $tempRoot "pilot_study_flow.png"
$phMethod = Join-Path $tempRoot "diffusion_method_placeholder.png"

New-TaskOverviewMontage -Path $taskMontage
New-PilotFlowImage -Path $pilotFlow
New-PlaceholderImage -Path $phMethod -Title "Diffusion-Policy Steering Framework" -Subtitle "Reserved methodology placeholder in the thesis-aligned defense deck"

Replace-MediaImage -MediaRelativePath "ppt/media/image1.png" -SourceImagePath $taskMontage
Replace-MediaImage -MediaRelativePath "ppt/media/image2.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\pipeline_diagram.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image3.png" -SourceImagePath $pilotFlow
Replace-MediaImage -MediaRelativePath "ppt/media/image4.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\vlm_pipeline_with_frames.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image5.png" -SourceImagePath $phMethod
Replace-MediaImage -MediaRelativePath "ppt/media/image6.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\figure1_main_comparison.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image7.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig4_time_to_legibility.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image8.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig5_iou_agreement.png"

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

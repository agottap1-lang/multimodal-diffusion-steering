$ErrorActionPreference = "Stop"

$sourcePptx = "C:\Users\anude\Downloads\HRI_Legibility_Thesis_Analysis.pptx"
$outputPptx = Join-Path $PSScriptRoot "Anudeep_Thesis_Defense_Final_Strict.pptx"
$tempRoot = Join-Path $env:TEMP ("pptx_hri_strict_v2_" + [guid]::NewGuid().ToString("N"))
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
    param([string]$SlideXmlPath,[string[]]$Texts)
    [xml]$xml = Get-Content -LiteralPath $SlideXmlPath -Raw
    $ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
    $ns.AddNamespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
    $nodes = $xml.SelectNodes("//a:t", $ns)
    $limit = [Math]::Min($Texts.Count, $nodes.Count)
    for ($i = 0; $i -lt $limit; $i++) { $nodes[$i].InnerText = $Texts[$i] }
    for ($i = $limit; $i -lt $nodes.Count; $i++) { $nodes[$i].InnerText = "" }
    $xml.Save($SlideXmlPath)
}

function Replace-MediaImage {
    param([string]$MediaRelativePath,[string]$SourceImagePath)
    if (-not (Test-Path $SourceImagePath)) { throw "Replacement image not found: $SourceImagePath" }
    $dest = Join-Path $unpacked $MediaRelativePath.Replace("/", "\")
    Copy-Item -LiteralPath $SourceImagePath -Destination $dest -Force
}

function New-Canvas {
    param([int]$Width,[int]$Height)
    Add-Type -AssemblyName System.Drawing
    $bmp = New-Object System.Drawing.Bitmap $Width, $Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    return @{ Bitmap = $bmp; Graphics = $g }
}

function Save-And-CloseCanvas {
    param([hashtable]$Canvas,[string]$Path)
    try { $Canvas.Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png) }
    finally { $Canvas.Graphics.Dispose(); $Canvas.Bitmap.Dispose() }
}

function New-ResearchQuestionsGraphic {
    param([string]$Path)
    $c = New-Canvas -Width 1800 -Height 1000
    $g = $c.Graphics
    $bg = [System.Drawing.Color]::FromArgb(249, 247, 243)
    $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
    $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
    $blue = [System.Drawing.Color]::FromArgb(73, 113, 153)
    $green = [System.Drawing.Color]::FromArgb(83, 130, 109)
    $terracotta = [System.Drawing.Color]::FromArgb(176, 100, 78)
    $white = [System.Drawing.Color]::White
    $gray = [System.Drawing.Color]::FromArgb(70, 70, 70)
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
    $g.DrawString("Three Research Questions from the Thesis Document", (New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 48, 24)
    $cards = @(
        @{ X = 90;  Y = 180; W = 500; H = 580; Color = $blue;       Num = "RQ1"; Text = "Can a multimodal vision-language model predict similarly to human judgment from partial robot motion?" },
        @{ X = 650; Y = 180; W = 500; H = 580; Color = $terracotta; Num = "RQ2"; Text = "To what extent do Gemini model predictions align with human judgments in evaluating legibility of robot motion?" },
        @{ X = 1210;Y = 180; W = 500; H = 580; Color = $green;      Num = "RQ3"; Text = "Can multimodal vision-language models provide a practical signal for scalable legibility evaluation in robot manipulation tasks?" }
    )
    foreach ($card in $cards) {
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), $card.X, $card.Y, $card.W, $card.H)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 4)), $card.X, $card.Y, $card.W, $card.H)
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($card.Color)), $card.X, $card.Y, $card.W, 86)
        $g.DrawString($card.Num, (New-Object System.Drawing.Font("Arial", 28, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), ($card.X + 24), ($card.Y + 22))
        $g.DrawString($card.Text, (New-Object System.Drawing.Font("Arial", 24, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), ([System.Drawing.RectangleF]::new(($card.X + 28), ($card.Y + 120), ($card.W - 56), 390)))
    }
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 90, 835, 1620, 18)
    $g.DrawString("Slides 11-13 answer these questions directly. Diffusion-policy methodology and results are kept as reserved placeholders later in the deck.", (New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 95, 875)
    Save-And-CloseCanvas -Canvas $c -Path $Path
}

function New-OverallPipelineGraphic {
    param([string]$Path)
    $c = New-Canvas -Width 1800 -Height 1000
    $g = $c.Graphics
    $bg = [System.Drawing.Color]::FromArgb(249, 247, 243)
    $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
    $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
    $blue = [System.Drawing.Color]::FromArgb(73, 113, 153)
    $green = [System.Drawing.Color]::FromArgb(83, 130, 109)
    $terracotta = [System.Drawing.Color]::FromArgb(176, 100, 78)
    $slate = [System.Drawing.Color]::FromArgb(86, 95, 110)
    $white = [System.Drawing.Color]::White
    $gray = [System.Drawing.Color]::FromArgb(70, 70, 70)
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
    $g.DrawString("Thesis Layout and Overall Evaluation Pipeline", (New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 48, 24)
    $chapterBoxes = @(
        @{ X = 80;  T = "Chapter 1"; B = "Introduction" },
        @{ X = 390; T = "Chapter 2"; B = "Background" },
        @{ X = 700; T = "Chapter 3"; B = "Methodology" },
        @{ X = 1010;T = "Chapter 4"; B = "Results and Discussion" },
        @{ X = 1320;T = "Chapter 5"; B = "Conclusion and Future Work" }
    )
    foreach ($box in $chapterBoxes) {
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), $box.X, 130, 260, 90)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $box.X, 130, 260, 90)
        $g.DrawString($box.T, (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($maroon)), ($box.X + 24), 152)
        $g.DrawString($box.B, (New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), ($box.X + 24), 184)
    }
    $pipe = @(
        @{ X = 100;  Y = 360; W = 260; H = 160; C = $blue;       T = "Robot motion tasks"; B = "Legible and ambiguous manipulation examples" },
        @{ X = 430;  Y = 360; W = 260; H = 160; C = $terracotta; T = "Stimulus generation"; B = "Videos and cumulative prefix frames" },
        @{ X = 760;  Y = 280; W = 260; H = 160; C = $green;      T = "Human pilot study"; B = "Goal choice, confidence, stop-time" },
        @{ X = 760;  Y = 530; W = 260; H = 160; C = $slate;      T = "Gemini evaluation"; B = "Matched timepoints and structured outputs" },
        @{ X = 1090; Y = 360; W = 260; H = 160; C = $blue;       T = "Comparison metrics"; B = "Accuracy, IoU, time-to-legibility" },
        @{ X = 1420; Y = 360; W = 260; H = 160; C = $terracotta; T = "Proxy insight"; B = "Does the VLM track human legibility judgment?" }
    )
    foreach ($p in $pipe) {
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($p.C)), $p.X, $p.Y, $p.W, $p.H)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $p.X, $p.Y, $p.W, $p.H)
        $g.DrawString($p.T, (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($p.X + 16), ($p.Y + 18), ($p.W - 32), 40)))
        $g.DrawString($p.B, (New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($p.X + 16), ($p.Y + 70), ($p.W - 32), 64)))
    }
    $lines = @(
        @{ X1 = 360; Y1 = 440; X2 = 430; Y2 = 440 },
        @{ X1 = 690; Y1 = 440; X2 = 760; Y2 = 360 },
        @{ X1 = 690; Y1 = 440; X2 = 760; Y2 = 610 },
        @{ X1 = 1020;Y1 = 360; X2 = 1090;Y2 = 440 },
        @{ X1 = 1020;Y1 = 610; X2 = 1090;Y2 = 440 },
        @{ X1 = 1350;Y1 = 440; X2 = 1420;Y2 = 440 }
    )
    foreach ($l in $lines) { $g.DrawLine((New-Object System.Drawing.Pen($maroon, 5)), $l.X1, $l.Y1, $l.X2, $l.Y2) }
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), 580, 810, 650, 110)
    $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), 580, 810, 650, 110)
    $g.DrawString("Exploratory extension in the thesis", (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($maroon)), 615, 835)
    $g.DrawString("Diffusion-policy steering appears later as a reserved methodology/results section in this defense deck.", (New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), ([System.Drawing.RectangleF]::new(615, 868, 580, 38)))
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 580, 795, 650, 10)
    Save-And-CloseCanvas -Canvas $c -Path $Path
}

function New-TaskOverviewMontage {
    param([string]$Path)
    $c = New-Canvas -Width 1800 -Height 1100
    $g = $c.Graphics
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
    $bg = [System.Drawing.Color]::FromArgb(249, 247, 243)
    $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
    $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
    $blue = [System.Drawing.Color]::FromArgb(73, 113, 153)
    $terracotta = [System.Drawing.Color]::FromArgb(176, 100, 78)
    $gray = [System.Drawing.Color]::FromArgb(70, 70, 70)
    $white = [System.Drawing.Color]::White
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
    $g.DrawString("Task Setup and Video Scenarios", (New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 48, 24)
    $tileW = 390; $tileH = 215; $left = 50; $top = 150; $gapX = 35; $gapY = 65
    for ($i = 0; $i -lt $items.Count; $i++) {
        $col = $i % 4; $row = [Math]::Floor($i / 4)
        $x = [int]($left + $col * ($tileW + $gapX)); $y = [int]($top + $row * ($tileH + $gapY))
        $accent = if ($items[$i].Type -eq "Legible") { $blue } else { $terracotta }
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), $x, $y, $tileW, $tileH)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $x, $y, $tileW, $tileH)
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($accent)), $x, $y, $tileW, 24)
        if (Test-Path $items[$i].Image) {
            $img = [System.Drawing.Image]::FromFile($items[$i].Image)
            try { $g.DrawImage($img, [System.Drawing.Rectangle]::new(($x + 8), ($y + 32), ($tileW - 16), ($tileH - 68))) }
            finally { $img.Dispose() }
        }
        $g.DrawString($items[$i].Label, (New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($gray)), ([System.Drawing.RectangleF]::new(($x + 8), ($y + $tileH - 34), ($tileW - 16), 26)))
    }
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 50, 1015, 1700, 18)
    $g.DrawString("8 videos total | 4 legible + 4 ambiguous | block-picking and drawer-closing tasks", (New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 55, 1040)
    Save-And-CloseCanvas -Canvas $c -Path $Path
}

function New-PilotFlowGraphic {
    param([string]$Path)
    $c = New-Canvas -Width 1800 -Height 1000
    $g = $c.Graphics
    $bg = [System.Drawing.Color]::FromArgb(249, 247, 243)
    $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
    $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
    $blue = [System.Drawing.Color]::FromArgb(73, 113, 153)
    $terracotta = [System.Drawing.Color]::FromArgb(176, 100, 78)
    $green = [System.Drawing.Color]::FromArgb(83, 130, 109)
    $white = [System.Drawing.Color]::White
    $gray = [System.Drawing.Color]::FromArgb(70, 70, 70)
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
    $g.DrawString("Pilot Study Design", (New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 48, 24)
    $boxes = @(
        @{ X = 90;  Y = 180; W = 250; H = 150; C = $blue;       T = "Participants"; B = "8 participants" },
        @{ X = 390; Y = 180; W = 250; H = 150; C = $terracotta; T = "Stimuli";      B = "8 robot videos" },
        @{ X = 690; Y = 180; W = 300; H = 150; C = $green;      T = "Phase A";      B = "Cumulative frame probes" },
        @{ X = 1040;Y = 180; W = 300; H = 150; C = $blue;       T = "Phase B";      B = "Full-video stop-time judgment" },
        @{ X = 1390;Y = 180; W = 300; H = 150; C = $terracotta; T = "Matched Gemini";B = "Same selected timepoints" }
    )
    foreach ($box in $boxes) {
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($box.C)), $box.X, $box.Y, $box.W, $box.H)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $box.X, $box.Y, $box.W, $box.H)
        $g.DrawString($box.T, (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($box.X + 18), ($box.Y + 20), ($box.W - 36), 30)))
        $g.DrawString($box.B, (New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($box.X + 18), ($box.Y + 72), ($box.W - 36), 45)))
    }
    for ($i = 0; $i -lt ($boxes.Count - 1); $i++) {
        $x1 = [int]($boxes[$i].X + $boxes[$i].W); $x2 = [int]$boxes[$i + 1].X; $midY = [int]($boxes[$i].Y + 75)
        $g.DrawLine((New-Object System.Drawing.Pen($maroon, 5)), ($x1 + 10), $midY, ($x2 - 24), $midY)
    }
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), 170, 470, 1460, 285)
    $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), 170, 470, 1460, 285)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 170, 470, 1460, 16)
    $g.DrawString("Recorded outputs", (New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($maroon)), 220, 525)
    $g.DrawString("Human: goal choice, confidence, cue text, and stop-time judgments", (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 220, 585)
    $g.DrawString("Gemini: structured goal predictions at exactly the same selected prefixes", (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 220, 635)
    $g.DrawString("Dataset used in the analysis: 312 human observations and 31 matched VLM predictions", (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 220, 685)
    Save-And-CloseCanvas -Canvas $c -Path $Path
}

function New-VlmPipelineMetricsGraphic {
    param([string]$Path)
    $sourcePipeline = "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\vlm_pipeline_with_frames.png"
    $c = New-Canvas -Width 1800 -Height 1100
    $g = $c.Graphics
    $bg = [System.Drawing.Color]::FromArgb(249, 247, 243)
    $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
    $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
    $blue = [System.Drawing.Color]::FromArgb(73, 113, 153)
    $green = [System.Drawing.Color]::FromArgb(83, 130, 109)
    $terracotta = [System.Drawing.Color]::FromArgb(176, 100, 78)
    $white = [System.Drawing.Color]::White
    $gray = [System.Drawing.Color]::FromArgb(70, 70, 70)
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
    $g.DrawString("VLM Goal-Inference Pipeline and Metrics", (New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 48, 24)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), 85, 125, 1630, 520)
    $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), 85, 125, 1630, 520)
    if (Test-Path $sourcePipeline) {
        $img = [System.Drawing.Image]::FromFile($sourcePipeline)
        try { $g.DrawImage($img, [System.Drawing.Rectangle]::new(105, 145, 1590, 480)) }
        finally { $img.Dispose() }
    }
    $cards = @(
        @{ X = 120;  Y = 730; W = 470; H = 220; C = $blue;       T = "Accuracy"; B = "How often the predicted goal matches the ground truth" },
        @{ X = 665;  Y = 730; W = 470; H = 220; C = $terracotta; T = "IoU Agreement"; B = "How often Gemini and human choices overlap at matched observations" },
        @{ X = 1210; Y = 730; W = 470; H = 220; C = $green;      T = "Time-to-Legibility"; B = "When the goal becomes inferable from partial motion" }
    )
    foreach ($card in $cards) {
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($card.C)), $card.X, $card.Y, $card.W, $card.H)
        $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 3)), $card.X, $card.Y, $card.W, $card.H)
        $g.DrawString($card.T, (New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($card.X + 20), ($card.Y + 24), ($card.W - 40), 38)))
        $g.DrawString($card.B, (New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($white)), ([System.Drawing.RectangleF]::new(($card.X + 20), ($card.Y + 78), ($card.W - 40), 90)))
    }
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 120, 970, 1560, 16)
    $g.DrawString("Fairness rule from the analysis: Gemini is evaluated only at the same selected timepoints shown to humans.", (New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 125, 1000)
    Save-And-CloseCanvas -Canvas $c -Path $Path
}

function New-PlaceholderGraphic {
    param([string]$Path,[string]$Title,[string]$Line1,[string]$Line2,[string]$Line3)
    $c = New-Canvas -Width 1800 -Height 1000
    $g = $c.Graphics
    $bg = [System.Drawing.Color]::FromArgb(249, 247, 243)
    $maroon = [System.Drawing.Color]::FromArgb(140, 29, 64)
    $gold = [System.Drawing.Color]::FromArgb(255, 198, 39)
    $gray = [System.Drawing.Color]::FromArgb(80, 80, 80)
    $white = [System.Drawing.Color]::White
    $g.Clear($bg)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($maroon)), 0, 0, 1800, 90)
    $g.DrawString($Title, (New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($white)), 48, 24)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($white)), 150, 170, 1500, 620)
    $g.DrawRectangle((New-Object System.Drawing.Pen($maroon, 4)), 150, 170, 1500, 620)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($gold)), 150, 170, 1500, 18)
    $g.DrawString("Reserved space in the defense deck", (New-Object System.Drawing.Font("Arial", 28, [System.Drawing.FontStyle]::Bold)), (New-Object System.Drawing.SolidBrush($maroon)), 520, 245)
    $g.DrawString($Line1, (New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 250, 390)
    $g.DrawString($Line2, (New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 250, 455)
    $g.DrawString($Line3, (New-Object System.Drawing.Font("Arial", 22, [System.Drawing.FontStyle]::Regular)), (New-Object System.Drawing.SolidBrush($gray)), 250, 520)
    $g.DrawString("This section is intentionally left open so you can add the final diffusion-policy material later.", (New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Italic)), (New-Object System.Drawing.SolidBrush($gray)), 250, 650)
    Save-And-CloseCanvas -Canvas $c -Path $Path
}

$slidesDir = Join-Path $unpacked "ppt/slides"
$slideText = @{
    "slide1.xml" = @("Vision-Language Models as Proxies for Human Judgment of Robot Motion Legibility","Anudeep Sai Gottapu | Master of Science in Technology","Arizona State University | April 2026","Nakul Gopalan, Chair | Hani Ben Amor | Kunal Garg")
    "slide2.xml" = @("Abstract","From the thesis document","Legibility in robot motion refers to how easily an observer infers a robot's intent from its movement.","This thesis examines whether a vision-language model can serve as a proxy for human judgment in evaluating robot motion legibility.","Gemini 2.5 Flash is used to analyze cumulative video frames of robot manipulation tasks and infer the intended goal.","Human pilot data and Gemini predictions are compared at matched timepoints using accuracy, IoU, and time-to-legibility.","The results suggest VLMs capture meaningful aspects of human goal inference from partial robot motion.","Humans remain more reliable overall, so the VLM is best treated as a scalable evaluation tool rather than a full replacement.")
    "slide3.xml" = @("Thesis Structure","Following the PDF table of contents","Chapter 1","Introduction","Motivation and context","Problem statement","Research Questions","Contributions","Chapter 2","Background and Related Work","Chapter 3","Methodology","Chapter 4","Results and Discussion","Chapter 5","Conclusion and Future Work")
    "slide4.xml" = @("Motivation, Problem Statement, and Contributions","Thesis framing in presentation form","Motivation","Humans infer intent from partial robot motion","Problem","Human-subject legibility studies are costly and hard to scale","Question","Can Gemini act as a practical proxy?","Dataset","8 videos | 8 participants | matched timepoints","Contribution 1","Web-based human pilot study","Contribution 2","Gemini 2.5 Flash evaluation pipeline","Contribution 3","Matched human-model comparison","Contribution 4","Accuracy, IoU, and TTL analysis")
    "slide5.xml" = @("Research Questions","Exact questions from Chapter 1.3","Slides 11-13 answer these directly.")
    "slide6.xml" = @("Overall Thesis Pipeline","How the thesis is organized for this defense","Front half","Introduction, background, and methodology","Center of the defense","Pilot study and Gemini-as-proxy results","Reserved later section","Diffusion-policy methodology and results placeholders")
    "slide7.xml" = @("Task Setup and Video Scenarios","Stimuli used across the pilot study and Gemini evaluation","8 videos total","4 legible + 4 ambiguous","Tasks","Block picking and drawer closing")
    "slide8.xml" = @("Pilot Study Design","Human study and matched Gemini protocol","Participants","8","Study phases","Cumulative frame probes and full-video stop-time judgment","Recorded outputs","Goal choice, confidence, cue text, and stop-time judgment","Matched model evaluation","Gemini queried only at the same selected human timepoints")
    "slide9.xml" = @("VLM Goal-Inference Evaluation Pipeline","Methodology used in the thesis","Input","Cumulative prefix frames at selected timepoints","Model","Gemini 2.5 Flash","Outputs","Structured goal predictions and supporting fields","Metrics","Accuracy, IoU, and time-to-legibility")
    "slide10.xml" = @("Diffusion-Policy Methodology","Reserved blank section","Keep this section open for the final defense version.")
    "slide11.xml" = @("RQ1","Can a multimodal vision-language model predict similarly to human judgment from partial robot motion?","Human accuracy: 66.7%","Gemini accuracy: 64.5%","Overall difference: not statistically significant","p = 0.966","Both human and Gemini are above chance.")
    "slide12.xml" = @("RQ2: Alignment with Human Judgments","Overall agreement: 74.2% IoU | Main outlier: amb_r_block","Agreement and failure cases")
    "slide13.xml" = @("RQ3: Practical Signal for Scalable Evaluation","Can multimodal vision-language models provide a practical signal for scalable legibility evaluation in robot manipulation tasks?","Evidence","Observation","Why it matters","Matched timepoints","Gemini and humans see the same selected prefixes","Fair proxy comparison","Comparable accuracy","66.7% human vs 64.5% Gemini","Near-human screening signal","Agreement","74.2% IoU overall","Tracks human choices in most cases","Timing sensitivity","Legible motion becomes inferable earlier than ambiguous motion","Captures legibility as an early signal","Practical answer","Yes","Useful as a scalable evaluator","Important caveat","Not a replacement","Human validation is still needed on hard cases")
    "slide14.xml" = @("Diffusion-Policy Results","Reserved blank section from the thesis flow","Qualitative rollouts","Insert final trajectory examples here","Quantitative plots","Insert task-success and legibility comparisons here","Key takeaways","Insert final conclusions for the diffusion-policy part here")
    "slide15.xml" = @("Discussion and Limitations","What these results support","Gemini is close enough to human performance to be useful as a proxy evaluator.","The strongest legibility signal is when the goal becomes clear, not only the final answer.","What these results do not support","This pilot does not justify replacing human studies.","Important limits","Small sample size, limited power per video, and clear hard cases such as amb_r_block.")
    "slide16.xml" = @("Conclusion and Future Work","Main conclusion","Vision-language models can act as useful proxies for human judgment of robot motion legibility.","Next steps","Larger human studies, more trajectories, stronger temporal prompting, and final diffusion-policy integration.","Participants","8","Human accuracy","66.7%","Gemini accuracy","64.5%","Mean IoU","74.2%","Thank you")
}

foreach ($name in $slideText.Keys) { Set-SlideText -SlideXmlPath (Join-Path $slidesDir $name) -Texts $slideText[$name] }

$rqGraphic = Join-Path $tempRoot "rq_graphic.png"
$pipelineGraphic = Join-Path $tempRoot "overall_pipeline.png"
$taskGraphic = Join-Path $tempRoot "task_montage.png"
$pilotGraphic = Join-Path $tempRoot "pilot_flow.png"
$vlmGraphic = Join-Path $tempRoot "vlm_pipeline_metrics.png"
$diffMethodGraphic = Join-Path $tempRoot "diff_method_placeholder.png"

New-ResearchQuestionsGraphic -Path $rqGraphic
New-OverallPipelineGraphic -Path $pipelineGraphic
New-TaskOverviewMontage -Path $taskGraphic
New-PilotFlowGraphic -Path $pilotGraphic
New-VlmPipelineMetricsGraphic -Path $vlmGraphic
New-PlaceholderGraphic -Path $diffMethodGraphic -Title "Diffusion-Policy Methodology" -Line1 "Add the model architecture and training setup here" -Line2 "Add the behavior-generation pipeline here" -Line3 "Add the methodology figure for the diffusion-policy part here"

Replace-MediaImage -MediaRelativePath "ppt/media/image1.png" -SourceImagePath $rqGraphic
Replace-MediaImage -MediaRelativePath "ppt/media/image2.png" -SourceImagePath $pipelineGraphic
Replace-MediaImage -MediaRelativePath "ppt/media/image3.png" -SourceImagePath $taskGraphic
Replace-MediaImage -MediaRelativePath "ppt/media/image4.png" -SourceImagePath $pilotGraphic
Replace-MediaImage -MediaRelativePath "ppt/media/image5.png" -SourceImagePath $vlmGraphic
Replace-MediaImage -MediaRelativePath "ppt/media/image6.png" -SourceImagePath $diffMethodGraphic
Replace-MediaImage -MediaRelativePath "ppt/media/image7.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\analysis_results_2\figure1_main_comparison.png"
Replace-MediaImage -MediaRelativePath "ppt/media/image8.png" -SourceImagePath "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\thesis_figures\fig5_iou_agreement.png"

if (Test-Path $outputPptx) { Remove-Item -LiteralPath $outputPptx -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipOut = Join-Path $tempRoot "rewritten.zip"
$archive = [System.IO.Compression.ZipFile]::Open($zipOut, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in (Get-ChildItem -LiteralPath $unpacked -Recurse -File)) {
        $relative = $file.FullName.Substring($unpacked.Length + 1).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $relative) | Out-Null
    }
} finally { $archive.Dispose() }

Move-Item -LiteralPath $zipOut -Destination $outputPptx
Remove-Item -LiteralPath $tempRoot -Recurse -Force
Write-Output "Created: $outputPptx"

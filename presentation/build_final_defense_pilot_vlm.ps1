$ErrorActionPreference = "Stop"

$sourcePptx = "C:\Users\anude\Downloads\HRI_Legibility_Thesis_Analysis.pptx"
$outputPptx = Join-Path $PSScriptRoot "asu_final_defense_pilot_vlm_clean.pptx"
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

$slidesDir = Join-Path $unpacked "ppt\slides"

$slideText = @{
    "slide1.xml" = @(
        "VLM as a Proxy for Human Legibility Judgment",
        "Pilot study supporting a thesis on legible robot motion",
        "Anudeep Gottapu",
        "MS Thesis Defense | Arizona State University | April 2026"
    )
    "slide2.xml" = @(
        "Where This Fits in the Thesis",
        "Why this pilot study matters",
        "Bigger thesis question",
        "Can robot motion communicate intent early enough for a person to act with it?",
        "Why this pilot study",
        "Before using a VLM to score robot motion, I wanted to check whether VLM judgments track human judgments.",
        "What I compare",
        "Human choices, VLM choices, confidence, and time-to-legibility",
        "Study scope",
        "8 participants | 8 videos | matched observation points",
        "Outcome",
        "This chapter tests whether a VLM is useful as a fast proxy, not a replacement for human evaluation.",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide3.xml" = @(
        "Study Questions",
        "What I wanted to learn from the pilot",
        "Q1",
        "Do legible videos become readable earlier than ambiguous videos?",
        "Q2",
        "Is VLM accuracy close to human accuracy on the same clips?",
        "Q3",
        "When do humans and the VLM agree or diverge?",
        "Q4",
        "Does the VLM reach a correct inference earlier than humans?",
        "Q5",
        "Does confidence rise when the observer is actually correct?",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide4.xml" = @(
        "Pilot Study at a Glance",
        "Human-VLM comparison on matched legibility judgments",
        "Human accuracy",
        "63.8%",
        "Legible",
        "Human accuracy",
        "59.4%",
        "Ambiguous",
        "VLM accuracy",
        "66.7%",
        "Legible",
        "VLM accuracy",
        "62.5%",
        "Ambiguous",
        "Human TTL",
        "6.45s",
        "Legible",
        "Human TTL",
        "7.90s",
        "Ambiguous",
        "VLM TTL",
        "3.25s",
        "Legible",
        "VLM TTL",
        "6.75s",
        "Ambiguous",
        "Mean agreement",
        "73.8%",
        "IoU",
        "Agreement on legible videos",
        "77.8%",
        "IoU",
        "Agreement on ambiguous videos",
        "69.8%",
        "IoU",
        "VLM faster in",
        "75%",
        "6 / 8 videos",
        "N = 8 participants",
        "8 videos (4 legible, 4 ambiguous)",
        "248 human observations | 31 matched VLM evaluations"
    )
    "slide5.xml" = @(
        "Accuracy by Trajectory Type",
        "Human and VLM final accuracy are close overall",
        "Main readout",
        "Final accuracy does not separate legible and ambiguous videos strongly.",
        "Human",
        "63.8% on legible | 59.4% on ambiguous",
        "VLM",
        "66.7% on legible | 62.5% on ambiguous",
        "Takeaway",
        "Accuracy alone is not the best lens for legibility.",
        "Why",
        "Both observers usually catch up by the end of the clip.",
        "So the better question becomes",
        "How early does the correct inference emerge?",
        "Pilot-study interpretation",
        "Use timing and agreement metrics, not just endpoint accuracy.",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide6.xml" = @(
        "Temporal Dynamics: Key Result",
        "Legibility shows up in how fast inference improves",
        "Timeline pattern",
        "t0 (baseline)",
        "Legible 9.4% ± 12.0",
        "Ambiguous 12.5% ± 10.2",
        "t1 (early)",
        "Legible 59.4% ± 49.3",
        "Ambiguous 37.5% ± 42.1",
        "t2 (mid)",
        "Legible 100% ± 0.0",
        "Ambiguous 87.5% ± 25.0",
        "t3 (final)",
        "Legible 91.7% ± 14.4",
        "Ambiguous 100% ± 0.0",
        "Key takeaway",
        "The legibility effect is mainly early, not at the final frame.",
        "",
        "",
        ""
    )
    "slide7.xml" = @(
        "Confidence Tracks Correctness",
        "Confidence is informative, but it does not separate video type much",
        "Result",
        "Participants were more confident when they were correct.",
        "Correct choices",
        "M = 9.70 | SD = 0.95",
        "Incorrect choices",
        "M = 8.25 | SD = 3.44",
        "Interpretation",
        "Confidence is meaningful as a quality signal.",
        "But",
        "Confidence alone does not tell us whether a trajectory is legible.",
        "Legible videos",
        "M = 9.18",
        "Ambiguous videos",
        "M = 9.11",
        "Practical use",
        "Confidence can complement accuracy and timing in the evaluator.",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide8.xml" = @(
        "Time to Legibility: VLM vs Human",
        "The VLM usually reaches a correct inference earlier",
        "Key numbers",
        "Human legible",
        "6.45s | SD = 1.43s",
        "Human ambiguous",
        "7.90s | SD = 1.80s",
        "VLM legible",
        "3.25s | SD = 1.71s",
        "VLM ambiguous",
        "6.75s | SD = 5.12s",
        "Readout",
        "The VLM is faster than the human mean on 6 of 8 videos.",
        "Legibility effect",
        "VLM: 3.25s vs 6.75s",
        "Human: 6.45s vs 7.90s",
        "Takeaway",
        "The VLM preserves the direction of the human temporal effect.",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide9.xml" = @(
        "Human-VLM Agreement",
        "Agreement is strong overall, with one clear failure case",
        "Result",
        "Overall mean IoU = 73.8% | SD = 16.5%",
        "Legible videos",
        "77.8% | SD = 12.3%",
        "Ambiguous videos",
        "69.8% | SD = 21.1%",
        "Highest agreement",
        "amb_l_block | 95.8%",
        "le_l_block | 94.4%",
        "Lowest agreement",
        "amb_r_block | 45.8%",
        "Interpretation",
        "Most videos show strong alignment between human and VLM judgments.",
        "Failure case",
        "The right-block ambiguous case exposes a limit of the proxy.",
        "Why it matters",
        "The VLM is useful for screening and ranking, but edge cases still need human review.",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide10.xml" = @(
        "Participant-to-Participant Variation",
        "Agreement varies across people, but the overall pattern is still consistent",
        "Individual Differences",
        "Range: 64.5% to 80.6% IoU across participants",
        ""
    )
    "slide11.xml" = @(
        "Manipulation Check",
        "Do the labeled trajectory types behave differently for observers?",
        "Result",
        "Final accuracy alone does not separate legible and ambiguous videos strongly.",
        "Legible videos",
        "M = 63.8% | SD = 13.9%",
        "Ambiguous videos",
        "M = 59.4% | SD = 12.8%",
        "Important nuance",
        "The difference appears in timing, not in saturated final accuracy.",
        "Early timepoint",
        "Legible 59.4% vs Ambiguous 37.5%",
        "Final timepoint",
        "Both conditions approach ceiling",
        "Interpretation",
        "That pattern is exactly what we expect from a legibility manipulation.",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide12.xml" = @(
        "Pilot Study Dashboard",
        "Summary view of the human-VLM comparison",
        ""
    )
    "slide13.xml" = @(
        "Statistics and Practical Reading",
        "I treat the pilot as directional evidence, not a final validation",
        "Question",
        "Measure",
        "Test",
        "Result",
        "Read",
        "Q1",
        "Human accuracy by type",
        "t-test",
        "p = .655",
        "Small difference; underpowered",
        "Q2",
        "VLM accuracy by type",
        "t-test",
        "p = .773",
        "Small difference; underpowered",
        "Q3",
        "VLM vs human accuracy",
        "t-test",
        "p = .764",
        "Comparable in this pilot",
        "Q4",
        "Agreement",
        "IoU = 73.8%",
        "Practical threshold",
        "Proxy is promising",
        "Q5",
        "Time-to-legibility",
        "Descriptive",
        "3.25s vs 6.75s",
        "Clear temporal separation",
        "Q6",
        "Confidence vs correctness",
        "Mann-Whitney U",
        "p < .001",
        "Confidence is meaningful",
        "Reading the stats",
        "Small sample",
        "8 participants and 8 videos",
        "Best use",
        "Directional evidence and failure-case discovery",
        "Not yet",
        "A replacement for a larger confirmatory human study",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide14.xml" = @(
        "How This Pilot Supports the Full Thesis",
        "Why the VLM chapter matters even though the thesis is broader",
        "Full thesis arc",
        "1. Learn a policy that can generate multi-goal motion",
        "2. Test whether a VLM tracks human legibility judgments",
        "3. Use the VLM to score or rank generated trajectories at scale",
        "What this defense version emphasizes",
        "The human-study validation and the VLM-as-proxy results",
        "What I am keeping light here",
        "Detailed diffusion-policy methodology and full policy results",
        "Why",
        "For the defense talk, the strongest evidence is the human/VLM bridge.",
        "If asked",
        "I can expand on policy training, guidance, and rollout evaluation.",
        "",
        "",
        "",
        "",
        ""
    )
    "slide15.xml" = @(
        "Takeaways, Limits, and Next Steps",
        "What this pilot lets us claim, and what it does not",
        "Main takeaways",
        "VLM and human endpoint accuracy are similar in this pilot.",
        "The stronger signal is temporal: legible motions become readable earlier.",
        "Agreement is high overall, with one informative failure case.",
        "What I can claim",
        "The VLM is a useful proxy for screening, ranking, and rapid iteration.",
        "What I cannot claim yet",
        "That the VLM fully replaces human evaluation in all settings.",
        "Current limits",
        "Small sample, online study, and only 8 videos",
        "Next step",
        "Run a larger human study while using the VLM to scale analysis",
        "Proxy principle",
        "Use humans for validation and VLMs for throughput.",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )
    "slide16.xml" = @(
        "Conclusion",
        "What this pilot contributes to the thesis",
        "Participants",
        "N = 8",
        "Mean IoU",
        "73.8%",
        "VLM faster in",
        "75%",
        "Key message",
        "The VLM tracks human judgments well enough to be useful as a legibility proxy.",
        "Use in thesis",
        "This supports using VLM-based evaluation in the larger legible-motion pipeline.",
        "",
        ""
    )
}

foreach ($name in $slideText.Keys) {
    $path = Join-Path $slidesDir $name
    Set-SlideText -SlideXmlPath $path -Texts $slideText[$name]
}

if (Test-Path $outputPptx) {
    Remove-Item -LiteralPath $outputPptx -Force
}

$zipOut = Join-Path $tempRoot "rewritten.zip"
Add-Type -AssemblyName System.IO.Compression.FileSystem
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

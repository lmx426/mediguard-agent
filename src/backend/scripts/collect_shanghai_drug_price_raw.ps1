param(
  [string]$Python = "python",
  [string]$ProjectRoot = (Resolve-Path ".").Path,
  [string]$KeywordFile = "",
  [string]$SnapshotDate = (Get-Date -Format "yyyyMMdd"),
  [int]$PageSize = 20,
  [int]$MaxPages = 1,
  [int]$LimitKeywords = 0
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"

$origin = "https://bjxt.smiic.net.cn"
$searchUrl = "$origin/ypcx/?sessionid=#/pages/drugs-query/search"
$serviceBase = "$origin/hsa-mbs-pub/api/v1/main"
$scriptPath = Join-Path $ProjectRoot "src\backend\scripts\shanghai_drug_price_crypto.py"
$referenceDir = Join-Path $ProjectRoot "src\backend\policy_corpus\rag_ready\reference\drug_product_price_reference"
if (-not $KeywordFile) {
  $KeywordFile = Join-Path $referenceDir "shanghai_drug_price_keywords_v1.txt"
}
$rawOutput = Join-Path $referenceDir "shanghai_drug_price_raw_$SnapshotDate.jsonl"
$issueOutput = Join-Path $referenceDir "shanghai_drug_price_raw_${SnapshotDate}_issues.jsonl"
$tmpDir = Join-Path $ProjectRoot "tmp\shanghai_drug_price_collect\$SnapshotDate"

New-Item -ItemType Directory -Force -Path $referenceDir | Out-Null
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
if (Test-Path -Path $rawOutput) { Remove-Item -Path $rawOutput -Force }
if (Test-Path -Path $issueOutput) { Remove-Item -Path $issueOutput -Force }

$htmlPath = Join-Path $tmpDir "index.html"
$indexJsPath = Join-Path $tmpDir "index.js"
$commonJsPath = Join-Path $tmpDir "common.js"
$keysPath = Join-Path $tmpDir "keys.json"
$sessionPath = Join-Path $tmpDir "session.json"

$commonHeaders = @(
  "-H", "Referer: $searchUrl",
  "-H", "Origin: $origin",
  "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
  "-H", "Accept: application/json;charset=UTF-8",
  "-H", "Content-Type: application/json"
)

function Invoke-CurlToFile {
  param(
    [string]$Url,
    [string]$OutputPath,
    [string[]]$ExtraArgs = @()
  )
  $content = & curl.exe -sS -L $Url @ExtraArgs
  if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE for $Url"
  }
  Set-Content -Path $OutputPath -Value $content -Encoding UTF8
}

Invoke-CurlToFile -Url $searchUrl -OutputPath $htmlPath
$html = Get-Content -Path $htmlPath -Raw -Encoding UTF8
if ($html -notmatch 'src="/ypcx/(?<asset>assets/index-[^"]+\.js)"') {
  throw "Cannot find index JS asset in H5 HTML."
}
$indexAsset = $Matches["asset"]
Invoke-CurlToFile -Url "$origin/ypcx/$indexAsset" -OutputPath $indexJsPath
$indexJs = Get-Content -Path $indexJsPath -Raw -Encoding UTF8
if ($indexJs -notmatch '"(?<asset>assets/common\.[^"]+\.js)"') {
  throw "Cannot find common JS asset in H5 index JS."
}
$commonAsset = $Matches["asset"]
Invoke-CurlToFile -Url "$origin/ypcx/$commonAsset" -OutputPath $commonJsPath

Invoke-CurlToFile -Url "$serviceBase/keys" -OutputPath $keysPath -ExtraArgs $commonHeaders
& $Python $scriptPath prepare-session --keys-response $keysPath --common-js $commonJsPath --session-output $sessionPath --service-base $serviceBase
$session = Get-Content -Path $sessionPath -Raw -Encoding UTF8 | ConvertFrom-Json
$clientId = $session.client_id

$keywords = @(Get-Content -Path $KeywordFile -Encoding UTF8 | Where-Object {
  $item = $_.Trim()
  $item -and -not $item.StartsWith("#")
})
if ($LimitKeywords -gt 0) {
  $keywords = @($keywords | Select-Object -First $LimitKeywords)
}
if ($keywords.Count -gt 0) {
  Write-Host "[sh-drug-raw] first_keyword=$($keywords[0])"
}

$total = $keywords.Count * $MaxPages
$done = 0
foreach ($keyword in $keywords) {
  for ($pageNum = 1; $pageNum -le $MaxPages; $pageNum++) {
    $done += 1
    $requestPath = Join-Path $tmpDir ("request_{0:D4}_{1}.json" -f $done, $pageNum)
    $responsePath = Join-Path $tmpDir ("response_{0:D4}_{1}.json" -f $done, $pageNum)
    & $Python $scriptPath prepare-request --session $sessionPath --keyword $keyword --page-num $pageNum --page-size $PageSize --request-output $requestPath
    Write-Host "[sh-drug-raw] $done/$total keyword=$keyword page=$pageNum"
    $postArgs = @("-X", "POST", "-H", "Client-ID: $clientId") + $commonHeaders + @("--data-binary", "@$requestPath")
    Invoke-CurlToFile -Url "$serviceBase/operateRsa" -OutputPath $responsePath -ExtraArgs $postArgs
    & $Python $scriptPath decode-response --session $sessionPath --response $responsePath --raw-output $rawOutput --issue-output $issueOutput --keyword $keyword --page-num $pageNum --page-size $PageSize
    Start-Sleep -Milliseconds 300
  }
}

$rawCount = 0
if (Test-Path -Path $rawOutput) {
  $rawCount = (Get-Content -Path $rawOutput -Encoding UTF8 | Measure-Object -Line).Lines
}
$issueCount = 0
if (Test-Path -Path $issueOutput) {
  $issueCount = (Get-Content -Path $issueOutput -Encoding UTF8 | Measure-Object -Line).Lines
}

@{
  status = "ok"
  keyword_count = $keywords.Count
  page_size = $PageSize
  max_pages = $MaxPages
  raw_count = $rawCount
  issue_count = $issueCount
  raw_output = $rawOutput
  issue_output = $issueOutput
} | ConvertTo-Json -Depth 5

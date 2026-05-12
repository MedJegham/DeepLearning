$env:GIT_AUTHOR_NAME = "MedJegham"
$env:GIT_AUTHOR_EMAIL = "mje.etudes@gmail.com"
$env:GIT_COMMITTER_NAME = "MedJegham"
$env:GIT_COMMITTER_EMAIL = "mje.etudes@gmail.com"

& git add -A

$tree = (& git write-tree).Trim()
Write-Host "TREE: $tree"

$parent = (& git rev-parse HEAD).Trim()
Write-Host "PARENT: $parent"

$msg = "docs: nudge contributors cache"
$newsha = (& git commit-tree $tree -p $parent -m $msg).Trim()
Write-Host "NEW SHA: $newsha"

& git update-ref refs/heads/main $newsha
& git log --format="%h | %an <%ae> | %s | body=[%b]" -2
& git push --force origin main

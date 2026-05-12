$env:GIT_AUTHOR_NAME = "MedJegham"
$env:GIT_AUTHOR_EMAIL = "mje.etudes@gmail.com"
$env:GIT_COMMITTER_NAME = "MedJegham"
$env:GIT_COMMITTER_EMAIL = "mje.etudes@gmail.com"

# Force-remove the helper script from index
& git rm --cached -f -- _finalize.ps1 _clean_commit.ps1 2>$null
& git add -A
$tree = (& git write-tree).Trim()
$parent = (& git rev-parse HEAD).Trim()
$msg = "chore: remove helper script"
$newsha = (& git commit-tree $tree -p $parent -m $msg).Trim()
Write-Host "NEW SHA: $newsha"
& git update-ref refs/heads/main $newsha
& git log --format="%h | %an <%ae> | %s" -3
& git push --force origin main

set d=%date:~0,10%
set t=%time:~0,8%
set times=%d% %t%

echo current directory:%cd%
git status 
git add .
set /p declation=commit message:
git commit -m "%times% %declation%"
git push origin master
exit
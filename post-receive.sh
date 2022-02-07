echo $(pwd)
echo Start of post-receive

source /etc/profile
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

git fetch git@github.com:JeanHaiz/Polytopia.git master

HEADHASH=$(git rev-parse FETCH_HEAD)
UPSTREAMHASH=$(git rev-parse master@{upstream})

echo $HEADHASH $UPSTREAMHASH

if [ "$HEADHASH" != "$UPSTREAMHASH" ]
then
	echo Not up to date with origin. Pulling and Restarting.
	git pull git@github.com:JeanHaiz/Polytopia.git master
	chmod +x /Users/jean/Documents/Coding/Polytopia/post-receive.sh
	docker-compose down
	docker-compose up --detach helper
	crontab crontab.txt
	echo End of Pulling
else
	echo Current branch is up to date with origin/master.
fi

echo End of post-receive

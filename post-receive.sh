echo $(pwd)
echo Start of post-receive

echo $(git fetch)

HEADHASH=$(git rev-parse HEAD)
UPSTREAMHASH=$(git rev-parse master@{upstream})

echo $HEADHASH $UPSTREAMHASH

if [ "$HEADHASH" != "$UPSTREAMHASH" ]
then
	echo Not up to date with origin. Pulling and Restarting.
	echo $(git pull)
	echo $(chmod +x /Users/jean/Documents/Coding/Polytopia/post-receive.sh)
	echo $(docker-compose down)
	echo $(docker-compose up --detach helper)
	echo $(crontab crontab.txt)
	echo End of Pulling
else
	echo Current branch is up to date with origin/master.
fi

echo End of post-receive

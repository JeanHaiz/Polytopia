echo Start of post-receive

git fetch
HEADHASH=$(git rev-parse HEAD)
UPSTREAMHASH=$(git rev-parse master@{upstream})

if [ "$HEADHASH" != "$UPSTREAMHASH" ]
then
	echo Not up to date with origin. Pulling & Restarting.
	git pull
	chmod +x /Users/jean/Documents/Coding/Polytopia/post-receive.sh
	docker-compose down
	docker-compose up --build --detach helper
	crontab crontab.txt
else
	echo Current branch is up to date with origin/master.
fi

echo End of post-receive
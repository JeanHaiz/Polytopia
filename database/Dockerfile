FROM postgres:14.1
ADD . / /docker-entrypoint-initdb.d/
RUN chmod a+r /docker-entrypoint-initdb.d/*
EXPOSE 5432
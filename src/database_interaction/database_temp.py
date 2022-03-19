import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, BIGINT, VARCHAR, TIMESTAMP,BOOLEAN, INT
from sqlalchemy.dialects.postgresql import UUID
import uuid
import time

Base = declarative_base()

adminUser = "postgres"
host = "localhost"
password = "password123"
port = 5432
database = "polytopiahelper_dev_temp"

engine = create_engine(f"postgresql://{adminUser}:{password}@{host}:{port}/{database}", echo=True)

class discord_server(Base):
    __tablename__ = "discord_server"

    server_discord_id  = Column(BIGINT, primary_key=True, unique=True)
    server_name = Column(VARCHAR(40))

class discord_channel(Base):
    __tablename__ = "discord_channel"

    channel_discord_id  = Column(BIGINT, primary_key=True, unique=True)
    server_discord_id = Column(BIGINT, nullable=False)
    channel_name = Column(VARCHAR(40))
    date_added = Column(TIMESTAMP, nullable=False,server_default=time.ctime())
    is_active = Column(BOOLEAN, default=0)

    sqlalchemy.ForeignKeyConstraint(["server_discord_id"],["discord_server.server_discord_id"])

class polytopia_game(Base):
    __tablename__ = "polytopia_game"

    channel_discord_id  = Column(BIGINT, primary_key=True, unique=True)
    server_discord_id = Column(BIGINT, nullable=False)
    game_name  = Column(VARCHAR(40))
    n_players = Column(INT)
    map_size = Column(INT)
    latest_turn = Column(INT,default=-1)

class polytopia_player(Base):
    __tablename__ = "polytopia_player"

    game_player_uuid = Column(UUID, primary_key=True, default=uuid.uuid4(), nullable=False) 
    discord_player_id = Column(BIGINT, unique=True)
    discord_player_name = Column(VARCHAR(40))
    polytopia_player_name = Column(VARCHAR(40))
    


class game_players(Base):
    __tablename__ = "game_players"

    game_player_uuid = Column(VARCHAR(40), primary_key=True)
    channel_discord_id  = Column(BIGINT, nullable=False)
    is_alive = Column(BOOLEAN)

class game_player_scores(Base):
    __tablename__ = "game_player_scores" #unfinished

    score_uuid = Column(UUID, primary_key=True, default=uuid.uuid4, nullable=False)
    channel_discord_id  = Column(BIGINT, nullable=False)
    turn = Column(INT)
    score = Column(INT)
    confirmed = Column(BOOLEAN)

class message_resources(Base):
    __tablename__  = "message_resources" #unfinished

    filename = Column(UUID, primary_key=True, default=uuid.uuid4(), nullable=False)
    source_channel__id  = Column(BIGINT, nullable=False)
    source_message__id  = Column(BIGINT, nullable=False)
    author_id = Column(BIGINT)
    operation = Column(INT)
    date_added = Column(TIMESTAMP, nullable=False)

tempPassword = "password123"
user = "testuser"
schema = "schema_temp"

        
def new_db():
    engine = create_engine(f"postgresql://{user}:{tempPassword}@{host}:{port}/{database}", echo=True)
    engine.execute(f"SET search_path={schema};")
    meta = sqlalchemy.MetaData(engine)
    meta.drop_all()
    Base.metadata.create_all(engine)
    engine.connect()
    return engine


new_db()
        


from io import BytesIO
from typing import List
from typing import Optional
from typing import Callable
from typing import Coroutine
from typing import Any
from typing import Union
from interactions import Channel, LibraryException
from interactions import Message

from common import image_utils
from database.database_client import DatabaseClient
from common.image_operation import ImageOp


async def get_or_prepare_file_inputs(
        database_client: DatabaseClient,
        channel: Channel,
        image_op: Union[ImageOp, List[ImageOp]]
) -> List[str]:
    
    # dict: source_message_id, operation
    resource_messages = database_client.get_channel_resource_messages(
        channel.id,
        image_op
    )
    
    messages = {}
    
    for rm_i in resource_messages:
        message_i_id = rm_i["source_message_id"]
        if message_i_id in messages:
            messages[message_i_id].append(rm_i)  # TODO check this works
        else:
            messages[message_i_id] = [rm_i]
    
    image_inputs = []
    
    for message_i_id, message_i_message_resources in messages.items():
        
        for message_resource_i in message_i_message_resources:
            resource_number = message_resource_i["resource_number"]
            
            resource_i = database_client.get_resource(message_i_id, resource_number)
            
            if resource_i is None:
                message_i = await channel.get_message(message_i_id)
                
                print("message_i", message_i, flush=True)
                
                check = await get_or_register_input(
                    database_client,
                    lambda i: message_i.attachments[i].download(),
                    channel,
                    message_i_id,
                    message_i,
                    resource_number,
                    image_op
                )
                
                if check is None:
                    continue
                
                resource_i = database_client.get_resource(
                    message_i_id,
                    resource_number
                )
            
            filename = str(resource_i["filename"])
            
            image_inputs.append(filename)
    
    return image_inputs


async def get_or_register_input(
        database_client: DatabaseClient,
        download_fct: Callable[[int], Coroutine[Any, Any, BytesIO]],
        channel: Channel,
        message_id: int,
        message: Optional[Message],  # TODO remove as only used for author, author could be passed
        resource_number: int,
        image_op: Union[ImageOp, List[ImageOp]]
) -> Optional[str]:
    
    resource = database_client.get_resource(message_id, resource_number)

    if message is None:  # TODO remove message
        print("loading message: %d" % message_id, flush=True)
        try:
            message = await channel.get_message(message_id)
        except LibraryException as e:
            print(e, flush=True)
            return None  # TODO exploit error

    if resource is None:

        filename = database_client.add_resource(
            channel.guild_id,
            channel.id,
            message_id,
            message.author.id,
            message.author.username,
            ImageOp.INPUT,
            resource_number
        )
        print("filename", filename)
        operation = ImageOp.INPUT

    else:
        filename = str(resource["filename"])
        operation = ImageOp(int(resource["operation"]))

    if hasattr(image_op, '__iter__'):  # image_op is a sequence
        for image_op_i in image_op:
            check = await image_utils.get_or_fetch_image_check(
                database_client,
                download_fct,
                channel.name,
                message_id,
                filename,
                image_op_i
            )
            if check:
                break
    else:
        check = await image_utils.get_or_fetch_image_check(
            database_client,
            download_fct,
            channel.name,
            message_id,
            filename,
            image_op
        )
        if check and operation != str(image_op.value):
            print("setting new image operation", operation, str(image_op.value), flush=True)
            database_client.set_resource_operation(message_id, image_op, resource_number)
    
    return filename

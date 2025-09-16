import asyncio
from aiogram import Bot

from taskiq_broker.broker import broker
from taskiq import TaskiqDepends


@broker.task()
async def simple_task_1(user_id, text, bot: Bot = TaskiqDepends()):
    print('Simple task 2 is running')
    await asyncio.sleep(2)
    print('Simple task 2 is done')
    await bot.send_message(user_id, text)
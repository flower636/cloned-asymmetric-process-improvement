@router.message(CommandStart())
async def get_phone(message: Message, bot: Bot, session_maker_db: Session,
                    user_dict: dict, agcm,
                    scheduler: ContextSchedulerDecorator, config: Config):
...
<<остальной код>>
...

text = 'Тест'
await simple_task_1.kiq(user_id, text)
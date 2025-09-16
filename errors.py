[2025-09-16 09:52:45,620][asyncio][ERROR  ][worker-1] Task exception was never retrieved
future: <Task finished name='Task-35' coro=<Receiver.callback() done, defined at C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\taskiq\receiver\receiver.py:87> exception=PydanticSchema
GenerationError("Unable to generate pydantic-core schema for <class 'aiogram.client.bot.Bot'>. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_pydantic_core_schema__` on you
r type to fully support it.\n\nIf you got this error by calling handler(<some type>) within `__get_pydantic_core_schema__` then you likely need to call `handler.generate_schema(<some type>)` since we do not call `__get_py
dantic_core_schema__` on `<some type>` otherwise to avoid infinite recursion.")>
Traceback (most recent call last):
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\type_adapter.py", line 270, in _init_core_attrs
    self.core_schema = _getattr_no_parents(self._type, '__pydantic_core_schema__')
                       ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\type_adapter.py", line 55, in _getattr_no_parents
    raise AttributeError(attribute)
AttributeError: __pydantic_core_schema__

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\taskiq\receiver\receiver.py", line 148, in callback
    result = await self.run_task(
             ^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\taskiq\receiver\receiver.py", line 218, in run_task
    parse_params(signature, self.task_hints.get(message.task_name) or {}, message)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\taskiq\receiver\params_parser.py", line 77, in parse_params
    message.args[argnum] = parse_obj_as(annot, value)
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\taskiq\compat.py", line 22, in parse_obj_as
    return create_type_adapter(annot).validate_python(obj)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\taskiq\compat.py", line 19, in create_type_adapter
    return pydantic.TypeAdapter(annot)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\type_adapter.py", line 227, in __init__
    self._init_core_attrs(
    ~~~~~~~~~~~~~~~~~~~~~^
        ns_resolver=_namespace_utils.NsResolver(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        force=False,
        ^^^^^^^^^^^^
    )
    ^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\type_adapter.py", line 289, in _init_core_attrs
    core_schema = schema_generator.generate_schema(self._type)
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 711, in generate_schema
    schema = self._generate_schema_inner(obj)
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 1009, in _generate_schema_inner
    return self.match_type(obj)
           ~~~~~~~~~~~~~~~^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 1127, in match_type
    return self._unknown_type_schema(obj)
           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "C:\Users\DOrlov\WorkDirectory\Projects\renident-telegram-bot\venv\Lib\site-packages\pydantic\_internal\_generate_schema.py", line 639, in _unknown_type_schema
    raise PydanticSchemaGenerationError(
    ...<7 lines>...
    )
pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'aiogram.client.bot.Bot'>. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_
pydantic_core_schema__` on your type to fully support it.

If you got this error by calling handler(<some type>) within `__get_pydantic_core_schema__` then you likely need to call `handler.generate_schema(<some type>)` since we do not call `__get_pydantic_core_schema__` on `<some
 type>` otherwise to avoid infinite recursion.

For further information visit https://errors.pydantic.dev/2.11/u/schema-for-unknown-type
[2025-09-16 09:52:45,721][taskiq.receiver.receiver][INFO   ][worker-1] Executing task tasks.tasks:simple_task_1 with ID: a33579e2b52a4435a9db62c64f7cd2e9
# Minimal Per‑User Memory Chat

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=YOUR_KEY
```

## Chat (CLI)

```
python chat_cli.py <user_id> "<prompt>"
# Example
python chat_cli.py u1 "I prefer matcha lattes over coffee."
```

Please use "exit" to end the conversation because I am using it to save the memory in "session" mem_mode, you can interact in "user" mode using this command. 
```
python chat_cli.py <user_id> "<prompt>" --mem-mode "user"
```

## Manage Memories
```
python mem_cli.py list <user_id>
python mem_cli.py delete <user_id> --id 12
python mem_cli.py clear <user_id> --yes
```
